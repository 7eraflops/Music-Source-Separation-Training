import functools
import os
import warnings

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data.distributed import DistributedSampler

from train import train_one_epoch
from utils.dataset import MSSDataset
from utils.losses import choice_loss
from utils.model_utils import (
    get_lora,
    get_optimizer,
    save_last_weights,
    save_weights,
)
from utils.settings import (
    cleanup_ddp,
    get_model_from_config,
    get_scheduler,
    initialize_environment_ddp,
    parse_args_train,
    wandb_init,
)
from valid import valid_multi_gpu

# Import model layers for FSDP wrapping and checkpointing
from models.demucs4ht import HEncLayer, HDecLayer
try:
    from demucs.transformer import CrossTransformerEncoder
except ImportError:
    CrossTransformerEncoder = None
from models.demucs4ht_internal_fusion import CrossTransformerEncoderWithLatents

warnings.filterwarnings("ignore")


def train_model_fsdp(rank: int, world_size: int, args=None):
    """
    Train model using FSDP (Fully Sharded Data Parallel) to shard model across GPUs.
    This allows pooling memory across GPUs (e.g., 2x40GB = 80GB effective).

    Args:
        rank: GPU rank
        world_size: Total number of GPUs
        args: Training arguments
    """
    # Initialize distributed environment
    initialize_environment_ddp(rank, world_size, args.seed, args.results_path)

    should_print = rank == 0

    # Load config and model
    if should_print:
        print("Loading model configuration...")

    model, config = get_model_from_config(args.model_type, args.config_path)
    model = get_lora(args, config, model)

    # Setup device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Configure FSDP wrapping policy
    # We use ModuleWrapPolicy to specifically target model layers
    # This ensures they are sharded individually
    wrap_classes = {HEncLayer, HDecLayer, CrossTransformerEncoderWithLatents}
    if CrossTransformerEncoder is not None:
        wrap_classes.add(CrossTransformerEncoder)
    
    # Also include local transformer layers if we can access them
    # For now, wrapping the high-level blocks is usually sufficient
    
    fsdp_wrap_policy = ModuleWrapPolicy(wrap_classes)
    
    if should_print:
        print(f"FSDP Wrapping Policy targets: {[c.__name__ for c in wrap_classes]}")

    # Configure mixed precision
    # Use FP32 for parameters to avoid cuFFT BFloat16 incompatibility with STFT
    # Only cast gradients and reduce operations to BF16/FP16 for memory savings
    if torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.float32,
        )
        if should_print:
            print("Using mixed precision: FP32 params, BF16 reduce (cuFFT compatible)")
    else:
        mp_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float32,
        )
        if should_print:
            print("Using mixed precision: FP32 params, FP16 reduce (cuFFT compatible)")

    # Move model to device before FSDP wrapping
    model = model.to(device)

    # Wrap model with FSDP
    # FULL_SHARD: Shard parameters, gradients, and optimizer states across all GPUs
    if should_print:
        print("Wrapping model with FSDP (FULL_SHARD strategy)...")

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=fsdp_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,  # Better compatibility with optimizers
    )

    # Apply Activation Checkpointing (Gradient Checkpointing)
    # This is critical for reducing memory usage by recomputing activations during backward pass
    if should_print:
        print("Applying activation checkpointing to Transformer blocks...")
        
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    # Checkpoint the heavy Transformer blocks
    # We can also checkpoint HEncLayer/HDecLayer if memory is still tight
    checkpoint_classes = {CrossTransformerEncoderWithLatents}
    if CrossTransformerEncoder is not None:
        checkpoint_classes.add(CrossTransformerEncoder)
        
    check_fn = lambda submodule: isinstance(submodule, tuple(checkpoint_classes))
    
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    if should_print:
        print("Model wrapped with FSDP and Checkpointing successfully")
        print(f"Model: {model.__class__.__name__}")

    # Load checkpoint if specified
    checkpoint = None
    if args.start_check_point:
        from utils.model_utils import load_start_checkpoint

        checkpoint = torch.load(
            args.start_check_point, weights_only=False, map_location="cpu"
        )
        load_start_checkpoint(args, model, checkpoint, type_="train")

    # Setup optimizer
    optimizer = get_optimizer(config, model)

    # Setup scheduler
    scheduler = get_scheduler(config, optimizer)

    # Load optimizer/scheduler state if available
    if args.start_check_point and checkpoint:
        if "optimizer_state_dict" in checkpoint and args.load_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and args.load_scheduler:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Setup training state
    start_epoch = 0
    best_metric = float("-inf")
    all_time_all_metrics = {}
    all_losses = {}

    if args.start_check_point and checkpoint:
        if "epoch" in checkpoint and args.load_epoch:
            start_epoch = checkpoint["epoch"] + 1
        if "best_metric" in checkpoint and args.load_best_metric:
            best_metric = checkpoint["best_metric"]
        if "all_metrics" in checkpoint and args.load_all_metrics:
            all_time_all_metrics = checkpoint["all_metrics"]
        if "all_losses" in checkpoint and args.load_all_losses:
            all_losses = checkpoint["all_losses"]

    # Setup datasets
    if should_print:
        print("Setting up datasets...")

    train_dataset = MSSDataset(
        config,
        args.data_path,
        metadata_path=os.path.join(args.results_path, "metadata_train.pkl"),
        dataset_type=args.dataset_type,
        batch_size=config.training.batch_size,
        latents_path=args.latents_path,
        verbose=True,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers if args.num_workers else 0,
        pin_memory=args.pin_memory if args.pin_memory else False,
        persistent_workers=args.persistent_workers
        if args.num_workers > 0 and args.persistent_workers
        else False,
        prefetch_factor=args.prefetch_factor
        if args.num_workers > 0 and args.prefetch_factor
        else None,
    )

    # Setup gradient accumulation
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # Setup AMP
    use_amp = getattr(config.training, "use_amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Setup loss function
    multi_loss = choice_loss(args, config)

    # Memory management
    if args.set_per_process_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(1.0)
    torch.cuda.empty_cache()

    # Initialize wandb
    if should_print:
        batch_size = config.training.batch_size
        wandb_init(args, config, batch_size)

    # Print training info
    if should_print:
        print(f"Instruments: {config.training.instruments}")
        print(
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}"
        )
        print(
            f"Patience: {config.training.patience} Reduce factor: {config.training.reduce_factor}"
        )
        print(
            f"Batch size: {batch_size} Grad accum steps: {gradient_accumulation_steps} Num gpus: {world_size} Effective batch size: {batch_size * gradient_accumulation_steps * world_size}"
        )
        print(f"Dataset type: {args.dataset_type}")
        print(f"Optimizer: {config.training.optimizer}")
        print(f"Train for: {config.training.num_epochs} epochs")
        print(f"FSDP Strategy: FULL_SHARD (model sharded across {world_size} GPUs)")
        print(f"Effective GPU memory: {world_size * 40}GB (pooled)")

    # Training loop
    for epoch in range(start_epoch, config.training.num_epochs):
        train_sampler.set_epoch(epoch)

        # Train one epoch
        train_one_epoch(
            model,
            config,
            args,
            optimizer,
            device,
            [rank],
            epoch,
            use_amp,
            scaler,
            scheduler,
            gradient_accumulation_steps,
            train_loader,
            multi_loss,
            all_losses,
            world_size,
        )

        # Save checkpoints
        if should_print:
            save_last_weights(
                args,
                model,
                [rank],
                optimizer,
                epoch,
                all_time_all_metrics,
                all_losses,
                best_metric,
                scheduler,
            )

        # Validation
        metrics_avg, all_metrics = valid_multi_gpu(
            model, args, config, args.device_ids, verbose=False
        )

        if rank == 0:
            all_time_all_metrics[f"epoch_{epoch}"] = all_metrics

            # Check if this is the best model
            metric_name = args.metric_for_scheduler
            current_metric = metrics_avg.get(metric_name, float("-inf"))

            if current_metric > best_metric:
                best_metric = current_metric
                # Save best model checkpoint
                store_path = f"{args.results_path}/model_{args.model_type}_ep_{epoch}_{metric_name}_{current_metric:.4f}.ckpt"
                print(f"Store best weights: {store_path}")
                save_weights(
                    store_path=store_path,
                    model=model,
                    device_ids=[rank],
                    optimizer=optimizer,
                    epoch=epoch,
                    all_time_all_metrics=all_time_all_metrics,
                    all_losses=all_losses,
                    best_metric=best_metric,
                    args=args,
                    scheduler=scheduler,
                )
                print(f"New best {metric_name}: {best_metric:.4f}")

    # Cleanup
    cleanup_ddp()


def train_model_fsdp_spawn(args=None):
    """Spawn FSDP training processes"""
    world_size = torch.cuda.device_count()

    if world_size < 2:
        print("ERROR: FSDP requires at least 2 GPUs")
        print("For single GPU training, use train_ddp.py instead")
        return

    try:
        mp.spawn(
            train_model_fsdp, args=(world_size, args), nprocs=world_size, join=True
        )
    except Exception as e:
        # Only cleanup if distributed was initialized
        import torch.distributed as dist

        if dist.is_initialized():
            cleanup_ddp()
        raise e


if __name__ == "__main__":
    from utils.settings import parse_args_train

    args = parse_args_train(None)
    train_model_fsdp_spawn(args)
