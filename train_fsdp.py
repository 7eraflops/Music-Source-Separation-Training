import functools
import os
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler

from train import (
    compute_epoch_metrics,
    get_lora,
    save_best_weights,
    save_last_weights,
    train_one_epoch,
)
from utils.dataset import MSSDataset
from utils.settings import (
    cleanup_ddp,
    get_model_from_config,
    get_optimizer,
    get_scheduler,
    initialize_environment_ddp,
    parse_args_train,
    wandb_init,
)

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
    # Wrap layers larger than 1M parameters
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=1_000_000,  # 1M parameters
    )

    # Configure mixed precision
    # Use BF16 if available, otherwise FP16
    if torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        if should_print:
            print("Using BF16 mixed precision")
    else:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        if should_print:
            print("Using FP16 mixed precision")

    # Move model to device before FSDP wrapping
    model = model.to(device)

    # Wrap model with FSDP
    # FULL_SHARD: Shard parameters, gradients, and optimizer states across all GPUs
    if should_print:
        print("Wrapping model with FSDP (FULL_SHARD strategy)...")

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,  # Better compatibility with optimizers
    )

    if should_print:
        print("Model wrapped with FSDP successfully")
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

    # Initialize wandb
    if should_print:
        wandb_init(args, config)

    # Print training info
    if should_print:
        batch_size = config.training.batch_size
        print(f"Instruments: {config.training.instruments}")
        print(
            f"Metrics for training: {config.training.metrics}. Metric for scheduler: {config.training.metrics[0]}"
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
            args.device_ids,
            epoch,
            use_amp,
            scaler,
            scheduler,
            gradient_accumulation_steps,
            train_loader,
            None,
            all_losses,
            world_size,
        )

        # Save checkpoints
        if should_print:
            save_last_weights(
                args,
                model,
                args.device_ids,
                optimizer,
                epoch,
                all_time_all_metrics,
                best_metric,
                scheduler,
            )

        # Validation
        from utils.valid import valid_multi_gpu

        metrics_avg, all_metrics = valid_multi_gpu(
            model, args, config, args.device_ids, verbose=False
        )

        if rank == 0:
            all_time_all_metrics[f"epoch_{epoch}"] = all_metrics

            # Check if this is the best model
            metric_name = config.training.metrics[0]
            current_metric = metrics_avg.get(metric_name, float("-inf"))

            if current_metric > best_metric:
                best_metric = current_metric
                save_best_weights(
                    args,
                    model,
                    args.device_ids,
                    optimizer,
                    epoch,
                    all_time_all_metrics,
                    best_metric,
                    scheduler,
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
        cleanup_ddp()
        raise e


if __name__ == "__main__":
    from utils.settings import parse_args_train

    args = parse_args_train()
    train_model_fsdp_spawn(args)
