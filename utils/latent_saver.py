import os
import torch
import torch.nn.functional as F

BOTTLENECK_MODULES = {
    "bs_roformer": {
        "final_norm": "final_norm",
    },
    "scnet": {
        "separation_net": "separation_net",
    },
    "htdemucs": {
        "crosstransformer": "crosstransformer",
    },
}


class LatentSaver:
    def __init__(
        self,
        model,
        model_type,
        latents_path,
        dataset_name,
        subset_name,
        track_name,
        run_name,
    ):
        self.model = model
        self.model_type = model_type
        self.latents_path = latents_path
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.track_name = track_name
        self.run_name = run_name
        self.hooks = []
        self.collected_latents = {}
        # self.collected_skips = {"skips": {}, "time_skips": {}} # Removed: no longer collecting skips

        if self.model_type not in BOTTLENECK_MODULES:
            raise ValueError(
                f"Model type '{self.model_type}' not supported for latent saving."
            )

    def _get_save_path(self):
        # <run>/<model_type>/<dataset>/<subset>/<track_name>.pt
        return os.path.join(
            self.latents_path,
            self.run_name,
            self.model_type,
            self.dataset_name,
            self.subset_name,
            f"{self.track_name}.pt",
        )

    def _hook_fn(self, module, input, output, module_name):
        if module_name not in self.collected_latents:
            self.collected_latents[module_name] = []

        if isinstance(output, tuple):
            self.collected_latents[module_name].append(
                tuple(o.clone().detach().cpu() for o in output)
            )
        else:
            self.collected_latents[module_name].append(output.clone().detach().cpu())

    # _skip_hook_fn removed: no longer used

    def register_hooks(self):
        module_names = BOTTLENECK_MODULES[self.model_type]
        for name, module in self.model.named_modules():
            if name in module_names.values():
                module_key = [k for k, v in module_names.items() if v == name][0]
                if module_key not in self.collected_latents:
                    self.collected_latents[module_key] = []

                hook = module.register_forward_hook(
                    lambda module, input, output, module_name=module_key: self._hook_fn(
                        module, input, output, module_name
                    )
                )
                self.hooks.append(hook)
                print(f"Registered hook for {self.model_type}: {name}")

        # Removed: Special handling for htdemucs to capture skip connections

    def save_and_remove_hooks(self):
        # Special case for htdemucs to save its bottleneck latents into one file
        if self.model_type == "htdemucs":
            if "crosstransformer" not in self.collected_latents:
                print("Warning: No htdemucs bottleneck latents were collected.")
                return
            
            # Process bottleneck latents (which are lists of chunks)
            latents = self.collected_latents["crosstransformer"]
            latents_x = [item[0] for item in latents]
            latents_xt = [item[1] for item in latents]

            # Pad tensors to the same length before concatenation
            if len(latents_x) > 1:
                max_len_x = max(t.shape[3] for t in latents_x)
                padded_latents_x = []
                for t in latents_x:
                    pad_len = max_len_x - t.shape[3]
                    if pad_len > 0:
                        padded_latents_x.append(F.pad(t, (0, pad_len)))
                    else:
                        padded_latents_x.append(t)
                latents_x = padded_latents_x

            if len(latents_xt) > 1:
                max_len_xt = max(t.shape[2] for t in latents_xt)
                padded_latents_xt = []
                for t in latents_xt:
                    pad_len = max_len_xt - t.shape[2]
                    if pad_len > 0:
                        padded_latents_xt.append(F.pad(t, (0, pad_len)))
                    else:
                        padded_latents_xt.append(t)
                latents_xt = padded_latents_xt

            full_latent_x = torch.cat(latents_x, dim=3)
            full_latent_xt = torch.cat(latents_xt, dim=2)

            # Combine just the bottleneck latents into a dictionary
            data_to_save = {
                'freq_latent': full_latent_x,
                'time_latent': full_latent_xt,
            }

            # Save to a single file
            path = self._get_save_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(data_to_save, path)
            print(f"Saved htdemucs bottleneck latents to {path}")

        else:
            # Logic for other models (scnet, bs_roformer)
            for module_name, latents in self.collected_latents.items():
                if not latents:
                    continue
                # Determine time dimension for concatenation
                time_dim = 1  # bs_roformer (b, t, f, d)
                if self.model_type == "scnet":
                    time_dim = 3  # scnet (b, c, fr, t)
                
                # Also apply padding for other models just in case
                if len(latents) > 1:
                    max_len = max(t.shape[time_dim] for t in latents)
                    padded_latents = []
                    for t in latents:
                        pad_len = max_len - t.shape[time_dim]
                        if pad_len > 0:
                            # Create a padding tuple dynamically based on the dimension
                            # (pad_left, pad_right, pad_top, pad_bottom, ...)
                            # We only pad the last dimension used for time
                            pad_tuple = [0] * (2 * len(t.shape))
                            pad_tuple[2 * (len(t.shape) - 1 - time_dim) + 1] = pad_len
                            padded_latents.append(F.pad(t, tuple(pad_tuple)))
                        else:
                            padded_latents.append(t)
                    latents = padded_latents

                full_latent = torch.cat(latents, dim=time_dim)
                path = self._get_save_path()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(full_latent, path)
                print(f"Saved full latent to {path}")

        # Cleanup
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.collected_latents = {}
        print("Removed all hooks.")


def setup_latent_saver(model, args, config, track_path):
    if not args.save_latents_path:
        return None

    model_type = args.model_type
    if args.config_path:
        try:
            # Infer model type from config path, e.g., configs/scnet_xl/config.yaml -> scnet_xl
            path_parts = args.config_path.split(os.sep)
            if "configs" in path_parts:
                config_dir_index = path_parts.index("configs")
                if config_dir_index + 1 < len(path_parts):
                    inferred_model_type = path_parts[config_dir_index + 1]
                    if inferred_model_type:
                        model_type = inferred_model_type
                        print(f"Inferred model type for latent path: {model_type}")
        except ValueError:
            pass  # 'configs' not in path_parts

    latents_path = args.save_latents_path

    # Infer dataset and subset from input folder structure
    # Assumes input_folder is something like /path/to/dataset_name/subset_name
    # Examples:
    #   - /home/.../inference/moisesdb_musdb18_style_split/train
    #   - /home/.../datasets/musdb18hq/test
    input_folder = os.path.abspath(args.input_folder)

    # Get the last component of input_folder - this should be the subset (train/test/valid)
    subset_name = os.path.basename(input_folder)

    # Get the second-to-last component - this should be the dataset name
    parent_dir = os.path.dirname(input_folder)
    dataset_name = os.path.basename(parent_dir)

    # Fallback to sensible defaults if parsing fails
    if not subset_name or subset_name == ".":
        subset_name = "unknown_subset"
    if not dataset_name or dataset_name == ".":
        dataset_name = "unknown_dataset"

    track_name = os.path.splitext(os.path.basename(track_path))[0]

    # Determine run name from checkpoint path
    run_name = "default_run"
    if args.start_check_point:
        abs_ckpt_path = os.path.abspath(args.start_check_point)

        # Try to infer training session from checkpoint path structure
        if "checkpoints" in abs_ckpt_path:
            path_parts = abs_ckpt_path.split(os.sep)
            ckpt_index = path_parts.index("checkpoints")

            # Check if it's in a pattern like: checkpoints/{model_type}/model.ckpt
            if ckpt_index > 0 and path_parts[ckpt_index - 1] not in ["checkpoints"]:
                # Check for training_X or outputs/session pattern before checkpoints
                for i in range(ckpt_index - 1, -1, -1):
                    part = path_parts[i]
                    if (
                        part.startswith("training_")
                        or part == "results"
                        or part == "outputs"
                    ):
                        if part == "outputs" and i + 1 < ckpt_index:
                            run_name = path_parts[i + 1]
                        elif part == "results" and i + 1 < ckpt_index:
                            run_name = path_parts[i + 1]
                        else:
                            run_name = part
                        break
                else:
                    # No training session found, assume original_models
                    run_name = "original_models"
            else:
                run_name = "original_models"
        else:
            # Fallback to checkpoint filename
            run_name = os.path.splitext(os.path.basename(args.start_check_point))[0]

    print(
        f"Latent saver setup: run={run_name}, model_type={model_type}, dataset={dataset_name}, subset={subset_name}, track={track_name}"
    )

    try:
        saver = LatentSaver(
            model,
            model_type,
            latents_path,
            dataset_name,
            subset_name,
            track_name,
            run_name,
        )
        saver.register_hooks()
        return saver
    except ValueError as e:
        print(f"Warning: {e}")
        return None