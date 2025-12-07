#!/usr/bin/env python3
"""
Quick verification script to test latent loading for both BS-Roformer and SCNet-XL.
"""

import os
import sys
from glob import glob

import torch


def check_latent_directory(latents_path):
    """Check what latent sources are available in the directory."""
    print(f"\n{'=' * 80}")
    print(f"Checking latents directory: {latents_path}")
    print(f"{'=' * 80}\n")

    if not os.path.exists(latents_path):
        print(f"❌ ERROR: Directory does not exist!")
        return False

    # Check for source subdirectories
    subdirs = [
        d
        for d in os.listdir(latents_path)
        if os.path.isdir(os.path.join(latents_path, d))
    ]

    print(f"Found subdirectories: {subdirs}\n")

    sources = {}
    for subdir in subdirs:
        if subdir in ["bs_roformer", "scnet_xl", "scnet", "htdemucs"]:
            # Search recursively to handle nested structures
            latent_files = glob(
                os.path.join(latents_path, subdir, "**", "*.pt"), recursive=True
            )
            sources[subdir] = len(latent_files)
            print(f"  {subdir}: {len(latent_files)} files")

    if not sources:
        print("❌ ERROR: No latent source directories found!")
        print("   Expected: bs_roformer/ and/or scnet_xl/")
        return False

    print()
    return sources


def test_latent_format(latent_file, source_name):
    """Test loading and check format of a latent file."""
    print(f"\nTesting {source_name} latent format...")
    print(f"  File: {os.path.basename(latent_file)}")

    try:
        latents = torch.load(latent_file, map_location="cpu", weights_only=False)

        if isinstance(latents, torch.Tensor):
            print(f"  ✓ Format: Tensor")
            print(f"  ✓ Shape: {tuple(latents.shape)}")
            print(f"  ✓ Dtype: {latents.dtype}")
            print(
                f"  ✓ Memory: {latents.numel() * latents.element_size() / 1024**3:.2f} GB"
            )

            # Detect likely format
            if latents.ndim == 4:
                if latents.shape[1] > latents.shape[3]:
                    print(
                        f"  ✓ Detected format: (B, Time, Freq, Channels) - BS-Roformer style"
                    )
                else:
                    print(
                        f"  ✓ Detected format: (B, Channels, Freq, Time) - SCNet style"
                    )

            return True
        elif isinstance(latents, dict):
            print(f"  ✓ Format: Dictionary")
            print(f"  ✓ Keys: {list(latents.keys())}")
            return True
        else:
            print(f"  ❌ Unknown format: {type(latents)}")
            return False

    except Exception as e:
        print(f"  ❌ ERROR loading file: {e}")
        return False


def main():
    if len(sys.argv) > 1:
        latents_path = sys.argv[1]
    else:
        # Default path from your setup
        latents_path = os.path.expandvars(
            "$SCRATCH/latents_batch_size_1/original_models/"
        )

    # Check directory structure
    sources = check_latent_directory(latents_path)
    if not sources:
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("Testing latent file formats")
    print(f"{'=' * 80}")

    # Test one file from each source
    all_ok = True
    for source_name, count in sources.items():
        source_dir = os.path.join(latents_path, source_name)
        # Use recursive glob to handle nested structures
        latent_files = glob(os.path.join(source_dir, "**", "*.pt"), recursive=True)

        if latent_files:
            # Test first file
            ok = test_latent_format(latent_files[0], source_name)
            all_ok = all_ok and ok
        else:
            print(f"\n❌ {source_name}: No .pt files found!")
            all_ok = False

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")

    for source_name, count in sources.items():
        status = "✓" if count > 0 else "❌"
        print(f"  {status} {source_name}: {count} latent files")

    print()

    if all_ok and len(sources) >= 2:
        print("✓ All checks passed! Both latent sources are available and loadable.")
        print("\nYour config should use:")
        print(f"  latent_sources: {list(sources.keys())}")
    elif all_ok and len(sources) == 1:
        print("⚠ Only one latent source available.")
        print(f"\nYour config should use:")
        print(f"  latent_sources: {list(sources.keys())}")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
