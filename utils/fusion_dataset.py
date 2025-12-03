# coding: utf-8
import torch
import torch.utils.data
import numpy as np
import os
from glob import glob
import random

class FusionDataset(torch.utils.data.Dataset):
    """
    Dataset for training the FusionModel.
    It loads a mixture audio, the corresponding target stems, and pre-computed
    latent tensors from scnet and bs_roformer.
    """
    def __init__(self, config, args):
        super(FusionDataset, self).__init__()
        
        self.config = config
        self.args = args
        self.sr = config.training.samplerate
        self.instruments = config.training.instruments
        
        # Assume a file structure where audio and latents share a base name
        # e.g., /path/to/audio/track01.wav
        #       /path/to/scnet_latents/track01.pt
        #       /path/to/bsr_latents/track01.pt
        
        self.audio_files_paths = self.get_audio_files_list()
        self.scnet_latent_dir = args.scnet_latent_dir
        self.bsr_latent_dir = args.bsr_latent_dir
        
        # Simple validation that latent dirs exist
        if not os.path.isdir(self.scnet_latent_dir):
            raise ValueError(f"SCNet latent directory not found at: {self.scnet_latent_dir}")
        if not os.path.isdir(self.bsr_latent_dir):
            raise ValueError(f"BSRoformer latent directory not found at: {self.bsr_latent_dir}")

    def get_audio_files_list(self):
        # This is adapted from utils/dataset.py
        if self.args.dataset_type == 'musdb':
            # Simplified for musdb, assuming a root directory
            root = self.config.training.musdb_path
            tracks = glob(f'{root}/train/*')
            return tracks
        elif self.args.dataset_type == 'custom_dataset':
            # For a custom dataset, assuming it's a list of files
            with open(self.config.training.training_list) as f:
                tracks = [line.strip() for line in f if line.strip()]
            return tracks
        else:
            raise ValueError(f"Unsupported dataset type: {self.args.dataset_type}")

    def __len__(self):
        return len(self.audio_files_paths)

    def __getitem__(self, index):
        audio_path = self.audio_files_paths[index]
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 1. Load target audio stems and create mixture
        # This part is a simplification of the logic in dataset.py
        all_stems = []
        for instr in self.instruments:
            # Assuming files are named like: /path/to/track/vocals.wav
            stem_path = os.path.join(audio_path, f"{instr}.wav")
            try:
                stem_audio = self.load_audio(stem_path)
                all_stems.append(stem_audio)
            except (IOError, FileNotFoundError):
                # If a stem is missing, use silence
                silence = torch.zeros(self.config.training.audio_channels, self.config.training.segment * self.sr)
                all_stems.append(silence)

        target_stems = torch.stack(all_stems)
        mix_audio = torch.sum(target_stems, dim=0)

        # Randomly chunk if needed (simplified from original dataset)
        if mix_audio.shape[-1] > self.config.training.segment * self.sr:
            start = random.randint(0, mix_audio.shape[-1] - self.config.training.segment * self.sr)
            end = start + self.config.training.segment * self.sr
            mix_audio = mix_audio[:, start:end]
            target_stems = target_stems[:, :, start:end]

        # 2. Load pre-computed latent tensors
        scnet_latent_path = os.path.join(self.scnet_latent_dir, f"{basename}.pt")
        bsr_latent_path = os.path.join(self.bsr_latent_dir, f"{basename}.pt")
        
        scnet_latent = torch.load(scnet_latent_path)
        bsr_latent = torch.load(bsr_latent_path)

        # Ensure latents are single precision
        scnet_latent = scnet_latent.float()
        bsr_latent = bsr_latent.float()

        # The original dataset returns (batch, mixes)
        # We will return (targets, mixes, scnet_latent, bsr_latent)
        return target_stems, mix_audio, scnet_latent, bsr_latent

    def load_audio(self, path):
        # A placeholder for an actual audio loading library like librosa or torchaudio
        # The original dataset has more complex logic for this.
        # This needs to be replaced with a proper implementation.
        # For now, returning silence to have a runnable structure.
        # A real implementation would use torchaudio.load, resample, etc.
        import torchaudio
        wav, sr = torchaudio.load(path)
        # Ensure consistent sample rate and channel count
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        if wav.shape[0] != self.config.training.audio_channels:
            # simple mono to stereo or stereo to mono
            if self.config.training.audio_channels == 2:
                wav = wav.expand(2, -1)
            else:
                wav = wav.mean(dim=0, keepdim=True)
        return wav
