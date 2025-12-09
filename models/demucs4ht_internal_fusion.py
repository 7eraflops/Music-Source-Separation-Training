import math
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.hdemucs import pad1d
from demucs.spec import ispectro, spectro
from einops import rearrange
from openunmix.filtering import wiener
from torch.utils.checkpoint import checkpoint

from models.demucs4ht import (
    HDecLayer,
    HEncLayer,
    MultiWrap,
    ScaledEmbedding,
    capture_init,
    rescale_module,
)


class LatentPreprocessor(nn.Module):
    """
    Preprocesses external latents (BS-Roformer, SCNet-XL) for fusion.
    - Only permutes dimensions (no interpolation)
    - Projects channels to model dimension
    - Handles batch size mismatch
    - Pads shorter sequences to match the longest one
    - Returns attention mask for padded positions
    """

    def __init__(self, model_channels, latent_sources=["bs_roformer", "scnet_xl"]):
        super().__init__()
        self.model_channels = model_channels
        self.latent_sources = latent_sources

        # Channel projection layers for each latent source
        if "bs_roformer" in latent_sources:
            # BS-Roformer has 384 channels
            self.bs_roformer_proj = nn.Conv1d(384, model_channels, 1)

        if "scnet_xl" in latent_sources:
            # SCNet-XL has 256 channels
            self.scnet_proj = nn.Conv1d(256, model_channels, 1)

    def forward(self, latents, batch_size, target_shape=None):
        """
        Args:
            latents: dict with keys like 'bs_roformer', 'scnet_xl'
            batch_size: target batch size B
            target_shape: (Freq, Time) tuple for Adaptive Pooling. 
                          This aligns external latents to the HTDemucs bottleneck resolution.

        Returns:
            combined: Tensor of shape (B, C_total, seq_len) where C_total is sum of all latent channels.
                     Sequence length will be exactly Freq * Time from target_shape.
            mask: None (no padding needed as adaptive pool forces exact size)
        """
        processed = []
        
        if target_shape is None:
            target_shape = (8, 400) # Fallback

        if "bs_roformer" in latents and "bs_roformer" in self.latent_sources:
            bsr = latents["bs_roformer"]  # (1, T, Fr, C)

            # Handle DataLoader collation
            if bsr.dim() == 5 and bsr.shape[1] == 1:
                bsr = bsr.squeeze(1)

            # Sanitize input
            if not torch.isfinite(bsr).all():
                bsr = torch.nan_to_num(bsr, nan=0.0, posinf=0.0, neginf=0.0)

            # Permute to (1, C, Fr, T)
            bsr = bsr.permute(0, 3, 2, 1)
            
            # Adaptive Pool to match bottleneck resolution
            bsr = F.adaptive_avg_pool2d(bsr, target_shape)

            bsr = bsr.flatten(2)
            bsr = self.bs_roformer_proj(bsr)

            if batch_size > 1:
                bsr = bsr.expand(batch_size, -1, -1)

            processed.append(bsr)

        if "scnet_xl" in latents and "scnet_xl" in self.latent_sources:
            scn = latents["scnet_xl"]  # (1, C, Fr, T)

            if scn.dim() == 5 and scn.shape[1] == 1:
                scn = scn.squeeze(1)

            if not torch.isfinite(scn).all():
                scn = torch.nan_to_num(scn, nan=0.0, posinf=0.0, neginf=0.0)

            # Already in (1, C, Fr, T) format
            scn = F.adaptive_avg_pool2d(scn, target_shape)
            
            scn = scn.flatten(2)
            scn = self.scnet_proj(scn)

            if batch_size > 1:
                scn = scn.expand(batch_size, -1, -1)

            processed.append(scn)

        if len(processed) == 0:
            return None, None

        # Concatenate along channel dimension
        combined = torch.cat(processed, dim=1)

        return combined, None


class TransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with self-attention.
    Used for the freq branch processing.
    """

    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src):
        """
        Args:
            src: (B, C, seq_len) - reshaped to (B, seq_len, C) internally
        Returns:
            (B, C, seq_len)
        """
        # Reshape for attention: (B, C, L) -> (B, L, C)
        x = src.transpose(1, 2)

        # Self-attention
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout1(x2)

        # Feedforward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)

        # Reshape back: (B, L, C) -> (B, C, L)
        return x.transpose(1, 2)


class LatentCrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for fusing external latents.
    Query: freq branch, Key/Value: external latents
    Refactored to use F.scaled_dot_product_attention for FlashAttention support.
    """

    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        # Manual projections for SDPA
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout_p = dropout

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, query, key_value, key_padding_mask=None):
        """
        Args:
            query: (B, C, seq_len_q) - freq branch
            key_value: (B, C_kv, seq_len_kv) - external latents
            key_padding_mask: (B, seq_len_kv) - True for valid positions, False for padded positions
        Returns:
            (B, C, seq_len_q)
        """
        # Reshape for attention: (B, C, L) -> (B, L, C)
        q_in = query.transpose(1, 2)
        kv_in = key_value.transpose(1, 2)
        
        B, L_q, _ = q_in.shape
        _, L_kv, _ = kv_in.shape

        # Pre-norm (standard for this architecture)
        q = self.norm1(q_in)
        
        # Projections: (B, L, d_model)
        q = self.q_proj(q)
        k = self.k_proj(kv_in)
        v = self.v_proj(kv_in)

        # Reshape for SDPA: (B, nhead, L, head_dim)
        q = q.view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.nhead, self.head_dim).transpose(1, 2)

        # Mask preparation
        # Adaptive Pooling (applied in LatentPreprocessor) guarantees no padding,
        # so key_padding_mask is generally None now. 
        # If it were present, we'd need to reshape it for SDPA.
        attn_mask = None
        if key_padding_mask is not None:
            # Logic: mask is True for VALID. 
            # SDPA boolean mask expects True for IGNORED (Padded).
            # So we need ~mask.
            # Shape: (B, L_kv) -> (B, 1, 1, L_kv) for broadcasting
            sdpa_mask = ~key_padding_mask
            sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(1) 
            attn_mask = sdpa_mask

        # Flash Attention / SDPA
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # Reshape back: (B, nhead, L, head_dim) -> (B, L, d_model)
        x = x.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        # Output projection
        x = self.out_proj(x)
        
        # Residual connection + Post-processing
        q_out = q_in + self.dropout1(x)

        # Feedforward
        x2 = self.norm2(q_out)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        q_out = q_out + self.dropout2(x2)

        # Reshape back: (B, L, C) -> (B, C, L)
        return q_out.transpose(1, 2)


class CrossTransformerEncoderWithLatents(nn.Module):
    """
    Custom cross-domain transformer with interleaved latent fusion.

    Architecture (for 2 original blocks):
        reshape ->
        [self-att, cross-att(time)] -> [self-att, cross-att(latents)] ->
        [self-att, cross-att(time)] -> [self-att, cross-att(latents)] ->
        [self-att] -> reshape
    """

    def __init__(
        self,
        dim,
        emb,
        hidden_scale,
        num_heads,
        num_layers,
        cross_first,
        dropout,
        max_positions,
        norm_in,
        norm_in_group,
        group_norm,
        norm_first,
        norm_out,
        max_period,
        weight_decay,
        lr,
        layer_scale,
        gelu,
        sin_random_shift,
        weight_pos_embed,
        cape_mean_normalize,
        cape_augment,
        cape_glob_loc_scale,
        sparse_self_attn,
        sparse_cross_attn,
        mask_type,
        mask_random_seed,
        sparse_attn_window,
        global_window,
        sparsity,
        auto_sparsity,
        num_latent_blocks=2,
        latent_sources=["bs_roformer", "scnet_xl"],
        use_gradient_checkpointing=False,
    ):
        """
        Args:
            num_latent_blocks: Number of latent fusion blocks to insert (default: 2)
            Other args match demucs CrossTransformerEncoder for compatibility
        """
        super().__init__()

        # Import demucs transformer to use for original cross-domain layers
        from demucs.transformer import CrossTransformerEncoder

        self.dim = dim
        self.num_latent_blocks = num_latent_blocks
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Original cross-domain transformer (freq <-> time)
        # We'll use this for the freq-time cross-attention blocks
        self.original_transformer = CrossTransformerEncoder(
            dim=dim,
            emb=emb,
            hidden_scale=hidden_scale,
            num_heads=num_heads,
            num_layers=num_layers,
            cross_first=cross_first,
            dropout=dropout,
            max_positions=max_positions,
            norm_in=norm_in,
            norm_in_group=norm_in_group,
            group_norm=group_norm,
            norm_first=norm_first,
            norm_out=norm_out,
            max_period=max_period,
            weight_decay=weight_decay,
            lr=lr,
            layer_scale=layer_scale,
            gelu=gelu,
            sin_random_shift=sin_random_shift,
            weight_pos_embed=weight_pos_embed,
            cape_mean_normalize=cape_mean_normalize,
            cape_augment=cape_augment,
            cape_glob_loc_scale=cape_glob_loc_scale,
            sparse_self_attn=sparse_self_attn,
            sparse_cross_attn=sparse_cross_attn,
            mask_type=mask_type,
            mask_random_seed=mask_random_seed,
            sparse_attn_window=sparse_attn_window,
            global_window=global_window,
            sparsity=sparsity,
            auto_sparsity=auto_sparsity,
        )

        # Latent fusion layers
        # We add latent blocks after each original block
        self.latent_self_attn_layers = nn.ModuleList()
        self.latent_cross_attn_layers = nn.ModuleList()

        dim_feedforward = int(dim * hidden_scale)
        activation = "gelu" if gelu else "relu"

        for _ in range(num_latent_blocks):
            self.latent_self_attn_layers.append(
                TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
            )

            # Cross-attention expects query and kv to have same d_model
            # External latents will be projected to dim by LatentPreprocessor
            # But they might have n_sources * dim channels after concatenation
            # We need to handle this - either:
            # 1. Use a projection layer before cross-attention
            # 2. Use cross-attention that accepts different kv dimensions
            # For simplicity, we'll add a projection layer
            self.latent_cross_attn_layers.append(
                LatentCrossAttentionLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
            )

        # Projection for external latents if they have multiple sources concatenated
        # This projects from n_sources * dim back to dim for cross-attention
        self.latent_kv_proj = nn.Conv1d(dim * len(latent_sources), dim, 1)

    def forward(self, x, xt, external_latents=None, latent_mask=None):
        """
        Args:
            x: (B, C, Fr, T1) - freq branch before reshape
            xt: (B, C, T2) - time branch
            external_latents: (B, C_latents, seq_len) - preprocessed external latents
            latent_mask: (B, seq_len) - attention mask where True = valid, False = padded

        Returns:
            x: (B, C, Fr, T1) - freq branch
            xt: (B, C, T2) - time branch
        """
        # First, run the original cross-domain transformer
        # This handles the freq <-> time cross-attention
        # The original transformer expects (B, C, Fr, T) format
        x, xt = self.original_transformer(x, xt)

        # Now add latent fusion layers for the freq branch only
        if external_latents is not None:
            # Need to reshape x for latent fusion: (B, C, Fr, T) -> (B, C, Fr*T)
            B, C, Fr, T = x.shape
            x_flat = rearrange(x, "b c f t -> b c (f t)")

            # Project external latents to correct dimension
            latents_proj = self.latent_kv_proj(external_latents)  # (B, C, seq_len)

            # Add latent fusion blocks
            # We append them after the original transformer
            for self_attn, cross_attn in zip(
                self.latent_self_attn_layers, self.latent_cross_attn_layers
            ):
                if (
                    self.training
                    and hasattr(self, "use_gradient_checkpointing")
                    and self.use_gradient_checkpointing
                ):
                    # Use gradient checkpointing to save memory during training
                    x_flat = checkpoint(self_attn, x_flat, use_reentrant=False)
                    x_flat = checkpoint(
                        cross_attn,
                        x_flat,
                        latents_proj,
                        latent_mask,
                        use_reentrant=False,
                    )
                else:
                    # Self-attention on freq branch
                    x_flat = self_attn(x_flat)
                    # Cross-attention: freq <-> external latents with attention mask
                    x_flat = cross_attn(
                        x_flat, latents_proj, key_padding_mask=latent_mask
                    )

            # Reshape back to (B, C, Fr, T)
            x = rearrange(x_flat, "b c (f t) -> b c f t", f=Fr)

        return x, xt


class InternalFusionHTDemucs(nn.Module):
    """
    HTDemucs with internal transformer latent fusion.
    Latents are fused inside the transformer architecture.
    """

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=6,
        rewrite=True,
        hybrid_old=False,
        # Frequency Branch
        multi_freqs=None,
        multi_freqs_depth=2,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=4,
        dconv_attn=4,
        dconv_lstm=4,
        dconv_init=1e-3,
        # Pre/post processing
        normalize=True,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=7.8,
        use_train_segment=True,
        # Spectral representation
        num_subbands=1,
        # Bottom channels for the transformer
        bottom_channels=0,
        # transformer options
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=None,
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        t_cross_first=False,
        # Latent fusion options
        use_internal_fusion=True,
        freeze_encoder=False,
        latent_sources=["bs_roformer", "scnet_xl"],
        num_latent_blocks=2,
        use_gradient_checkpointing=False,
    ):
        """
        Args:
            use_internal_fusion: Enable internal latent fusion in transformer
            freeze_encoder: Freeze encoder parameters (only train transformer/decoder)
            latent_sources: List of external latent sources to fuse
            num_latent_blocks: Number of latent fusion blocks to add
            ... (other args same as HTDemucs)
        """
        super().__init__()

        self.num_subbands = num_subbands
        self.cac = cac
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        self.use_internal_fusion = use_internal_fusion
        self.latent_sources = latent_sources
        assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        if self.num_subbands > 1:
            chin_z *= self.num_subbands
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=bool(dconv_mode & 1), context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=bool(dconv_mode & 1),
                    context=context_enc,
                    empty=last_freq,
                    **kwt,
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
                if self.num_subbands > 1:
                    chin_z *= self.num_subbands
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=bool(dconv_mode & 2),
                last=index == 0,
                context=context,
                **kw_dec,
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=bool(dconv_mode & 2),
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt,
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb
            else:
                if index == 0:
                    self.freq_emb = None

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            if use_internal_fusion:
                # Use custom transformer with latent fusion
                self.crosstransformer = CrossTransformerEncoderWithLatents(
                    dim=transformer_channels,
                    emb=t_emb,
                    hidden_scale=t_hidden_scale,
                    num_heads=t_heads,
                    num_layers=t_layers,
                    cross_first=t_cross_first,
                    dropout=t_dropout,
                    max_positions=t_max_positions,
                    norm_in=t_norm_in,
                    norm_in_group=t_norm_in_group,
                    group_norm=t_group_norm,
                    norm_first=t_norm_first,
                    norm_out=t_norm_out,
                    max_period=t_max_period,
                    weight_decay=t_weight_decay,
                    lr=t_lr,
                    layer_scale=t_layer_scale,
                    gelu=t_gelu,
                    sin_random_shift=t_sin_random_shift,
                    weight_pos_embed=t_weight_pos_embed,
                    cape_mean_normalize=t_cape_mean_normalize,
                    cape_augment=t_cape_augment,
                    cape_glob_loc_scale=t_cape_glob_loc_scale,
                    sparse_self_attn=t_sparse_self_attn,
                    sparse_cross_attn=t_sparse_cross_attn,
                    mask_type=t_mask_type,
                    mask_random_seed=t_mask_random_seed,
                    sparse_attn_window=t_sparse_attn_window,
                    global_window=t_global_window,
                    sparsity=t_sparsity,
                    auto_sparsity=t_auto_sparsity,
                    num_latent_blocks=num_latent_blocks,
                    latent_sources=latent_sources,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                )

                # Latent preprocessor
                self.latent_preprocessor = LatentPreprocessor(
                    model_channels=transformer_channels,
                    latent_sources=latent_sources,
                )
            else:
                # Use standard demucs transformer
                from demucs.transformer import CrossTransformerEncoder

                self.crosstransformer = CrossTransformerEncoder(
                    dim=transformer_channels,
                    emb=t_emb,
                    hidden_scale=t_hidden_scale,
                    num_heads=t_heads,
                    num_layers=t_layers,
                    cross_first=t_cross_first,
                    dropout=t_dropout,
                    max_positions=t_max_positions,
                    norm_in=t_norm_in,
                    norm_in_group=t_norm_in_group,
                    group_norm=t_group_norm,
                    norm_first=t_norm_first,
                    norm_out=t_norm_out,
                    max_period=t_max_period,
                    weight_decay=t_weight_decay,
                    lr=t_lr,
                    layer_scale=t_layer_scale,
                    gelu=t_gelu,
                    sin_random_shift=t_sin_random_shift,
                    weight_pos_embed=t_weight_pos_embed,
                    cape_mean_normalize=t_cape_mean_normalize,
                    cape_augment=t_cape_augment,
                    cape_glob_loc_scale=t_cape_glob_loc_scale
                    if t_cape_glob_loc_scale is not None
                    else [5000.0, 1.0, 1.4],
                    sparse_self_attn=t_sparse_self_attn,
                    sparse_cross_attn=t_sparse_cross_attn,
                    mask_type=t_mask_type,
                    mask_random_seed=t_mask_random_seed,
                    sparse_attn_window=t_sparse_attn_window,
                    global_window=t_global_window,
                    sparsity=t_sparsity,
                    auto_sparsity=t_auto_sparsity,
                )
        else:
            self.crosstransformer = None

        # Freeze encoder if requested
        if freeze_encoder:
            self.freeze_encoder_parameters()

    def freeze_encoder_parameters(self):
        """
        Freeze all encoder and tencoder parameters.
        Only transformer and decoder will be trainable.
        """
        print("Freezing encoder parameters...")

        # Freeze frequency branch encoder
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False

        # Freeze time branch encoder
        for name, param in self.tencoder.named_parameters():
            param.requires_grad = False

        # Also freeze frequency embedding if it exists
        if self.freq_emb is not None:
            for param in self.freq_emb.parameters():
                param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.encoder.parameters())
        frozen_params += sum(p.numel() for p in self.tencoder.parameters())
        if self.freq_emb is not None:
            frozen_params += sum(p.numel() for p in self.freq_emb.parameters())

        print(f"Frozen {frozen_params:,} encoder parameters")

    def unfreeze_encoder_parameters(self):
        """Utility to unfreeze encoder parameters if needed."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.tencoder.parameters():
            param.requires_grad = True
        if self.freq_emb is not None:
            for param in self.freq_emb.parameters():
                param.requires_grad = True

    def _spec(self, x):
        """Compute spectrogram."""
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        """Inverse spectrogram."""
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = ispectro(z, hl, length)
        return z

    def _magnitude(self, z):
        """Compute magnitude of complex spectrogram."""
        if self.cac:
            B, C, Fq, T = z.shape
            z = torch.view_as_real(z)
            z = z.permute(0, 1, 4, 2, 3)
            z = z.reshape(B, C * 2, Fq, T)
        else:
            z = z.abs()
        return z

    def _mask(self, z, m):
        """Apply mask to spectrogram."""
        # Apply masking given the mixture spectrogram `z` and the estimated mask `m`.
        # If `cac` is True, `m` is actually a full spectrogram and `z` is ignored.
        niters = self.wiener_iters
        if self.cac:
            B, S, C, Fr, T = m.shape
            out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
            out = torch.view_as_complex(out.contiguous())
            return out
        if self.training:
            niters = self.end_iters
        if niters < 0:
            z = z[:, None]
            return z / (1e-8 + z.abs()) * m
        else:
            return self._wiener(m, z, niters)

    def _wiener(self, mag_out, mix_stft, niters):
        """Apply Wiener filtering for post-processing."""
        # Apply Wiener filtering given the magnitude spectrogram estimates.
        # `mag_out` has shape (B, S, C, Fr, T)
        # `mix_stft` has shape (B, C, Fr, T)
        # Returns (B, S, C, Fr, T) complex spectrogram
        mag_out = mag_out.permute(0, 4, 3, 2, 1)
        mix_stft = mix_stft.permute(0, 3, 2, 1)

        outs = []
        for sample in range(mag_out.shape[0]):
            out = []
            for src in range(mag_out.shape[-1]):
                out.append(mag_out[sample, ..., src])

            out = torch.stack(out, dim=-1)
            out = wiener(
                out,
                mix_stft[sample],
                niters,
                residual=self.wiener_residual,
            )
            outs.append(out)
        out = torch.stack(outs, dim=0)
        out = out.permute(0, 4, 3, 2, 1)
        return out

    def valid_length(self, length):
        """
        Return the valid length of the input based on model architecture.
        """
        length = int(length)
        for _ in range(self.depth):
            if length <= 1:
                return 1
            length = (length - self.kernel_size) // self.stride + 1
            length = max(length, 1) + 2 * self.context
        for _ in range(self.depth):
            length = length * self.stride + self.kernel_size
        return int(length)

    def cac2cws(self, x):
        """Convert CAC (Channel, Audio, Complex) to CWS (Channel, Window, Subband)."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, f, k, t // k)
        x = x.reshape(b, c, k * f, t // k)
        return x

    def cws2cac(self, x):
        """Convert CWS back to CAC."""
        k = self.num_subbands
        b, c, f, t = x.shape
        x = x.reshape(b, c, f // k, k, t)
        x = x.reshape(b, c, f // k, k * t)
        return x

    def forward(self, mix, latents=None):
        """
        Forward pass with optional latent fusion.

        Args:
            mix: (B, C, T) - input audio mixture
            latents: dict with external latents (optional)

        Returns:
            (B, n_sources, C, T) - separated sources
        """
        length = mix.shape[-1]
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))

        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        if self.num_subbands > 1:
            x = self.cac2cws(x)

        B, C, Fq, T = x.shape

        # Normalize
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # Prepare time branch
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Encoder
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                freq_emb_module = self.freq_emb
                assert freq_emb_module is not None
                emb = freq_emb_module(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        # Transformer with latent fusion
        if self.crosstransformer:
            # Save frequency dimension before any transformations
            freq_dim = x.shape[2]

            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            # Preprocess external latents if using internal fusion
            processed_latents = None
            latent_mask = None
            if self.use_internal_fusion and latents is not None and len(latents) > 0:
                processed_latents, latent_mask = self.latent_preprocessor(
                    latents, batch_size=B
                )

            # Apply transformer with latent fusion
            # Note: x is still (B, C, Fr, T), transformer will handle reshaping internally
            if self.use_internal_fusion:
                if self.crosstransformer is not None:
                    x, xt = self.crosstransformer(x, xt, processed_latents, latent_mask)
            elif self.crosstransformer is not None:
                x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=freq_dim)
                xt = self.channel_downsampler_t(xt)

        # Decoder
        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)

        if self.num_subbands > 1:
            x = x.view(B, -1, Fq, T)
            x = self.cws2cac(x)

        x = x.view(B, S, -1, Fq * self.num_subbands, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, int(self.segment * self.samplerate))
        else:
            x = self._ispec(zout, length)

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, int(self.segment * self.samplerate))
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x

        if length_pre_pad:
            x = x[..., :length_pre_pad]

        return x


def get_model(args):
    """
    Factory function to create InternalFusionHTDemucs model from config.
    """
    extra = {
        "sources": list(args.training.instruments),
        "audio_channels": args.training.channels,
        "samplerate": args.training.samplerate,
        "segment": args.training.segment,
    }

    # Get model-specific parameters from config
    # args is already a ConfigDict, so we can access it directly
    model_config = getattr(args, args.model)

    # Convert ConfigDict to regular dict
    if hasattr(model_config, "to_dict"):
        kw = model_config.to_dict()
    else:
        kw = dict(model_config)

    # Create model
    model = InternalFusionHTDemucs(**extra, **kw)

    return model
