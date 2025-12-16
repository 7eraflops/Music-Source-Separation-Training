import math
from fractions import Fraction

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

# Import the base class and helper components
from models.demucs4ht import HTDemucs, capture_init
from demucs.transformer import CrossTransformerEncoder, create_2d_sin_embedding, MyTransformerEncoderLayer

class LatentPreprocessor(nn.Module):
    """
    Preprocesses external latents (BS-Roformer, SCNet-XL) for fusion.
    Outputs (B, L, C) format for transformer.
    """

    def __init__(self, model_channels, latent_sources=["bs_roformer", "scnet_xl"]):
        super().__init__()
        self.model_channels = model_channels
        self.latent_sources = latent_sources

        # Channel projection layers for each latent source
        # We use GroupNorm(1, C) which is equivalent to LayerNorm but works on (B, C, L)
        if "bs_roformer" in latent_sources:
            # BS-Roformer has 384 channels
            self.bs_roformer_proj = nn.Sequential(
                nn.Conv1d(384, model_channels, 1),
                nn.GroupNorm(1, model_channels)
            )

        if "scnet_xl" in latent_sources:
            # SCNet-XL has 256 channels
            self.scnet_proj = nn.Sequential(
                nn.Conv1d(256, model_channels, 1),
                nn.GroupNorm(1, model_channels)
            )

    def forward(self, latents, batch_size):
        processed = []

        if "bs_roformer" in latents and "bs_roformer" in self.latent_sources:
            bsr = latents["bs_roformer"]
            bsr = bsr.detach()
            if bsr.dim() == 5 and bsr.shape[1] == 1:
                bsr = bsr.squeeze(1)
            if not torch.isfinite(bsr).all():
                bsr = torch.nan_to_num(bsr, nan=0.0, posinf=0.0, neginf=0.0)

            # (B, T, F, C) -> (B, C, F, T) -> (B, C, F*T)
            bsr = bsr.permute(0, 3, 2, 1)
            bsr = bsr.flatten(2)
            bsr = self.bs_roformer_proj(bsr) # (B, C, L)
            bsr = bsr.transpose(1, 2) # (B, L, C)

            if batch_size > 1:
                bsr = bsr.expand(batch_size, -1, -1)
            processed.append(bsr)

        if "scnet_xl" in latents and "scnet_xl" in self.latent_sources:
            scn = latents["scnet_xl"]
            scn = scn.detach()
            if scn.dim() == 5 and scn.shape[1] == 1:
                scn = scn.squeeze(1)
            if not torch.isfinite(scn).all():
                scn = torch.nan_to_num(scn, nan=0.0, posinf=0.0, neginf=0.0)

            # (B, C, F, T) -> (B, C, F*T)
            scn = scn.flatten(2)
            scn = self.scnet_proj(scn) # (B, C, L)
            scn = scn.transpose(1, 2) # (B, L, C)

            if batch_size > 1:
                scn = scn.expand(batch_size, -1, -1)
            processed.append(scn)

        if len(processed) == 0:
            return None, None

        # Concatenate along sequence dimension (dim=1 now because (B, L, C))
        combined = torch.cat(processed, dim=1)
        return combined, None


class LatentCrossAttentionLayer(nn.Module):
    """Cross-attention layer for fusing external latents. Expects (B, L, C)"""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu", layer_norm_eps=1e-5):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, query, key_value, key_padding_mask=None):
        # query: (B, Lq, C), key_value: (B, Lkv, C)
        B, L_q, _ = query.shape
        _, L_kv, _ = key_value.shape

        q = self.norm1(query)
        q = self.q_proj(q)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = q.view(B, L_q, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.nhead, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask is (B, L_kv) where True is valid?
            # Usually mask logic is complex. 
            # If we assume no mask needed for concat (since all are valid parts of song), we can skip.
            # But if passed:
            sdpa_mask = ~key_padding_mask
            sdpa_mask = sdpa_mask.unsqueeze(1).unsqueeze(1) 
            attn_mask = sdpa_mask

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0)
        x = x.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        x = self.out_proj(x)
        
        q_out = query + self.dropout1(x)
        x2 = self.norm2(q_out)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        q_out = q_out + self.dropout2(x2)
        return q_out


class InterleavedFusionTransformer(CrossTransformerEncoder):
    """
    Inherits from Demucs CrossTransformerEncoder but injects fusion layers
    interleaved after each original cross-attention layer.
    """
    def __init__(self, *args, num_latent_blocks=2, **kwargs):
        # Initialize original layers
        super().__init__(*args, **kwargs)
        
        # Initialize fusion layers
        # We need one set of fusion blocks for every Original Cross-Attention Layer.
        # Original Cross-Attn layers occur when idx % 2 != self.classic_parity
        
        dim = kwargs.get('dim', args[0] if args else 512)
        hidden_scale = kwargs.get('hidden_scale', 4.0)
        nhead = kwargs.get('num_heads', 8)
        dropout = kwargs.get('dropout', 0.0)
        gelu = kwargs.get('gelu', True)
        
        # Arguments for MyTransformerEncoderLayer to match original
        activation = F.gelu if gelu else F.relu
        kwargs_common = {
            "d_model": dim,
            "nhead": nhead,
            "dim_feedforward": int(dim * hidden_scale),
            "dropout": dropout,
            "activation": activation,
            "group_norm": kwargs.get('group_norm', False),
            "norm_first": kwargs.get('norm_first', True),
            "norm_out": kwargs.get('norm_out', True),
            "layer_scale": kwargs.get('layer_scale', True),
            "mask_type": kwargs.get('mask_type', "diag"),
            "mask_random_seed": kwargs.get('mask_random_seed', 42),
            "sparse_attn_window": kwargs.get('sparse_attn_window', 500),
            "global_window": kwargs.get('global_window', 100),
            "sparsity": kwargs.get('sparsity', 0.95),
            "auto_sparsity": kwargs.get('auto_sparsity', False),
            "batch_first": True,
        }
        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update({
            "sparse": kwargs.get('sparse_self_attn', False),
        })
        
        self.fusion_layers = nn.ModuleList()
        dim_feedforward = int(dim * hidden_scale)
        activation_str = "gelu" if gelu else "relu"
        
        for idx in range(self.num_layers):
            if idx % 2 != self.classic_parity:
                # This is a cross-attention layer location
                block_units = nn.ModuleList()
                for _ in range(num_latent_blocks):
                     block_units.append(nn.ModuleList([
                         # Use original Demucs Self-Attention Layer
                         MyTransformerEncoderLayer(**kwargs_classic_encoder),
                         # Use custom Cross-Attention Layer for Latents
                         LatentCrossAttentionLayer(
                             dim, nhead=nhead, dim_feedforward=dim_feedforward, 
                             dropout=dropout, activation=activation_str
                         )
                     ]))
                self.fusion_layers.append(block_units)

    def forward(self, x, xt, external_latents=None, latent_mask=None):
        # --- Logic copied from CrossTransformerEncoder.forward ---
        # We cannot call super().forward because we need to inject code in the middle of the loop.
        
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        )  # (1, C, Fr, T1)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  # now T2, B, C
        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        cross_layer_idx = 0
        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)
                
                # --- INJECT FUSION BLOCKS ---
                if external_latents is not None:
                     units = self.fusion_layers[cross_layer_idx]
                     # x is (B, L, C)
                     for fusion_self, fusion_cross in units:
                         x = fusion_self(x)
                         x = fusion_cross(x, external_latents, key_padding_mask=latent_mask)
                     cross_layer_idx += 1
                # ----------------------------

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt


class InternalFusionHTDemucs(HTDemucs):
    """
    HTDemucs with internal transformer latent fusion.
    Inherits from HTDemucs to reuse encoder/decoder and logic.
    Replaces the CrossTransformerEncoder with a fusion-capable one.
    """

    @capture_init
    def __init__(
        self,
        # Standard HTDemucs args (captured via kwargs)
        # We explicitly list fusion args
        use_internal_fusion=True,
        freeze_encoder=False,
        latent_sources=["bs_roformer", "scnet_xl"],
        num_latent_blocks=2,
        use_gradient_checkpointing=False,
        normalize=True,
        # Capture the rest
        **kwargs,
    ):
        """
        Args:
            use_internal_fusion: Enable internal latent fusion in transformer
            freeze_encoder: Freeze encoder parameters (only train transformer/decoder)
            latent_sources: List of external latent sources to fuse
            num_latent_blocks: Number of latent fusion blocks to add
            normalize: Whether to apply normalization internally (default True for HTDemucs)
            **kwargs: Arguments passed to HTDemucs
        """
        # Initialize standard HTDemucs
        super().__init__(**kwargs)

        self.use_internal_fusion = use_internal_fusion
        self.latent_sources = latent_sources
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # self.normalize is not used because we use unconditional normalization in forward 
        # (copied from HTDemucs)

        # Replace the standard transformer with our Interleaved Fusion Transformer
        if self.crosstransformer:
            # Re-calculate transformer dimension as CrossTransformerEncoder doesn't expose it
            channels = kwargs.get('channels', 48)
            growth = kwargs.get('growth', 2)
            depth = kwargs.get('depth', 4)
            bottom_channels = kwargs.get('bottom_channels', 0)
            
            transformer_channels = channels * growth ** (depth - 1)
            if bottom_channels > 0:
                transformer_channels = bottom_channels
            
            # We reconstruct the transformer using the exact same arguments
            # passed to HTDemucs, but utilizing our Interleaved class.
            self.crosstransformer = InterleavedFusionTransformer(
                dim=transformer_channels,
                emb=kwargs.get('t_emb', "sin"),
                hidden_scale=kwargs.get('t_hidden_scale', 4.0),
                num_heads=kwargs.get('t_heads', 8),
                num_layers=kwargs.get('t_layers', 5),
                cross_first=kwargs.get('t_cross_first', False),
                dropout=kwargs.get('t_dropout', 0.0),
                max_positions=kwargs.get('t_max_positions', 10000),
                norm_in=kwargs.get('t_norm_in', True),
                norm_in_group=kwargs.get('t_norm_in_group', False),
                group_norm=kwargs.get('t_group_norm', False),
                norm_first=kwargs.get('t_norm_first', True),
                norm_out=kwargs.get('t_norm_out', True),
                max_period=kwargs.get('t_max_period', 10000.0),
                weight_decay=kwargs.get('t_weight_decay', 0.0),
                lr=kwargs.get('t_lr', None),
                layer_scale=kwargs.get('t_layer_scale', True),
                gelu=kwargs.get('t_gelu', True),
                sin_random_shift=kwargs.get('t_sin_random_shift', 0),
                weight_pos_embed=kwargs.get('t_weight_pos_embed', 1.0),
                cape_mean_normalize=kwargs.get('t_cape_mean_normalize', True),
                cape_augment=kwargs.get('t_cape_augment', True),
                cape_glob_loc_scale=kwargs.get('t_cape_glob_loc_scale', [5000.0, 1.0, 1.4]),
                sparse_self_attn=kwargs.get('t_sparse_self_attn', False),
                sparse_cross_attn=kwargs.get('t_sparse_cross_attn', False),
                mask_type=kwargs.get('t_mask_type', "diag"),
                mask_random_seed=kwargs.get('t_mask_random_seed', 42),
                sparse_attn_window=kwargs.get('t_sparse_attn_window', 500),
                global_window=kwargs.get('t_global_window', 100),
                sparsity=kwargs.get('t_sparsity', 0.95),
                auto_sparsity=kwargs.get('t_auto_sparsity', False),
                
                # Fusion specific args
                num_latent_blocks=num_latent_blocks,
            )

            if use_internal_fusion:
                self.latent_preprocessor = LatentPreprocessor(
                    model_channels=transformer_channels,
                    latent_sources=latent_sources,
                )

        if freeze_encoder:
            self.freeze_encoder_parameters()

    def freeze_encoder_parameters(self):
        print("Freezing encoder parameters...")
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.tencoder.named_parameters():
            param.requires_grad = False
        if self.freq_emb is not None:
            for param in self.freq_emb.parameters():
                param.requires_grad = False
        frozen_params = sum(p.numel() for p in self.encoder.parameters()) + \
                        sum(p.numel() for p in self.tencoder.parameters())
        if self.freq_emb is not None:
            frozen_params += sum(p.numel() for p in self.freq_emb.parameters())
        print(f"Frozen {frozen_params:,} encoder parameters")

    def unfreeze_encoder_parameters(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.tencoder.parameters():
            param.requires_grad = True
        if self.freq_emb is not None:
            for param in self.freq_emb.parameters():
                param.requires_grad = True

    # Override forward to inject latents
    def forward(self, mix, latents=None):
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

        # Normalize (Unconditional)
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
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        # Transformer
        if self.crosstransformer:
            freq_dim = x.shape[2]
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            if self.use_internal_fusion:
                processed_latents = None
                latent_mask = None
                if latents is not None and len(latents) > 0:
                    processed_latents, latent_mask = self.latent_preprocessor(
                        latents, batch_size=B
                    )
                x, xt = self.crosstransformer(x, xt, processed_latents, latent_mask)
            else:
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
    model_config = getattr(args, args.model)
    if hasattr(model_config, "to_dict"):
        kw = model_config.to_dict()
    else:
        kw = dict(model_config)
    model = InternalFusionHTDemucs(**extra, **kw)
    return model