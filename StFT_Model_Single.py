'''
- predicts 15 min intervals for hour ahead for single location
'''

import torch
import torch.nn as nn
import math
import datetime as _dt
import torch.nn.functional as F
from pysolar.solar import get_altitude as _get_solar_altitude

# encoder for MSG (unchanged)
#small CNN that independently encodes every (timestep, patch) MSG image into a D-dim token
class PatchEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden=64, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.AdaptiveAvgPool2d((2,2)) #collapses spatial size down to a fixed 2x2 regardless of input patch size
        self.mlp = nn.Sequential(
            nn.Linear(64*2*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, patches):
        B, T, P, C, h, w = patches.shape
        S = T * P #flattens (time, patch) into one token sequence dimension
        x = patches.view(B * S, C, h, w) #treats every (time,patch) combination as an independent image for the CNN
        x = self.conv(x)
        x = self.pool(x).view(B * S, -1)
        x = self.mlp(x)
        x = x.view(B, S, -1)
        return x

# FourierEncoder (expects 2D spatial input per-channel)
#encodes the spatial pattern of one ERA5 channel over time using its low-frequency 2D Fourier components
class FourierEncoder(nn.Module):
    def __init__(self, in_H=5, in_W=5, low_k=3, out_dim=128, hidden=128):
        super().__init__()
        self.in_H = in_H
        self.in_W = in_W
        self.low_k = low_k
        feat_len = low_k * low_k * 2 #magnitude + phase for each of the low_k x low_k retained frequency components
        self.proj = nn.Sequential(
            nn.Linear(feat_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):  # x: (B, T, H, W)
        B, T, H, W = x.shape
        assert H == self.in_H and W == self.in_W, f"FourierEncoder expects {self.in_H}x{self.in_W}"
        x_fft = torch.fft.rfft2(x, dim=(-2, -1)) #2D real FFT over the spatial dims
        Hf, Wf = x_fft.shape[-2], x_fft.shape[-1]
        k_h = min(self.low_k, Hf)
        k_w = min(self.low_k, Wf)
        low = x_fft[..., :k_h, :k_w] #keeps only the lowest-frequency components - the coarse/large-scale spatial pattern
        real = low.real.reshape(B, T, -1)
        imag = low.imag.reshape(B, T, -1)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-6) #magnitude of each retained frequency
        phase = torch.atan2(imag, real) #phase of each retained frequency
        feat = torch.cat([mag, phase], dim=-1)
        expected = self.low_k * self.low_k * 2
        if feat.shape[-1] < expected: #pads with zeros if the grid was smaller than low_k (shouldn't normally happen for a 5x5 grid)
            pad = feat.new_zeros(B, T, expected - feat.shape[-1])
            feat = torch.cat([feat, pad], dim=-1)
        mu = feat.mean(dim=-1, keepdim=True)
        sigma = feat.std(dim=-1, keepdim=True) + 1e-6
        feat = (feat - mu) / sigma #per-token normalisation of the spectral feature vector before projecting
        feat = feat.view(B * T, -1)
        tok = self.proj(feat).view(B, T, -1)
        return tok

# SpatialEncoder (can accept multi-channel when x.dim()==5)
#simpler alternative to the Fourier encoder - just flattens the raw spatial grid per timestep and passes it through an MLP
class SpatialEncoder(nn.Module):
    def __init__(self, in_H=5, in_W=5, in_ch=1, out_dim=128, hidden=128):
        super().__init__()
        self.in_H = in_H
        self.in_W = in_W
        self.in_ch = in_ch
        in_size = in_ch * in_H * in_W
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        if x.dim() == 5: #multi-channel input (B,T,C,H,W)
            B, T, C, H, W = x.shape
            feat = x.view(B, T, C * H * W)
        else: #single-channel input (B,T,H,W)
            B, T, H, W = x.shape
            feat = x.view(B, T, H * W)
        feat = feat.view(B * T, -1)
        tok = self.mlp(feat).view(B, T, -1)
        return tok

# AttnPool (unchanged)
#learned single-query attention pooling - collapses a variable-length, partially-masked token sequence into one summary vector
class AttnPool(nn.Module):
    def __init__(self, D, num_heads=4, dropout=0.0):
        super().__init__()
        assert D % max(1, num_heads) == 0, "D must be divisible by num_heads"
        self.q = nn.Parameter(torch.randn(1, 1, D) * 0.02) #a single learned query vector shared across all samples
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=num_heads, batch_first=True, dropout=dropout)

    def forward(self, tokens, key_padding_mask=None):
        B, S, D = tokens.shape
        q = self.q.expand(B, -1, -1)
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=1) #samples where every single token is masked out (no valid data at all)
        else:
            all_masked = torch.zeros(B, dtype=torch.bool, device=tokens.device)
        out = torch.zeros(B, 1, D, device=tokens.device, dtype=tokens.dtype) #defaults to a zero vector for fully-masked samples
        if (~all_masked).any(): #only runs attention on the samples that actually have at least one valid token, avoiding a crash
            keep_idx = (~all_masked).nonzero(as_tuple=False).squeeze(-1)
            tokens_keep = tokens[keep_idx]
            kpm_keep = key_padding_mask[keep_idx] if key_padding_mask is not None else None
            out_keep, _ = self.attn(q[keep_idx], tokens_keep, tokens_keep, key_padding_mask=kpm_keep)
            out[keep_idx] = out_keep
        return out.squeeze(1)

# Updated full model with per-channel Fourier + updated flat proj
#the full StFT fusion model (version 7 / "single-site"): fuses MSG cloud imagery with ERA5 reanalysis to predict a residual correction on top of the AIFS baseline
class StFTFusionModelSingle(nn.Module):
    """
    Updated to produce 4 outputs corresponding to t_init + [15,30,45,60] minutes.
    The rest of the model architecture is unchanged.
    """
    def __init__(
        self,
        D=128,
        spec_dim=64,            # interpreted as per-channel spectral dim
        era5_ch=3,              # number of ERA5 channels
        era5_feat_dim=64,
        attn_heads=8,
        n_fusion_layers=2,
        fusion_mlp_ratio=4,
        dropout=0.1,
        center_lat: float = None,
        center_lon: float = None,
    ):
        super().__init__()

        self.center_lat = float(center_lat) if center_lat is not None else None #fallback site coords, used if a sample's meta lacks them
        self.center_lon = float(center_lon) if center_lon is not None else None

        # record era5 channel count locally
        self.era5_ch = era5_ch
        self.spec_dim_per_ch = spec_dim
        self.spec_dim_total = spec_dim * max(1, era5_ch)

        # encoders
        self.msg_encoder = PatchEncoder(in_channels=2, out_dim=D)
        self.era5_spatial = SpatialEncoder(in_H=5, in_W=5, in_ch=self.era5_ch, out_dim=D)
        self.era5_fourier = nn.ModuleList([ #one independent Fourier encoder per ERA5 channel
            FourierEncoder(in_H=5, in_W=5, low_k=3, out_dim=self.spec_dim_per_ch)
            for _ in range(max(1, self.era5_ch))
        ])

        self.spec_to_D = nn.Linear(self.spec_dim_total, D) #projects the concatenated per-channel spectral features up to D

        # norms / layers
        self.msg_ln = nn.LayerNorm(D)
        self.spec_ln = nn.LayerNorm(D)
        self.all_ln = nn.LayerNorm(D)
        self.spec_pool_ln = nn.LayerNorm(D)
        self.patch_ratio_proj = nn.Sequential( #turns each MSG token's valid-pixel-ratio scalar into a D-dim additive feature
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.cloud_proj = nn.Sequential( #turns the sample's cloud fraction scalar into a D-dim embedding
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        self.mod_emb = nn.Embedding(3, D) #learned embedding identifying which of the 3 token "modalities" (era5-spatial/era5-spectral/msg) a token is
        self.time_proj = nn.Linear(1, D) #projects a scalar time-offset into a D-dim embedding, added to tokens as positional info

        self.cross_attn = nn.MultiheadAttention(embed_dim=D, num_heads=attn_heads, batch_first=True, dropout=dropout)
        self.cross_ln = nn.LayerNorm(D)

        encoder_layer = nn.TransformerEncoderLayer(d_model=D, nhead=attn_heads,
                                                   dim_feedforward=int(D * fusion_mlp_ratio),
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_fusion_layers) #joint self-attention fusion over every token

        # era5_flat_proj accepts flattened multi-channel maps
        flat_in = 6 * max(1, self.era5_ch) * 5 * 5
        self.era5_flat_proj = nn.Sequential( #a simple MLP on the raw flattened ERA5 values, bypassing the fancy encoders entirely
            nn.Linear(flat_in, era5_feat_dim),
            nn.ReLU(),
            nn.Linear(era5_feat_dim, era5_feat_dim)
        )

        self.init_time_proj = nn.Linear(3, D) #projects [sin(hour), cos(hour), solar_altitude_norm] into a D-dim embedding

        # pooling & head: head now outputs 4 values (quarter-hour predictions)
        self.msg_pool_proj = nn.Linear(D, D)
        self.spec_pool_proj = nn.Linear(D, D)
        fusion_dim = D + D + era5_feat_dim + D #msg_pooled + spec_pooled + raw era5 features + init-time embedding
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, D),
            nn.ReLU(),
            nn.Linear(D, 4)   # <-- 4 outputs: 15,30,45,60 min
        )

        self.msg_attnpool = AttnPool(D, num_heads=max(1, attn_heads // 2), dropout=dropout)
        self.spec_attnpool = AttnPool(D, num_heads=max(1, attn_heads // 2), dropout=dropout)

    #adds a modality-identity embedding and a time-offset embedding onto a set of tokens - acts as a lightweight positional encoding
    def _add_mod_time_embeddings(self, tokens, mod_id, time_offsets):
        B, S, D = tokens.shape
        dev = tokens.device
        mod = self.mod_emb(torch.tensor([mod_id], device=dev)).view(1, 1, D)
        mod = mod.expand(B, S, D)
        if time_offsets is None:
            time_emb = torch.zeros(B, S, D, device=dev)
        else:
            if time_offsets.dim() == 1: #same offsets shared across the whole batch
                t = time_offsets.unsqueeze(0).unsqueeze(-1).to(dev)
                t = t.expand(B, S, 1)
            else: #per-sample offsets already provided
                t = time_offsets.unsqueeze(-1).to(dev)
            time_emb = self.time_proj(t)
        return tokens + mod + time_emb

    def forward(self, batch):
        device = next(self.parameters()).device

        # MSG
        msg_patches = batch['msg_patches']
        msg_mask = batch['msg_token_valid_mask']
        B = msg_patches.shape[0]
        msg_tokens = self.msg_encoder(msg_patches) #(B, S_msg, D)

        #pulls cloud fraction at t_init out of each sample's meta dict (defaults to 0 if missing/unparseable)
        cloud_vals = []
        meta = batch.get('meta', None)
        if isinstance(meta, list) and len(meta) == B and isinstance(meta[0], dict):
            for m in meta:
                try:
                    cloud_vals.append(float(m.get('cloud_frac_at_t_init', 0.0)))
                except Exception:
                    cloud_vals.append(0.0)
        else:
            cloud_vals = [0.0] * B

        cloud_t = torch.tensor(cloud_vals, dtype=msg_tokens.dtype, device=msg_tokens.device).unsqueeze(-1)  # (B,1)
        cloud_emb = self.cloud_proj(cloud_t)  # (B, D) - computed but not currently added into any downstream tensor in this version

        msg_tokens = self.msg_ln(msg_tokens)
        pv = batch.get('msg_patch_valid_ratio', None)  # stored as (B, T, P) or (B, S)
        if pv is not None:
            # normalize/convert to tensor and reshape to (B, S, 1)
            if isinstance(pv, torch.Tensor):
                pv_t = pv
            else:
                pv_t = torch.as_tensor(pv, device=msg_tokens.device, dtype=msg_tokens.dtype)

            # if shape is (B, T, P) -> flatten to (B, S)
            if pv_t.dim() == 3:
                Bp, Tp, Pp = pv_t.shape
                pv_t = pv_t.view(Bp, Tp * Pp)
            elif pv_t.dim() == 2:
                # already (B, S) or (B, some)
                pass
            else:
                pv_t = pv_t.view(pv_t.shape[0], -1)

            # ensure length matches S
            S_msg = msg_tokens.shape[1]
            if pv_t.shape[1] != S_msg:
                # Try to repeat or truncate to match; prefer truncation if too long
                if pv_t.shape[1] > S_msg:
                    pv_t = pv_t[:, :S_msg]
                else:
                    pv_t = pv_t.repeat(1, math.ceil(S_msg / max(1, pv_t.shape[1])))[:, :S_msg]

            pv_t = pv_t.unsqueeze(-1)  # (B, S, 1)

            # project to embedding dim
            pv_proj = self.patch_ratio_proj(pv_t)  # (B, S, D)
            msg_tokens = msg_tokens + pv_proj #adds patch-reliability info directly onto each msg token

        # ERA5: accept either (B,6,5,5) or (B,6,C,5,5)
        era5 = batch['era5']
        if era5.dim() == 4:
            era5 = era5.unsqueeze(2)  # (B,6,1,5,5)

        B_e, T_e, C_e, H_e, W_e = era5.shape

        era5_sp = self.era5_spatial(era5)  # (B, T, D) - raw-spatial token stream

        # per-channel Fourier
        spec_tokens_per_ch = []
        for ch_idx in range(C_e):
            ch_slice = era5[:, :, ch_idx, :, :]
            enc_idx = ch_idx if ch_idx < len(self.era5_fourier) else -1 #reuses the last encoder if there are more channels than encoders
            ch_spec = self.era5_fourier[enc_idx](ch_slice)  # (B, T, spec_dim_per_ch)
            spec_tokens_per_ch.append(ch_spec)
        spec_concat = torch.cat(spec_tokens_per_ch, dim=-1)
        if spec_concat.shape[-1] != self.spec_dim_total: #pads or truncates in case the channel count didn't exactly match expectations
            cur = spec_concat.shape[-1]
            if cur < self.spec_dim_total:
                pad = spec_concat.new_zeros(B_e, T_e, self.spec_dim_total - cur)
                spec_concat = torch.cat([spec_concat, pad], dim=-1)
            else:
                spec_concat = spec_concat[..., :self.spec_dim_total]
        era5_spec_p = self.spec_to_D(spec_concat)
        era5_spec_p = self.spec_ln(era5_spec_p)

        # modality + time embeddings
        era5_time = batch.get('era5_time_offsets_hours', None)
        msg_time = batch.get('msg_time_offsets_minutes', None)
        era5_sp_e = self._add_mod_time_embeddings(era5_sp, mod_id=0, time_offsets=era5_time)
        era5_spec_e = self._add_mod_time_embeddings(era5_spec_p, mod_id=1, time_offsets=era5_time)

        # MSG token offsets
        if msg_time is not None:
            T_msg = msg_time.shape[0]
            P = S_msg // T_msg #patches per timestep, derived from total tokens / number of timesteps
            msg_token_offsets = msg_time.repeat_interleave(P) #broadcasts each timestep's offset across its P patches
        else:
            msg_token_offsets = None
        msg_e = self._add_mod_time_embeddings(msg_tokens, mod_id=2, time_offsets=msg_token_offsets)

        # cross-attention
        #lets the msg tokens (query) attend over the era5 spatial+spectral tokens (key/value) before joint fusion
        spec_keys = torch.cat([era5_sp_e, era5_spec_e], dim=1)
        S_spec = spec_keys.shape[1]
        attn_out, attn_weights = self.cross_attn(query=msg_e, key=spec_keys, value=spec_keys)
        msg_attended = self.cross_ln(msg_e + attn_out)

        # fusion transformer
        all_tokens = torch.cat([msg_attended, spec_keys], dim=1)
        all_tokens = self.all_ln(all_tokens)

        msg_padding_mask = ~batch['msg_token_valid_mask'] #True where a msg token should be ignored by attention
        spec_padding_mask = torch.zeros(B, S_spec, dtype=torch.bool, device=device) #era5 tokens are always considered valid
        src_key_padding_mask = torch.cat([msg_padding_mask, spec_padding_mask], dim=1)

        fused = self.fusion_transformer(all_tokens, src_key_padding_mask=src_key_padding_mask)

        # split fused parts
        idx = 0
        msg_fused = fused[:, idx: idx + S_msg, :]; idx += S_msg
        s_era5_sp = era5_sp_e.shape[1]; s_era5_spec = era5_spec_e.shape[1]
        era5_sp_fused = fused[:, idx: idx + s_era5_sp, :]; idx += s_era5_sp
        era5_spec_fused = fused[:, idx: idx + s_era5_spec, :]; idx += s_era5_spec

        # pooling
        msg_padding_mask = ~batch['msg_token_valid_mask']
        msg_pooled = self.msg_attnpool(msg_fused, key_padding_mask=msg_padding_mask) #collapses the fused msg tokens into one vector
        msg_pooled = self.msg_pool_proj(msg_pooled)

        era5_spec_vec = self.spec_attnpool(era5_spec_fused, key_padding_mask=None) #collapses the fused spectral tokens into one vector
        spec_pooled = self.spec_pool_proj(era5_spec_vec)
        spec_pooled = self.spec_pool_ln(spec_pooled)

        # flat era5 features
        era5_flat = era5.view(B_e, -1)
        era5_feat = self.era5_flat_proj(era5_flat) #a raw, non-encoded view of the ERA5 grid, given directly to the head as well

        # init time sin/cos conditioning
        #cyclical time-of-day encoding: converts t_init's hour into an angle around a 24h circle, so 23:59 and 00:01 read as close together
        meta = batch.get('meta', None)
        if isinstance(meta, list) and len(meta) == B and meta[0] is not None and isinstance(meta[0], dict):
            hours_list = []
            for m in meta:
                try:
                    tinit_iso = m.get('t_init')
                    tinit = _dt.datetime.fromisoformat(tinit_iso)
                    if tinit.tzinfo is None:
                        tinit = tinit.replace(tzinfo=_dt.timezone.utc)
                    hour = tinit.hour + tinit.minute / 60.0 + tinit.second / 3600.0
                    ang = 2.0 * math.pi * (hour / 24.0)
                    hours_list.append([math.sin(ang), math.cos(ang)])
                except Exception:
                    hours_list.append([0.0, 0.0])
            t_sincos = torch.tensor(hours_list, dtype=torch.float32, device=device)
        else:
            t_sincos = torch.zeros((B, 2), dtype=torch.float32, device=device)

        # compute solar altitude per-sample (unchanged)
        #gives the model a direct physical signal for how high the sun is - strongly predictive of irradiance on its own
        solar_vals = []
        for i in range(B):
            alt_deg = 0.0
            lat = None; lon = None
            if isinstance(meta, list) and len(meta) == B and isinstance(meta[0], dict):
                lat = meta[i].get('center_lat', None)
                lon = meta[i].get('center_lon', None)
                if lat is None and 'center' in meta[i] and isinstance(meta[i]['center'], dict):
                    lat = meta[i]['center'].get('lat', None)
                if lon is None and 'center' in meta[i] and isinstance(meta[i]['center'], dict):
                    lon = meta[i]['center'].get('lon', None)
            if lat is None or lon is None: #falls back to the model's own configured site coords if the sample's meta lacks them
                lat = self.center_lat
                lon = self.center_lon
            if (lat is not None) and (lon is not None):
                try:
                    tinit_iso = None
                    if isinstance(meta, list) and len(meta) == B and isinstance(meta[0], dict):
                        tinit_iso = meta[i].get('t_init')
                    if tinit_iso is None:
                        alt_deg = 0.0
                    else:
                        tinit = _dt.datetime.fromisoformat(tinit_iso)
                        if tinit.tzinfo is None:
                            tinit = tinit.replace(tzinfo=_dt.timezone.utc)
                        alt_deg = float(_get_solar_altitude(float(lat), float(lon), tinit)) #actual astronomical solar altitude calc
                except Exception:
                    alt_deg = 0.0
            else:
                if isinstance(meta, list) and len(meta) == B and isinstance(meta[0], dict) and (
                        'solar_altitude_deg' in meta[i]):
                    try:
                        alt_deg = float(meta[i].get('solar_altitude_deg', 0.0)) #uses a precomputed value if lat/lon truly unavailable
                    except Exception:
                        alt_deg = 0.0
                else:
                    alt_deg = 0.0
            solar_vals.append(alt_deg)

        solar_arr = torch.tensor(solar_vals, dtype=torch.float32, device=device).unsqueeze(-1)
        solar_norm = torch.clamp(solar_arr / 90.0, min=-1.0, max=1.0) #normalises degrees to roughly [-1,1] (90 deg = directly overhead)

        # concatenate sin/cos + solar_norm -> (B,3)
        t_input = torch.cat([t_sincos.to(device).float(), solar_norm.to(device).float()], dim=1)  # (B,3)
        t_emb = self.init_time_proj(t_input)

        # final fusion & head
        concat = torch.cat([msg_pooled, spec_pooled, era5_feat, t_emb], dim=1)
        raw_residual = self.head(concat)  # (B, 4)  raw residuals for 15/30/45/60 min

        # Baseline handling: support several formats in batch for backward compatibility.
        # Expectation for new scheme: batch contains key 'aifs_baseline_quarter' or 'aifs_baseline_hourly'
        # - 'aifs_baseline_quarter' should be shape (B,4) (values at 15,30,45,60)
        # - If only hourly baseline (B,6) present, we attempt to use/interpolate the first hour into 4 quarters by simple slicing:
        #    * if (B,6) -> use first hourly value for each quarter as fallback (user should prepare correct 15-min baseline if available)
        aifs_q = None
        # prefer explicit quarter baseline
        if 'aifs_baseline_quarter' in batch:
            aifs_q = batch['aifs_baseline_quarter']
        elif 'aifs_baseline_15min' in batch:
            aifs_q = batch['aifs_baseline_15min']
        elif 'aifs_baseline_hourly' in batch:
            # could be shape (B,6) or (B,)
            aifs_raw = batch['aifs_baseline_hourly']
            # convert to tensor if needed
            aifs_t = torch.as_tensor(aifs_raw, device=raw_residual.device, dtype=raw_residual.dtype)
            if aifs_t.dim() == 2 and aifs_t.shape[1] >= 4:
                # pick first 4 entries as a simple strategy (user should ideally provide interpolated 15-min baseline)
                aifs_q = aifs_t[:, :4]
            elif aifs_t.dim() == 1:
                # single-value per sample -> expand to 4
                aifs_q = aifs_t.unsqueeze(-1).repeat(1, 4)
            else:
                # fallback: try to reshape to (B,1) and repeat
                try:
                    aifs_q = aifs_t.reshape(aifs_t.shape[0], -1)[:, :1].repeat(1, 4)
                except Exception:
                    aifs_q = None
        else:
            # no baseline provided in batch
            aifs_q = None

        if aifs_q is None:
            # missing baseline -> just return raw residuals (shape (B,4))
            out = raw_residual
        else:
            # ensure tensor on correct device/dtype and proper shape (B,4)
            aifs_q = torch.as_tensor(aifs_q, device=raw_residual.device, dtype=raw_residual.dtype)
            if aifs_q.dim() == 1:
                aifs_q = aifs_q.unsqueeze(-1).repeat(1, 4)
            elif aifs_q.dim() == 2 and aifs_q.shape[1] < 4:
                # repeat or pad to 4
                aifs_q = aifs_q.repeat(1, math.ceil(4 / max(1, aifs_q.shape[1])))[:, :4]
            elif aifs_q.dim() == 2 and aifs_q.shape[1] > 4:
                aifs_q = aifs_q[:, :4]

            # smooth enforce elementwise: out >= -aifs_q
            #softplus keeps this smooth/differentiable everywhere while guaranteeing (baseline + prediction) can never go negative,
            #since real solar irradiance is physically non-negative - a hard clamp/ReLU here would zero out the gradient instead
            out = -aifs_q + F.softplus(raw_residual + aifs_q)

        return out  # shape (B,4)
