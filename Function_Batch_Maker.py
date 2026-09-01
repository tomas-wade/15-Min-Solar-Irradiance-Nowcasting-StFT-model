'''creates batches out of raw samples using the single_sample_to_tensor script'''
# Function_Batch_Maker.py
import os
import sys
import glob
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

from Function_Single_Sample_to_Tensor import prepare_sample


#thin PyTorch Dataset wrapper - each item is loaded/prepared lazily on access rather than all upfront
class StFTDataset(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = list(file_paths)
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        p = self.file_paths[idx]
        sample = prepare_sample(p) #delegates the actual npz->tensor conversion to Function_Single_Sample_to_Tensor
        return sample


#custom collate function: turns a list of per-sample dicts into one batched dict of stacked tensors
def stft_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert isinstance(batch, list) and len(batch) > 0
    B = len(batch)
    first = batch[0]

    # ERA5 stack: ensure channel axis exists (B,6,C,5,5)
    era5 = torch.stack([torch.as_tensor(s['era5'], dtype=torch.float32) for s in batch], dim=0)
    if era5.dim() == 4:
        era5 = era5.unsqueeze(2) #adds a channel axis for legacy single-channel samples, matching the per-sample normalisation

    # Quarter-hour baseline + residual (required)
    aifs_baseline_quarter = torch.stack([torch.as_tensor(s['aifs_baseline_quarter'], dtype=torch.float32) for s in batch], dim=0)  # (B,4)
    residual_quarter = torch.stack([torch.as_tensor(s['residual_quarter'], dtype=torch.float32) for s in batch], dim=0)  # (B,4)

    # optional ground_quarter
    ground_quarter = None
    if 'ground_quarter' in first: #only stacked if present - lets this collate fn also serve inference-only batches without ground truth
        ground_quarter = torch.stack([torch.as_tensor(s['ground_quarter'], dtype=torch.float32) for s in batch], dim=0)

    # MSG patches: canonical key is 'msg_patches'
    msg_patches = torch.stack([torch.as_tensor(s['msg_patches'], dtype=torch.float32) for s in batch], dim=0)  # (B,T,P,C_msg,h,w)
    msg_patch_valid_ratio = torch.stack([torch.as_tensor(s['msg_patch_valid_ratio'], dtype=torch.float32) for s in batch], dim=0)  # (B,T,P)

    B, T, P, C_msg, h, w = msg_patches.shape
    S_msg = T * P #flattens the (time, patch) grid into one token sequence length for the transformer

    token_valid_mask = (msg_patch_valid_ratio.view(B, S_msg) > 0.0) #a token counts as valid if it has any non-NaN coverage at all

    # msg_time_offsets: use first sample's offsets (warn if inconsistent)
    #these offsets are fixed constants defined in prepare_sample, so every sample should agree - this just guards against surprises
    first_offsets = first['msg_time_offsets_minutes']
    msg_time_offsets = torch.as_tensor(first_offsets, dtype=torch.float32)
    for s in batch[1:]:
        other_t = torch.as_tensor(s['msg_time_offsets_minutes'], dtype=torch.float32)
        if not torch.allclose(other_t, msg_time_offsets):
            print("Warning: msg_time_offsets differ across samples; using first sample's offsets.")
            break

    #expands the per-timestep offset (T,) out to one value per token (S_msg,) since all P patches at a timestep share its offset
    msg_token_time_offsets_1 = msg_time_offsets.unsqueeze(1).repeat(1, P).view(-1)  # (S_msg,)
    msg_token_time_offsets = msg_token_time_offsets_1.unsqueeze(0).repeat(B, 1)     # (B,S_msg)

    #era5 offsets, patch centres and top-left corners are also fixed/constant, so just taken from the first sample in the batch
    era5_time_offsets = first.get('era5_time_offsets_hours', None)
    era5_time_offsets_t = torch.as_tensor(era5_time_offsets, dtype=torch.float32) if era5_time_offsets is not None else None

    centres_idx = first.get('msg_patch_centres', None)
    top_left = first.get('top_left_indices', None)

    meta = [s.get('meta', None) for s in batch] #kept as a plain python list since each entry is a dict, not stackable into a tensor

    out = {
        'era5': era5,                                      # (B,6,C,5,5)
        'aifs_baseline_quarter': aifs_baseline_quarter,    # (B,4)
        'residual_quarter': residual_quarter,              # (B,4)
        'ground_quarter': ground_quarter,                  # (B,4) or None
        'msg_patches': msg_patches,                        # (B,T,P,C_msg,h,w)
        'msg_patch_valid_ratio': msg_patch_valid_ratio,    # (B,T,P)
        'msg_token_valid_mask': token_valid_mask,          # (B,S_msg)
        'msg_token_time_offsets': msg_token_time_offsets,  # (B,S_msg)
        'msg_time_offsets_minutes': msg_time_offsets,      # (T,)
        'era5_time_offsets_hours': era5_time_offsets_t,    # (6,) or None
        'msg_token_centres': centres_idx,
        'top_left_indices': top_left,
        'meta': meta
    }
    return out


#CLI helper: resolves a mix of directory/file arguments into a flat list of .npz paths, defaulting to samples_out/ if none given
def find_npz_files_from_args(args: List[str]) -> List[str]:
    paths = []
    if len(args) == 0:
        paths = sorted(glob.glob("samples_out/*.npz"))
    else:
        for a in args:
            if os.path.isdir(a):
                found = sorted(glob.glob(os.path.join(a, "*.npz")))
                paths.extend(found)
            elif os.path.isfile(a) and a.lower().endswith(".npz"):
                paths.append(a)
    return paths


#debug entry point: builds one real batch from actual sample files and prints its shapes/sanity checks
if __name__ == "__main__":
    args = sys.argv[1:]
    file_list = find_npz_files_from_args(args)
    if len(file_list) == 0:
        print("No .npz files found. Provide sample paths or a directory containing .npz files.")
        sys.exit(1)

    file_list = file_list[:8]  # debug subset

    ds = StFTDataset(file_list)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=stft_collate_fn, num_workers=0)

    batch = next(iter(dl)) #pulls just one batch for inspection
    print("Batched keys and shapes/dtypes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)} (len={len(v) if hasattr(v,'__len__') else 'N/A'})")

    B, T, P = batch['msg_patches'].shape[0:3]
    S_msg = T * P
    print(f"\nSanity checks:")
    print(f"  B={B}, T={T}, P={P}, S_msg={S_msg}")
    print(f"  msg_token_valid_mask.sum() = {batch['msg_token_valid_mask'].sum().item()}")
    print(f"  msg_token_time_offsets[0,:10] = {batch['msg_token_time_offsets'][0,:10].tolist()}")
    print(f"  first meta[0] = {batch['meta'][0]}")
