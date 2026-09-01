# Function_Single_Sample_to_Tensor.py
# Function_Single_Sample_to_Tensor.py
import numpy as np
import torch
import torch.nn.functional as F
import sys

#loads one saved sample .npz file and unpacks its arrays/metadata
def load_npz_sample(path):
    d = np.load(path, allow_pickle=True)

    # Required keys (quarter-hour)
    #fails loudly if any of the arrays the rest of the pipeline depends on are missing
    if 'era5' not in d:
        raise KeyError(f"No 'era5' key found in {path}")
    if 'msg' not in d:
        raise KeyError(f"No 'msg' key found in {path}")
    if 'aifs_baseline_quarter' not in d:
        raise KeyError(f"No 'aifs_baseline_quarter' key found in {path}")
    if 'ground_quarter' not in d:
        raise KeyError(f"No 'ground_quarter' key found in {path}")

    era5 = d['era5']               # expected (6, C, 5, 5) or legacy (6,5,5)
    msg = d['msg']                 # expected (8,13,13)
    aifs_quarter = d['aifs_baseline_quarter']  # (4,)
    ground_quarter = d['ground_quarter']       # (4,)
    meta = d['meta'].item() if 'meta' in d else None #meta was saved as a 0-d object array, .item() unwraps it back to a plain dict

    return era5, msg, aifs_quarter, ground_quarter, meta


#converts the raw MSG cloud-mask grid (0/1/2/NaN codes) into two binary channel maps
def msg_to_channels_and_valid(msg):
    msg_f = np.asarray(msg).astype(np.float32)
    valid = ~np.isnan(msg_f) #a pixel is only invalid if it's NaN - 0/1/2 are all valid values
    cloud = (msg_f == 2).astype(np.float32) #1 where the pixel was classified cloudy
    land  = (msg_f == 1).astype(np.float32) #1 where the pixel was classified land/clear
    cloud[~valid] = 0.0 #zeroes out invalid pixels rather than leaving NaN, so downstream math stays finite
    land[~valid]  = 0.0
    channels = np.stack([cloud, land], axis=1)  # (T, 2, H, W)
    return channels.astype(np.float32), valid


#slides a 5x5 window (stride 2) across the 13x13 MSG grid to build 25 overlapping patches per timestep
def extract_5x5_patches_stride2(channels, valid_mask):
    T, C, H, W = channels.shape
    patch_size = 5
    stride = 2
    starts = [i * stride for i in range(5)] #5 start positions per axis (0,2,4,6,8) -> 5x5=25 patches total
    top_left = [(r, c) for r in starts for c in starts] #top-left corner coordinate of every patch

    ch_t = torch.from_numpy(channels)  # (T, C, H, W)

    patches_per_time = []
    valid_ratio_per_time = []

    #processes one timestep at a time since unfold operates on a single 2D image at once
    for t in range(T):
        x = ch_t[t:t+1]  # (1, C, H, W)
        #unfold extracts every sliding window in one vectorised call instead of a nested python loop over 25 positions
        patches_flat = F.unfold(x, kernel_size=patch_size, stride=stride)  # (1, C*25, P)
        _, c_mul, P = patches_flat.shape
        patches_t = patches_flat.view(1, C, patch_size*patch_size, P)
        patches_t = patches_t.permute(0, 3, 1, 2).contiguous()  # (1, P, C, 25)
        patches_t = patches_t.view(1, P, C, patch_size, patch_size)  # (1, P, C, 5,5)
        patches_per_time.append(patches_t.squeeze(0))  # (P, C, 5,5)

        #same unfold trick applied to the valid mask, so each patch gets its own fraction of usable (non-NaN) pixels
        vm = torch.from_numpy(valid_mask[t].astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        vm_flat = F.unfold(vm, kernel_size=patch_size, stride=stride)  # (1, 25, P)
        vm_flat = vm_flat.view(1, patch_size*patch_size, P)
        vm_flat = vm_flat.permute(0, 2, 1)  # (1, P, 25)
        patch_valid = vm_flat.mean(dim=-1).squeeze(0)  # (P,) - fraction of valid pixels in each patch
        valid_ratio_per_time.append(patch_valid)

    patches = torch.stack(patches_per_time, dim=0)            # (T, P, C, 5,5)
    patch_valid_ratio = torch.stack(valid_ratio_per_time, dim=0)  # (T, P)

    centres = [(r + 2, c + 2) for (r, c) in top_left] #centre pixel of each 5x5 patch (offset 2 from its top-left corner)
    centres_idx = torch.tensor(centres, dtype=torch.long)  # (P,2)

    return patches, centres_idx, patch_valid_ratio, top_left


#top-level function: turns one raw .npz sample into the dict of tensors the model actually consumes
def prepare_sample(npz_path):
    era5, msg, aifs_quarter, ground_quarter, meta = load_npz_sample(npz_path)

    # --- ERA5 shape normalization: (6,C,5,5) ---
    era5_arr = np.asarray(era5)
    if era5_arr.ndim == 3:
        era5_arr = era5_arr[:, np.newaxis, :, :]  # (6,1,5,5) - adds a channel axis for older single-channel samples
    elif era5_arr.ndim == 4:
        pass #already has a channel axis, nothing to do
    else:
        raise RuntimeError(f"Unexpected era5 shape {era5_arr.shape} in {npz_path}")
    era5_arr = era5_arr.astype(np.float32)

    # Quarter-hour targets (required)
    aifs_quarter = np.asarray(aifs_quarter, dtype=np.float32)
    ground_quarter = np.asarray(ground_quarter, dtype=np.float32)
    if aifs_quarter.ndim != 1 or aifs_quarter.shape[0] != 4:
        raise ValueError(f"aifs_quarter must have shape (4,), got {aifs_quarter.shape} in {npz_path}")
    if ground_quarter.ndim != 1 or ground_quarter.shape[0] != 4:
        raise ValueError(f"ground_quarter must have shape (4,), got {ground_quarter.shape} in {npz_path}")

    residual_quarter = ground_quarter - aifs_quarter  # (4,) - the actual training target: how far off the AIFS baseline was

    # MSG -> channels, valid mask, then extract patches
    channels, valid_mask = msg_to_channels_and_valid(msg)  # (T,2,H,W), (T,H,W)
    patches, centres_idx, patch_valid_ratio, top_left = extract_5x5_patches_stride2(channels, valid_mask)

    # Convert to tensors
    era5_t = torch.from_numpy(era5_arr)                         # (6,C,5,5)
    aifs_quarter_t = torch.from_numpy(aifs_quarter)             # (4,)
    ground_quarter_t = torch.from_numpy(ground_quarter)         # (4,)
    residual_quarter_t = torch.from_numpy(residual_quarter)     # (4,)

    #fixed offset labels (in minutes/hours) describing where each MSG/ERA5 frame sits in time - fed into the model as positional info
    msg_time_offsets_minutes = torch.tensor([-105, -90, -75, -60, -45, -30, -15, 0], dtype=torch.float32)
    era5_time_offsets_hours  = torch.tensor([-5, -4, -3, -2, -1, 0], dtype=torch.float32)

    out = {
        'era5': era5_t,                         # (6, C, 5, 5)
        'msg_patches': patches,                 # (T, P, C_msg, 5,5)
        'msg_patch_centres': centres_idx,
        'msg_patch_valid_ratio': patch_valid_ratio,
        'msg_time_offsets_minutes': msg_time_offsets_minutes,
        'era5_time_offsets_hours': era5_time_offsets_hours,
        # quarter-hour fields (required)
        'aifs_baseline_quarter': aifs_quarter_t,   # (4,)
        'ground_quarter': ground_quarter_t,        # (4,)
        'residual_quarter': residual_quarter_t,    # (4,)
        'meta': meta,
        'top_left_indices': top_left
    }

    return out

#debug entry point: prepares a single sample from the command line and prints the shape/dtype of every field
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Function_Single_Sample_to_Tensor.py path/to/sample.npz")
        sys.exit(1)
    p = sys.argv[1]
    sample = prepare_sample(p)
    print("Prepared sample keys and shapes:")
    for k,v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
