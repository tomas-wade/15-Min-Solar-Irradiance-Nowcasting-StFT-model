"""
Combined train + validate + held-out-evaluate script for StFTFusionModelSingle.

Chronological 70/10/20 -> actually 70/20/10 split: first 70% of samples (by t_init,
since filenames sort chronologically) are used for training, the next 20% for
validation during training (drives checkpoint selection + the validation timeseries
plot, same as StFT_Model_Single_Train.py), and the final 10% is held out completely
until after training finishes, then scored via Evaluate.py's own day/night/cloud-bucket
RMSE breakdown - the only genuinely unbiased number, since it was never touched by
training or checkpoint selection.

Supports two modes:
 - Single-dir mode: provide --samples_dir, script splits 70% train / 20% val / 10% eval
 - Three-dir mode: provide --train_dir, --val_dir, and --eval_dir directly

Example single-dir:
    python StFT_Model_Single_Train_Eval.py --samples_dir samples --epochs 20 --save_path stft_single.pt --cloud dataBUD/BUD_cloudmask_sample_prep.csv

Example three-dir:
    python StFT_Model_Single_Train_Eval.py --train_dir samples_train --val_dir samples_val --eval_dir samples_eval --epochs 10
"""
import math
import glob
import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Function_Batch_Maker import StFTDataset, stft_collate_fn
from StFT_Model_Single import StFTFusionModelSingle  # model that returns (B,4) quarter-hour residuals
from Evaluate import process_all  # reuses the day/night/cloud-bucket RMSE + plotting logic unchanged


def _move_batch_to_device(batch: dict, device: torch.device) -> None:
    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        if isinstance(x, list):
            return [move(v) for v in x]
        if isinstance(x, tuple):
            return tuple(move(v) for v in x)
        return x
    for k in list(batch.keys()):
        batch[k] = move(batch[k])


#chronological three-way split: earliest 70% train, next 20% val, final 10% held-out eval
def split_70_20_10(paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    paths = sorted(paths)
    n = len(paths)
    if n == 0:
        raise ValueError("No sample files found.")
    cutoff1 = math.ceil(0.7 * n)
    cutoff2 = math.ceil(0.9 * n)  # 0.7 + 0.2
    return paths[:cutoff1], paths[cutoff1:cutoff2], paths[cutoff2:]


def list_npz(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    paths = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    return paths


#unchanged from StFT_Model_Single_Train.py - trains on train_paths, tracks/checkpoints on val_paths
def train_on_paths(
    train_paths: List[str],
    val_paths: List[str],
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    save_path: str = "stft_single_checkpoint.pt",
    num_workers: int = 0,
    max_train_batches_per_epoch: int = None,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print(f"Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")

    train_ds = StFTDataset(train_paths)
    val_ds = StFTDataset(val_paths) if len(val_paths) > 0 else None

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=stft_collate_fn, num_workers=num_workers)
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=stft_collate_fn, num_workers=num_workers)

    model = StFTFusionModelSingle().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    avg_val_loss = None
    avg_train_loss = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for b_idx, batch in enumerate(train_dl):
            if (max_train_batches_per_epoch is not None) and (b_idx >= max_train_batches_per_epoch):
                break

            _move_batch_to_device(batch, device)
            opt.zero_grad()

            preds = model(batch)                # expected (B,4)
            target = batch['residual_quarter'].to(device)  # (B,4)
            if preds.shape != target.shape:
                raise RuntimeError(f"Model output shape {preds.shape} doesn't match target {target.shape}")

            loss = loss_fn(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())

        avg_train_loss = float(sum(train_losses) / max(1, len(train_losses))) if train_losses else float('nan')

        # validation
        avg_val_loss = None
        if val_dl is not None and len(val_paths) > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_dl:
                    _move_batch_to_device(batch, device)
                    preds = model(batch)
                    target = batch['residual_quarter'].to(device)
                    if preds.shape != target.shape:
                        raise RuntimeError(f"Model output shape {preds.shape} doesn't match target {target.shape} (val)")
                    val_loss = loss_fn(preds, target)
                    val_losses.append(val_loss.item())
            avg_val_loss = float(sum(val_losses) / max(1, len(val_losses))) if val_losses else float('nan')
            print(f"Epoch {epoch:3d}  train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch:3d}  train_loss={avg_train_loss:.6f}  (no val)")

        current_metric = avg_val_loss if avg_val_loss is not None else avg_train_loss
        if current_metric < best_val:
            best_val = current_metric
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  Saved checkpoint to {save_path} (epoch {epoch})")

    print(f"Training finished. Best epoch: {best_epoch}, best loss: {best_val}")
    return {
        "save_path": save_path,
        "best_epoch": best_epoch,
        "best_loss": best_val,
        "final_train_loss": avg_train_loss,
        "final_val_loss": avg_val_loss,
    }


#builds a predictions csv (same columns Evaluate.py's process_all expects) by running a trained model over a fixed set of sample paths
def generate_eval_csv_for_paths(model, paths: List[str], batch_size: int, device: torch.device, out_csv_path: str) -> str:
    ds = StFTDataset(paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=stft_collate_fn, num_workers=0)

    model.eval()
    all_times, all_baseline, all_model, all_ground = [], [], [], []

    with torch.no_grad():
        for batch in dl:
            _move_batch_to_device(batch, device)

            preds = model(batch)  # (B,4) raw residuals
            aifs = batch["aifs_baseline_quarter"]  # (B,4)
            ground = batch["ground_quarter"]       # (B,4)

            ssrd_pred = (aifs + preds).cpu().numpy()
            ssrd_base = aifs.cpu().numpy()
            ssrd_true = ground.cpu().numpy()

            meta_list = batch.get("meta", [None] * ssrd_pred.shape[0])

            for i in range(ssrd_pred.shape[0]):
                m = meta_list[i] if isinstance(meta_list, list) else None
                t_init_iso = m.get("t_init") if isinstance(m, dict) else None
                base_time = pd.to_datetime(t_init_iso) if t_init_iso is not None else pd.Timestamp.now()
                q_times = [base_time + pd.Timedelta(minutes=15 * (j + 1)) for j in range(4)]

                for j in range(4):
                    all_times.append(q_times[j].to_pydatetime())
                    all_baseline.append(float(ssrd_base[i, j]))
                    all_model.append(float(ssrd_pred[i, j]))
                    all_ground.append(float(ssrd_true[i, j]))

    out_df = pd.DataFrame({
        "valid_time": pd.to_datetime(all_times),
        "ssrd_baseline_Jm2": all_baseline,
        "ssrd_model_Jm2": all_model,
        "ssrd_ground_Jm2": all_ground,
    }).sort_values("valid_time").reset_index(drop=True)

    out_df.to_csv(out_csv_path, index=False)
    print(f"[held-out eval] Saved {len(out_df)} quarter-hour predictions to {out_csv_path}")
    return out_csv_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples_dir", type=str,
                        help="Single directory containing .npz samples to split chronologically: "
                             "first 70%% train, next 20%% val, final 10%% held-out eval")
    group.add_argument("--train_dir", type=str, help="Training .npz directory (use with --val_dir and --eval_dir)")

    parser.add_argument("--val_dir", type=str, help="Validation .npz directory (use with --train_dir)")
    parser.add_argument("--eval_dir", type=str, help="Held-out evaluation .npz directory (use with --train_dir)")

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training/validation/eval")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="stft_single_exp.pt", help="Path to save best checkpoint")
    parser.add_argument("--num_workers", type=int, default=0, help="Num workers for DataLoader")
    parser.add_argument("--max_train_batches_per_epoch", type=int, default=None,
                        help="Optional cap on train batches per epoch for quick debug")

    parser.add_argument("--out_preds", type=str, default="val_quarter_predictions.csv",
                        help="CSV to save quarter-hour predictions for the validation (20%%) split")
    parser.add_argument("--out_plot", type=str, default="val_quarter_timeseries.png",
                        help="Output plot path for the validation (20%%) split")

    parser.add_argument("--cloud", type=str, default=None,
                        help="Optional cloudmask CSV (or comma-separated/glob) for the held-out eval's cloud-bucket "
                             "RMSE breakdown, e.g. dataBUD/BUD_cloudmask_sample_prep.csv - the raw msg-step output "
                             "is already in the right format, no conversion needed. If omitted, the held-out eval "
                             "still runs (overall/day/night RMSE), just without the cloud-bucket breakdown.")
    parser.add_argument("--eval_outdir", type=str, default="eval_holdout",
                         help="Output directory for the held-out (10%%) evaluation stage")
    args = parser.parse_args()

    # Resolve train/val/eval paths
    if args.samples_dir:
        all_paths = list_npz(args.samples_dir)
        if len(all_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {args.samples_dir}")
        train_paths, val_paths, eval_paths = split_70_20_10(all_paths)
    else:
        if args.val_dir is None or args.eval_dir is None:
            raise ValueError("When using --train_dir you must also provide --val_dir and --eval_dir")
        train_paths = list_npz(args.train_dir)
        val_paths = list_npz(args.val_dir)
        eval_paths = list_npz(args.eval_dir)
        if len(train_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in train_dir {args.train_dir}")
        if len(val_paths) == 0:
            print(f"Warning: no .npz files found in val_dir {args.val_dir}; validation will be skipped")
        if len(eval_paths) == 0:
            print(f"Warning: no .npz files found in eval_dir {args.eval_dir}; held-out evaluation will be skipped")

    print(f"Resolved: train={len(train_paths)}, val={len(val_paths)}, held-out eval={len(eval_paths)} samples")

    # ---------------- Stage 1+2: train, with validation driving checkpoint selection ----------------
    stats = train_on_paths(
        train_paths=train_paths,
        val_paths=val_paths,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        save_path=args.save_path,
        num_workers=args.num_workers,
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
    )
    print("\nTraining stats:", stats)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- Validation-split evaluation + timeseries plot (unchanged from StFT_Model_Single_Train.py) ----------------
    if len(val_paths) == 0:
        print("No validation paths present - skipping validation-split evaluation.")
    else:
        val_ds = StFTDataset(val_paths)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=stft_collate_fn, num_workers=args.num_workers)

        model = StFTFusionModelSingle().to(device)
        if not os.path.isfile(args.save_path):
            raise FileNotFoundError(f"Checkpoint {args.save_path} not found. Make sure training saved the file.")
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        model.eval()

        all_times, all_baseline, all_model, all_ground = [], [], [], []
        baseline_errs, model_errs = [], []

        with torch.no_grad():
            for batch in val_dl:
                _move_batch_to_device(batch, device)

                B = None
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        B = v.shape[0]
                        break
                if B is None:
                    raise RuntimeError("Unable to infer batch size during evaluation.")

                preds_q = model(batch)  # (B,4)
                aifs_q = batch['aifs_baseline_quarter'].to(device)   # (B,4)
                ground_q = batch.get('ground_quarter', None)
                if ground_q is not None:
                    ground_q = ground_q.to(device)
                else:
                    ground_q = aifs_q + batch['residual_quarter'].to(device)

                def ensure_4(x):
                    x = torch.as_tensor(x, device=device, dtype=preds_q.dtype)
                    if x.dim() == 1:
                        x = x.unsqueeze(0).repeat(B, 1)
                    return x
                preds_q = ensure_4(preds_q)
                aifs_q = ensure_4(aifs_q)
                ground_q = ensure_4(ground_q)

                ssrd_true_q = ground_q
                ssrd_base_q = aifs_q
                ssrd_pred_q = aifs_q + preds_q

                baseline_errs.append(((ssrd_base_q - ssrd_true_q) ** 2).cpu().numpy().ravel())
                model_errs.append(((ssrd_pred_q - ssrd_true_q) ** 2).cpu().numpy().ravel())

                meta_list = batch.get('meta', [None] * B)
                for i_sample in range(B):
                    meta_i = meta_list[i_sample] if isinstance(meta_list, list) else None
                    t_init_iso = meta_i.get("t_init", None) if isinstance(meta_i, dict) else None
                    base_time = pd.to_datetime(t_init_iso) if t_init_iso is not None else pd.Timestamp.now()

                    q_times = [base_time + pd.Timedelta(minutes=15 * (j + 1)) for j in range(4)]
                    for j in range(4):
                        all_times.append(q_times[j].to_pydatetime())
                        all_baseline.append(float(ssrd_base_q[i_sample, j].cpu().item()))
                        all_model.append(float(ssrd_pred_q[i_sample, j].cpu().item()))
                        all_ground.append(float(ssrd_true_q[i_sample, j].cpu().item()))

        if len(baseline_errs) > 0:
            baseline_flat = np.concatenate(baseline_errs)
            model_flat = np.concatenate(model_errs)
            baseline_rmse = math.sqrt(float(baseline_flat.mean()))
            model_rmse = math.sqrt(float(model_flat.mean()))
        else:
            baseline_rmse = float("nan")
            model_rmse = float("nan")

        print("\n=== Validation (20%) SSRD comparison (quarter-hourly) ===")
        print(f"AIFS baseline RMSE : {baseline_rmse:.2f} J/m²")
        print(f"Model RMSE         : {model_rmse:.2f} J/m²")
        print(f"Δ RMSE (baseline - model) : {baseline_rmse - model_rmse:.2f} J/m²")

        out_df = pd.DataFrame({
            "valid_time": pd.to_datetime(all_times),
            "ssrd_baseline_Jm2": all_baseline,
            "ssrd_model_Jm2": all_model,
            "ssrd_ground_Jm2": all_ground,
        }).sort_values("valid_time").reset_index(drop=True)
        out_df.to_csv(args.out_preds, index=False)
        print(f"Saved validation quarter-hour predictions to {args.out_preds}")

        plt.figure(figsize=(14, 5))
        plt.plot(out_df["valid_time"], out_df["ssrd_ground_Jm2"], label="Ground truth (15min)", color="black", linewidth=1)
        plt.plot(out_df["valid_time"], out_df["ssrd_baseline_Jm2"], label="AIFS baseline (15min)", linestyle="--", linewidth=1)
        plt.plot(out_df["valid_time"], out_df["ssrd_model_Jm2"], label="Model prediction (15min)", linewidth=1)
        plt.xlabel("Valid time")
        plt.ylabel("SSRD (J/m² per 15min)")
        plt.title("Validation (20%) — Quarter-hour SSRD (15,30,45,60 min after t_init)")
        plt.legend()
        plt.grid(True)
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=200)
        print(f"Saved validation plot to {args.out_plot}")

    # ---------------- Stage 3: held-out (10%) evaluation - never touched until now ----------------
    if len(eval_paths) == 0:
        print("\nNo held-out eval paths present - skipping held-out evaluation stage.")
        raise SystemExit(0)

    print(f"\n=== Held-out evaluation (10%, {len(eval_paths)} samples) ===")
    if not os.path.isfile(args.save_path):
        raise FileNotFoundError(f"Checkpoint {args.save_path} not found. Make sure training saved the file.")

    eval_model = StFTFusionModelSingle().to(device)
    eval_model.load_state_dict(torch.load(args.save_path, map_location=device))

    os.makedirs(args.eval_outdir, exist_ok=True)
    eval_preds_csv = os.path.join(args.eval_outdir, "holdout_eval_predictions.csv")
    generate_eval_csv_for_paths(eval_model, eval_paths, args.batch_size, device, eval_preds_csv)

    summary_df = process_all(
        cloud_glob=args.cloud,
        models_glob=eval_preds_csv,
        outdir=args.eval_outdir,
        plot_start=None,
        plot_end=None,
    )
    print("\nHeld-out evaluation summary:")
    print(summary_df.to_string(index=False))
