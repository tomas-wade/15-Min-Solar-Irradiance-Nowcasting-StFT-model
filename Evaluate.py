"""
python Evaluate.py \
  --model "stft8_norm_cmw=5.pt" \
  --samples "samples_SON_norm" \
  --cloud "dataSON/SON_cloudmask_sample_prep.csv" \
  --outdir "SON_eval_stft8_cmw=5" \
  --start "2025-08-1" \
  --end "2025-08-14"
"""

import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from Function_Batch_Maker import StFTDataset, stft_collate_fn
from StFT_Model_General import StFTFusionModelGeneral #change to import single or general depending on what is being evaluated


# ---------------- CONFIG ----------------
CLOUD_TIME_COL = "time" #expected column names in the raw MSG cloudmask csv
CLOUD_VALUE_COL = "value"

MODEL_TIME_CANDIDATES = ["valid_time", "valid_times", "time"] #possible timestamp column names in a model-output csv
BASELINE_COL = "ssrd_baseline_Jm2"
MODEL_COL = "ssrd_model_Jm2"
GROUND_COL = "ssrd_ground_Jm2"

MERGE_TOLERANCE = pd.Timedelta("1min") #max time gap allowed when nearest-matching a model row to a cloud-fraction row

DAY_START = 7 #daytime window (hours) used for the day/night RMSE split
DAY_END = 19

CLOUD_BINS = [ #four cloud-fraction buckets each RMSE is broken down by, alongside overall/day/night
    ("lt0.25", lambda x: x < 0.25),
    ("0.25_0.5", lambda x: (x >= 0.25) & (x < 0.5)),
    ("0.5_0.75", lambda x: (x >= 0.5) & (x < 0.75)),
    ("gte0.75", lambda x: x >= 0.75),
]


# ---------------- HELPERS ----------------
#moves every tensor value in a batch dict onto the given device, in place
def move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)


#RMSE between two arrays, ignoring any pair where either value is NaN/inf
def compute_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return float("nan")
    return math.sqrt(((a[mask] - b[mask]) ** 2).mean())


def infer_valid_times_from_meta(meta, n_quarters=4):
    """
    Returns a list of valid times (length n_quarters) for a sample.
    Uses meta['t_init'] + 15,30,45,60 minutes.
    """
    if not isinstance(meta, dict) or "t_init" not in meta:
        return [pd.NaT] * n_quarters

    try:
        t_init = pd.to_datetime(meta["t_init"])
    except Exception:
        return [pd.NaT] * n_quarters

    return [(t_init + pd.Timedelta(minutes=15 * (q + 1))).to_pydatetime() for q in range(n_quarters)]


def is_target_month_sample(meta, target_month=8):
    """
    Returns True if meta['t_init'] falls in the requested month.
    """
    if not isinstance(meta, dict) or "t_init" not in meta:
        return False
    try:
        t_init = pd.to_datetime(meta["t_init"])
    except Exception:
        return False
    return t_init.month == target_month


#finds which column in a model-output csv holds the timestamp, trying known names first then any "time"/"date"-like column
def find_model_time_col(df):
    for cand in MODEL_TIME_CANDIDATES:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def compute_cloud_fraction_from_cloud_csvs(cloud_glob, fallback_times=None):
    """
    Compute cloud fraction per timestamp from cloudmask CSVs (value==2 -> cloudy).
    If cloud_glob is None or empty, falls back to a zero-filled table using fallback_times.
    """
    if not cloud_glob: #no cloud data supplied - builds a stand-in table so downstream merges still work, just with cloud_frac=0
        if fallback_times is None:
            raise ValueError("cloud_glob is empty and no fallback_times were provided.")
        times = pd.to_datetime(pd.Series(fallback_times)).dropna().drop_duplicates().sort_values()
        return pd.DataFrame({
            "time": times,
            "cloud_frac": 0.0,
            "count_total": 0,
            "count_cloudy": 0,
        }).reset_index(drop=True)

    #resolves the (possibly comma-separated, glob-containing) cloud_glob argument into a concrete, deduplicated file list
    paths = []
    for part in cloud_glob.split(","):
        part = part.strip()
        if not part:
            continue
        if "*" in part or "?" in part:
            paths.extend(sorted(glob.glob(part)))
        else:
            if os.path.exists(part):
                paths.append(part)

    paths = sorted(list(dict.fromkeys(paths))) #dict.fromkeys preserves order while dropping duplicate paths
    if not paths:
        raise FileNotFoundError(f"No cloud files found for pattern: {cloud_glob}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if CLOUD_TIME_COL not in df.columns:
            raise KeyError(f"Cloud file {p} missing expected time column '{CLOUD_TIME_COL}'")
        if CLOUD_VALUE_COL not in df.columns:
            raise KeyError(f"Cloud file {p} missing expected value column '{CLOUD_VALUE_COL}'")
        df[CLOUD_TIME_COL] = pd.to_datetime(df[CLOUD_TIME_COL], errors="coerce")
        df = df[~df[CLOUD_TIME_COL].isna()].copy()
        frames.append(
            df[[CLOUD_TIME_COL, CLOUD_VALUE_COL]].rename(
                columns={CLOUD_TIME_COL: "time", CLOUD_VALUE_COL: "value"}
            )
        )

    all_df = pd.concat(frames, ignore_index=True)
    #for every timestamp: total pixel count and how many were classified cloudy (code 2), giving a cloud fraction
    grouped = all_df.groupby("time")["value"].agg(
        count_total="count",
        count_cloudy=lambda s: (s == 2).sum()
    ).reset_index()
    grouped["cloud_frac"] = grouped["count_cloudy"] / grouped["count_total"]
    grouped = grouped.sort_values("time").reset_index(drop=True)
    return grouped[["time", "cloud_frac", "count_total", "count_cloudy"]]


def compute_metrics_on_merged(merged):
    """Return a dict of the single-column RMSE metrics computed on merged dataframe."""
    metrics = {}
    metrics["n_rows"] = int(len(merged))
    metrics["rmse_all"] = compute_rmse(merged[MODEL_COL], merged[GROUND_COL])

    hours = merged["time"].dt.hour
    day_mask = (hours >= DAY_START) & (hours < DAY_END)
    night_mask = ~day_mask

    metrics["rmse_day"] = compute_rmse(merged.loc[day_mask, MODEL_COL], merged.loc[day_mask, GROUND_COL])
    metrics["rmse_night"] = compute_rmse(merged.loc[night_mask, MODEL_COL], merged.loc[night_mask, GROUND_COL])

    for label, predicate in CLOUD_BINS: #one RMSE per cloud-fraction bucket
        colname = f"rmse_cloud_{label}"
        metrics[colname] = compute_rmse(
            merged.loc[predicate(merged["cloud_frac"]), MODEL_COL],
            merged.loc[predicate(merged["cloud_frac"]), GROUND_COL]
        )
    return metrics


def compute_baseline_metrics_from_merged(merged):
    """Compute baseline metrics (same column names) from merged df (baseline vs ground)."""
    #identical structure to compute_metrics_on_merged, just scored against BASELINE_COL instead of MODEL_COL
    metrics = {}
    metrics["n_rows"] = int(len(merged))
    metrics["rmse_all"] = compute_rmse(merged[BASELINE_COL], merged[GROUND_COL])

    hours = merged["time"].dt.hour
    day_mask = (hours >= DAY_START) & (hours < DAY_END)
    night_mask = ~day_mask

    metrics["rmse_day"] = compute_rmse(merged.loc[day_mask, BASELINE_COL], merged.loc[day_mask, GROUND_COL])
    metrics["rmse_night"] = compute_rmse(merged.loc[night_mask, BASELINE_COL], merged.loc[night_mask, GROUND_COL])

    for label, predicate in CLOUD_BINS:
        colname = f"rmse_cloud_{label}"
        metrics[colname] = compute_rmse(
            merged.loc[predicate(merged["cloud_frac"]), BASELINE_COL],
            merged.loc[predicate(merged["cloud_frac"]), GROUND_COL]
        )
    return metrics


def generate_evaluation_csv(
    model_path,
    data_dir,
    outdir,
    batch_size=16,
    eval_csv_name="generated_evaluation.csv",
    target_month=8,
):
    """
    Run the StFT model on samples and write a CSV with:
      valid_time, ssrd_baseline_Jm2, ssrd_model_Jm2, ssrd_ground_Jm2

    Only samples whose meta['t_init'] is in target_month are kept.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = StFTFusionModelGeneral().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() #disables dropout/batch-norm training behaviour for inference

    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Found {len(paths)} samples")

    if not paths:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    dataset = StFTDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, #keeps evaluation order deterministic
        collate_fn=stft_collate_fn,
    )

    all_times = []
    all_baseline = []
    all_model = []
    all_ground = []

    kept_samples = 0
    kept_quarters = 0

    with torch.no_grad(): #no gradient tracking needed for pure inference
        for batch in loader:
            move_to_device(batch, device)

            preds = model(batch)  # (B,4) raw residuals
            aifs = batch["aifs_baseline_quarter"]  # (B,4)
            ground = batch["ground_quarter"]       # (B,4)

            ssrd_pred = (aifs + preds).cpu().numpy() #model's actual prediction = AIFS baseline + predicted residual
            ssrd_base = aifs.cpu().numpy()
            ssrd_true = ground.cpu().numpy()

            meta_list = batch.get("meta", [None] * ssrd_pred.shape[0])

            for i in range(ssrd_pred.shape[0]):
                m = meta_list[i] if isinstance(meta_list, list) else None

                if not is_target_month_sample(m, target_month=target_month): #drops any sample outside the requested month
                    continue

                kept_samples += 1
                valid_times = infer_valid_times_from_meta(m, n_quarters=4)

                for q in range(4): #unpacks each sample's 4 quarter-hour predictions into individual rows
                    all_times.append(valid_times[q])
                    all_baseline.append(float(ssrd_base[i, q]))
                    all_model.append(float(ssrd_pred[i, q]))
                    all_ground.append(float(ssrd_true[i, q]))
                    kept_quarters += 1

    out_df = pd.DataFrame({
        "valid_time": pd.to_datetime(all_times),
        "ssrd_baseline_Jm2": all_baseline,
        "ssrd_model_Jm2": all_model,
        "ssrd_ground_Jm2": all_ground,
    }).sort_values("valid_time").reset_index(drop=True)

    eval_csv_path = os.path.join(outdir, eval_csv_name)
    out_df.to_csv(eval_csv_path, index=False)
    print(f"Saved evaluation CSV to: {eval_csv_path}")
    print(f"Kept {kept_samples} samples and {kept_quarters} quarter-hours from month={target_month}")

    return eval_csv_path, out_df


def process_all(cloud_glob, models_glob, outdir, plot_start=None, plot_end=None):
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) expand model paths
    #resolves the (possibly comma-separated, glob-containing) models_glob argument, letting several model output csvs be scored in one run
    model_paths = []
    for part in models_glob.split(","):
        part = part.strip()
        if not part:
            continue
        if "*" in part or "?" in part:
            model_paths.extend(sorted(glob.glob(part)))
        else:
            if os.path.exists(part):
                model_paths.append(part)

    model_paths = sorted(list(dict.fromkeys(model_paths)))
    if not model_paths:
        raise FileNotFoundError("No model CSVs found for provided patterns/paths")

    # 2) load all model dfs first so we can build a fallback cloud table if needed
    loaded = []
    all_times = []

    for p in model_paths:
        print(f"Loading: {p}")
        df = pd.read_csv(p)
        time_col = find_model_time_col(df)
        if time_col is None:
            raise KeyError(f"Cannot detect time column in {p}")

        for req in [BASELINE_COL, MODEL_COL, GROUND_COL]:
            if req not in df.columns:
                raise KeyError(f"Model file {p} missing required column '{req}'")

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df[~df[time_col].isna()].copy().sort_values(time_col).reset_index(drop=True)
        df = df.rename(columns={time_col: "time"})

        loaded.append((p, df))
        all_times.extend(df["time"].tolist())

    # 3) compute cloud fraction table
    cloud_frac_df = compute_cloud_fraction_from_cloud_csvs(cloud_glob, fallback_times=all_times)
    cloud_frac_csv = os.path.join(outdir, "cloud_fraction_by_time.csv")
    cloud_frac_df.to_csv(cloud_frac_csv, index=False)
    print(f"Saved cloud fraction table: {cloud_frac_csv} ({len(cloud_frac_df)} rows)")

    summary_rows = []
    baseline_metrics = None
    baseline_source = None

    # 4) process each CSV exactly like before
    for idx, (p, df) in enumerate(loaded):
        base = os.path.basename(p)
        name = os.path.splitext(base)[0]

        #merges in cloud fraction by exact timestamp match first, then fills any remaining gaps via nearest-time matching within tolerance
        merged = pd.merge(df, cloud_frac_df[["time", "cloud_frac"]], on="time", how="left")
        if merged["cloud_frac"].isna().any():
            left = df.sort_values("time").reset_index(drop=True)
            right = cloud_frac_df.sort_values("time").reset_index(drop=True)
            merged_asof = pd.merge_asof(
                left, right[["time", "cloud_frac"]],
                on="time",
                direction="nearest",
                tolerance=MERGE_TOLERANCE
            )
            merged["cloud_frac"] = merged["cloud_frac"].fillna(merged_asof["cloud_frac"])
        merged["cloud_frac"] = merged["cloud_frac"].fillna(0.0) #any timestamp still without cloud data defaults to 0

        before = len(merged)
        merged = merged.drop_duplicates().reset_index(drop=True)
        merged = merged.loc[~merged["time"].duplicated(keep="first")].reset_index(drop=True) #keeps only the first row per timestamp
        after = len(merged)

        cleaned_path = os.path.join(outdir, f"{name}_cleaned_with_cloudfrac.csv")
        merged.to_csv(cleaned_path, index=False)

        # plotting
        plot_df = merged.copy()
        if plot_start is not None:
            plot_df = plot_df[plot_df["time"] >= plot_start]
        if plot_end is not None:
            plot_df = plot_df[plot_df["time"] <= plot_end]

        if not plot_df.empty:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.plot(plot_df["time"], plot_df[GROUND_COL], label="Ground", linewidth=1, color="black")
            ax1.plot(plot_df["time"], plot_df[BASELINE_COL], label="AIFS baseline", linestyle="--", linewidth=1, color="tab:blue")
            ax1.plot(plot_df["time"], plot_df[MODEL_COL], label="Model", linewidth=1, color="y")
            ax1.set_xlabel("time")
            ax1.set_ylabel("SSRD (J/m² per 15min)")
            ax1.grid(True)

            ax2 = ax1.twinx() #cloud fraction plotted on a second y-axis sharing the same time axis
            ax2.plot(plot_df["time"], plot_df["cloud_frac"], label="Cloud Fraction", linewidth=1, color="c", alpha=0.7)
            ax2.set_ylabel("Cloud Fraction (0–1)")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right") #merges both axes' legends into one

            plt.title("Time series Comparison of Global Model, AIFS, Groundtruth and Cloud Fraction")
            plt.tight_layout()

            plot_path = os.path.join(plots_dir, f"{name}_timeseries.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
        else:
            plot_path = ""
            print(f"Plot window is empty for {base}; skipping plot.")

        model_metrics = compute_metrics_on_merged(merged)

        if idx == 0: #baseline metrics only computed once, from the first model file processed
            baseline_metrics = compute_baseline_metrics_from_merged(merged)
            baseline_source = p

        model_row = {
            "row_type": "model",
            "file": base,
            "cleaned_csv": cleaned_path,
            "plot_png": plot_path,
            "n_rows_before": int(before),
            "n_rows_after": int(after),
            "rmse_all": model_metrics.get("rmse_all"),
            "rmse_day": model_metrics.get("rmse_day"),
            "rmse_night": model_metrics.get("rmse_night"),
        }
        for label, _ in CLOUD_BINS:
            model_row[f"rmse_cloud_{label}"] = model_metrics.get(f"rmse_cloud_{label}")

        summary_rows.append(model_row)
        print(f"Saved outputs for {base}. Model rmse_all={model_metrics.get('rmse_all'):.2f}")

    if baseline_metrics is None:
        raise RuntimeError("No model files processed to derive baseline from the first model file.")

    baseline_row = {
        "row_type": "baseline",
        "file": os.path.basename(baseline_source),
        "cleaned_csv": "",
        "plot_png": "",
        "n_rows_before": int(baseline_metrics.get("n_rows", 0)),
        "n_rows_after": int(baseline_metrics.get("n_rows", 0)),
        "rmse_all": baseline_metrics.get("rmse_all"),
        "rmse_day": baseline_metrics.get("rmse_day"),
        "rmse_night": baseline_metrics.get("rmse_night"),
    }
    for label, _ in CLOUD_BINS:
        baseline_row[f"rmse_cloud_{label}"] = baseline_metrics.get(f"rmse_cloud_{label}")

    #builds the final summary table: baseline row first (delta 0 by definition), then every model row with its improvement over baseline
    final_rows = []
    baseline_row_out = baseline_row.copy()
    baseline_row_out["delta_vs_baseline"] = 0.0
    final_rows.append(baseline_row_out)

    baseline_value = baseline_row.get("rmse_all")
    for mrow in summary_rows:
        row = mrow.copy()
        if baseline_value is None or np.isnan(baseline_value) or row.get("rmse_all") is None:
            row["delta_vs_baseline"] = float("nan")
        else:
            row["delta_vs_baseline"] = float(baseline_value - row["rmse_all"]) #positive means the model beat the baseline
        final_rows.append(row)

    summary_df = pd.DataFrame(final_rows)
    summary_csv = os.path.join(outdir, "summary_all_models_and_baseline_single_rmse.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved single summary CSV: {summary_csv}")

    models_only = summary_df[summary_df["row_type"] == "model"].sort_values("rmse_all").reset_index(drop=True) #best (lowest RMSE) first
    ranking_csv = os.path.join(outdir, "summary_models_ranked_by_rmse.csv")
    models_only.to_csv(ranking_csv, index=False)
    print(f"Saved ranking CSV: {ranking_csv}")

    return summary_df


# ---------------- END-TO-END ENTRY POINT ----------------
#chains stage 1 (run the model, write per-quarter predictions) and stage 2 (score + plot) together
def evaluate_and_process(model_path, data_dir, cloud_glob, outdir, batch_size=16, plot_start=None, plot_end=None, target_month=8):
    os.makedirs(outdir, exist_ok=True)

    # Step 1: generate evaluation CSV from model + samples
    eval_csv_path, _ = generate_evaluation_csv(
        model_path=model_path,
        data_dir=data_dir,
        outdir=outdir,
        batch_size=batch_size,
        eval_csv_name="generated_evaluation.csv",
        target_month=target_month,
    )

    # Step 2: run the same processing pipeline on that CSV
    summary_df = process_all(
        cloud_glob=cloud_glob,
        models_glob=eval_csv_path,
        outdir=outdir,
        plot_start=plot_start,
        plot_end=plot_end,
    )

    print("\nTop summary rows:")
    print(summary_df.head(10).to_string(index=False))


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stft8_norm_cmw=2.pt", help="Path to the trained model checkpoint.")
    parser.add_argument("--samples", default="samples_SON_norm", help="Directory containing .npz samples.")
    parser.add_argument("--cloud", default=None, help="Optional glob or comma-separated cloudmask CSVs.")
    parser.add_argument("--outdir", default="SON_val", help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--start", type=str, default=None, help="Plot window start (e.g. 2025-08-01).")
    parser.add_argument("--end", type=str, default=None, help="Plot window end (e.g. 2025-08-31).")
    parser.add_argument("--month", type=int, default=8, help="Only evaluate samples whose t_init month equals this value.")
    args = parser.parse_args()

    plot_start = pd.to_datetime(args.start) if args.start else None
    plot_end = pd.to_datetime(args.end) if args.end else None

    evaluate_and_process(
        model_path=args.model,
        data_dir=args.samples,
        cloud_glob=args.cloud,
        outdir=args.outdir,
        batch_size=args.batch_size,
        plot_start=plot_start,
        plot_end=plot_end,
        target_month=args.month,
    )
