#!/usr/bin/env python3
"""
Consolidated per-site pipeline: AIFS + ERA5 + MSG cloud mask + ground truth
download, followed by Sample_Prep .npz generation - all driven by CLI args
instead of hand-edited constants in five separate scripts.

Usage:
    /usr/local/bin/python3.13 Sample_Download.py --site BUD --lat 47.4291 --lon 19.1822 --start 2026-01-01 --end 2026-03-31 --data-dir dataBUD --samples-dir samples --raw-tab-dir raw_ground --skip-existing

Before running the "ground" step, place the raw monthly PANGAEA/BSRN
{SITE}_radiation_YYYY-MM.tab files in --raw-tab-dir (defaults to --data-dir).
There is no API for that data source - it's a manual download.

Requires EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET env vars for the
"msg" step, and a configured ~/.cdsapirc for the "era5" step.
"""
import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ecmwf.opendata import Client
import xarray as xr
from scipy.interpolate import PchipInterpolator


# --------------------------------------------------------------------------
# Output filename templates - edit these in ONE place to rename output files.
# Every step (writers and readers) builds its paths from these via the
# SiteConfig helper methods below, so producer and consumer always agree.
# {site} is replaced with the site ID (e.g. BUD); {tinit} with a sample's t_init timestamp string.
# --------------------------------------------------------------------------
AIFS_CSV_NAME = "{site}_aifs_sample_prep_15min_2026Q1.csv"
ERA5_SSRD_CSV_NAME = "{site}_ERA5_sample_prep_ssrd_2026Q1.csv"
ERA5_D2M_CSV_NAME = "{site}_ERA5_sample_prep_d2m_2026Q1.csv"
ERA5_T2M_CSV_NAME = "{site}_ERA5_sample_prep_t2m_2026Q1.csv"
MSG_CSV_NAME = "{site}_cloudmask_sample_prep_2026Q1.csv"
GROUND_CSV_NAME = "{site}_ground_truth_sample_prep_2026Q1.csv"
SAMPLE_NPZ_NAME = "sample_tinit_{tinit}_{site}.npz"


@dataclass #creates data class for all the site info
class SiteConfig:
    site_id: str
    lat: float
    lon: float
    start_date: str
    end_date: str
    data_dir: Path #holds intermediate csv for this site
    samples_dir: Path #where step_sample_prep writes final .npz
    raw_tab_dir: Path #where raw ground truths are held
    msg_start: Optional[str] = None
    msg_end: Optional[str] = None
    msg_half_deg: float = 0.30
    aifs_source: str = "aws"

    def __post_init__(self): #sets msg start to start and end date if not changed explicitly
        if self.msg_start is None:
            self.msg_start = self.start_date
        if self.msg_end is None:
            self.msg_end = self.end_date

    #output filename helpers - all derived from the templates above so every step agrees on paths
    def aifs_csv(self) -> Path:
        return self.data_dir / AIFS_CSV_NAME.format(site=self.site_id)

    def era5_ssrd_csv(self) -> Path:
        return self.data_dir / ERA5_SSRD_CSV_NAME.format(site=self.site_id)

    def era5_d2m_csv(self) -> Path:
        return self.data_dir / ERA5_D2M_CSV_NAME.format(site=self.site_id)

    def era5_t2m_csv(self) -> Path:
        return self.data_dir / ERA5_T2M_CSV_NAME.format(site=self.site_id)

    def msg_csv(self) -> Path:
        return self.data_dir / MSG_CSV_NAME.format(site=self.site_id)

    def ground_csv(self) -> Path:
        return self.data_dir / GROUND_CSV_NAME.format(site=self.site_id)

    def sample_npz(self, tinit_str: str) -> Path:
        return self.samples_dir / SAMPLE_NPZ_NAME.format(tinit=tinit_str, site=self.site_id)



#AIFS download

def step_aifs(cfg: SiteConfig, skip_existing: bool = False) -> Path:
    out_path = cfg.aifs_csv() #if output csv already exists and skip existing is passed then skips
    if skip_existing and out_path.exists():
        print(f"[aifs] {out_path} exists, skipping")
        return out_path

    LEADS = [6] #only need 6 hour lead to get AIFS best forecast at each time
    client = Client(cfg.aifs_source, beta=False)
    steps = [str(L) for L in range(6, max(LEADS) + 1, 6)]

    #sets start and end dates
    start_dt = pd.Timestamp(cfg.start_date)
    end_dt = pd.Timestamp(cfg.end_date)

    #builds valid time list by keeping only the 6 hour intervals aifs is generated for
    vts = []
    t = start_dt
    while t <= end_dt:
        if t.hour in (0, 6, 12, 18):
            vts.append(t.to_pydatetime())
        t += timedelta(hours=6)

    def load_aifs(init):
        #builds unique grib filename and calls ECMWF opendata client to download ssrd
        fn = f"aifs_{cfg.site_id}_{init.strftime('%Y%m%dT%H')}.grib"
        client.retrieve(
            date=init.strftime("%Y%m%d"),
            time=init.hour,
            stream="oper",
            type="fc",
            levtype="sfc",
            model="aifs-single",
            param=["ssrd"],
            step=steps,
            target=fn,
        )

        #opens the dataset and sorts by step
        ds = xr.open_dataset(fn, engine="cfgrib")
        da = ds["ssrd"]
        if "step" not in da.dims:  # a single-lead request collapses step into a scalar coord instead of a dimension
            da = da.expand_dims("step")
        da = da.sortby("step")  # cumulative radiation J/m^2 sorted by lead

        #converts step into plain integer hours form timedelta64
        int_steps = [int(s / np.timedelta64(1, "h")) for s in da.step.values]
        da = da.assign_coords(step=("step", int_steps))

        #converts cumulative-since-tinit values if using more than just first lead to just 6h windows
        first = da.isel(step=0)
        if len(LEADS) > 1:
            diff = []
            for L in LEADS[1:]:
                diff.append((da.sel(step=L) - da.sel(step=(L - 6))).expand_dims({"step": [L]}))
            inter = xr.concat(
                [first.expand_dims({"step": [da.step.values[0]]}), xr.concat(diff, dim="step")],
                dim="step",
            )
        else:
            inter = first.expand_dims({"step": [da.step.values[0]]})

        ds.close() #closes dataset handle

        #deletes the downloaded .grib file
        try:
            os.remove(fn)
            for idxfn in Path(".").glob(f"{fn}*.idx"):
                idxfn.unlink()
        except Exception as e:
            print("[aifs] Warning: could not delete", fn, ":", e)

        return inter

    #main extraction loop
    rows = []
    cache = {}
    for i, vt in enumerate(vts, start=1):
        row = {"valid_time": vt}

        for L in LEADS:
            #for a calculated init fetches/downloads relevant file as defined by loadaifs
            init = vt - timedelta(hours=L)

            if init not in cache:
                cache[init] = load_aifs(init) #if aifs for this time has not been loaded it saves it to cache

            ds = cache[init]
            for s in ds.step.values:
                hr = int(s)
                if init + timedelta(hours=hr) == vt:
                    da_step = ds.sel(step=s)

                    #finds nearest grid cell to coords of interest in both 1D lat/lon coordinate case and 2D case
                    lat_coords = da_step.latitude.values
                    lon_coords = da_step.longitude.values

                    if lat_coords.ndim == 1 and lon_coords.ndim == 1:
                        iy = int(np.argmin(np.abs(lat_coords - cfg.lat)))
                        ix = int(np.argmin(np.abs(lon_coords - cfg.lon)))
                    else:
                        lat_flat = lat_coords.ravel()
                        lon_flat = lon_coords.ravel()
                        coslat = np.cos(np.deg2rad(cfg.lat))
                        d2 = (lat_flat - cfg.lat) ** 2 + ((lon_flat - cfg.lon) * coslat) ** 2
                        idx_flat = int(np.argmin(d2))
                        ny, nx = lat_coords.shape
                        iy = idx_flat // nx
                        ix = idx_flat % nx

                    P = 1 #extent of patch of AIFS data if want to use more than single point
                    pad = P // 2
                    patch = np.full((P, P), np.nan, dtype=float)

                    #fills patch with values from aifs
                    arr2d = np.asarray(da_step.values)
                    if arr2d.ndim == 3:
                        arr2d = np.squeeze(arr2d)
                    if arr2d.ndim == 2:
                        ny, nx = arr2d.shape
                        for dy in range(-pad, pad + 1):
                            for dx in range(-pad, pad + 1):
                                y = iy + dy
                                x = ix + dx
                                if 0 <= y < ny and 0 <= x < nx:
                                    patch[dy + pad, dx + pad] = float(arr2d[y, x])
                    #writes patch values into row under column names e.g AIFS_6h_r0_c0
                    for r in range(-pad, pad + 1):
                        for c in range(-pad, pad + 1):
                            rr = r + pad
                            cc = c + pad
                            colname = f"AIFS_{L}h_r{r}_c{c}"
                            row[colname] = float(patch[rr, cc]) if not np.isnan(patch[rr, cc]) else np.nan

                    break #breaks out of step search loop once matching step is found

        rows.append(row) #adds completed rows
        print(f"[aifs] row {i}/{len(vts)}")

    df6 = pd.DataFrame(rows).set_index("valid_time").sort_index() #builds data frame indexed by valid time

    # 6h increments -> 15-min increments: cumsum, PCHIP-interpolate the
    # cumulative series onto a 15-min grid, then diff back to increments.
    cum6 = df6.cumsum(axis=0) #calculates cumulative radiation since start of series which is needed for interpolation

    start = cum6.index[0]
    end = cum6.index[-1]
    index_15min = pd.date_range(start=start, end=end, freq="15T") #builds 15 min index spanning same range

    cum_15min = pd.DataFrame(index=index_15min, columns=cum6.columns, dtype=float)
    x_target = index_15min.view("int64").astype(float) / 1e9

    #for each column PCHIP-interpolates the cumulative series onto 15-min grid
    for col in cum6.columns:
        series6 = cum6[col]
        valid = series6.dropna()
        if len(valid) < 2:
            cum_15min[col] = series6.reindex(index_15min).interpolate(method="time")
            continue

        x_valid = valid.index.view("int64").astype(float) / 1e9
        y_valid = valid.values.astype(float)

        pchip = PchipInterpolator(x_valid, y_valid, extrapolate=False)
        cum_15min[col] = pchip(x_target)

    inc_15min = cum_15min.diff().copy() #differencing the interpolated cumulative series gives back 15-min increments, first row is always NaN since nothing precedes it

    intervals_in_6h = 6 * 4 #24 quarter-hours in 6 hours
    for col in inc_15min.columns:
        if pd.isna(inc_15min[col].iloc[0]): #handles that always-NaN first row per column
            try:
                first6_val = cum6[col].iloc[0] #the first known 6h cumulative value
                if not pd.isna(first6_val):
                    #evenly spreads that first 6h total across the first 24 fifteen-minute slots
                    inc_val = first6_val / intervals_in_6h
                    for i in range(min(intervals_in_6h, len(inc_15min))):
                        inc_15min.iloc[i, inc_15min.columns.get_loc(col)] = inc_val
                else:
                    #fallback if no usable first value: fill from nearby known values
                    inc_15min[col] = inc_15min[col].bfill().ffill()
            except Exception:
                #last resort: just zero it out
                inc_15min.at[inc_15min.index[0], col] = 0.0

    inc_15min.index.name = "valid_time"
    cfg.data_dir.mkdir(parents=True, exist_ok=True) #creates data dir if it doesn't already exist
    inc_15min.to_csv(out_path) #saves 15-min incremental AIFS csv
    print(f"[aifs] saved -> {out_path}")
    return out_path


# --------------------------------------------------------------------------
# ERA5
# --------------------------------------------------------------------------
#downloads hourly ERA5 reanalysis (ssrd, 2m dewpoint temp, 2m temp) on a real 5x5 grid around the site
def step_era5(cfg: SiteConfig, skip_existing: bool = False):
    #the three output csv paths, one per variable
    out_ssrd = cfg.era5_ssrd_csv()
    out_d2m = cfg.era5_d2m_csv()
    out_t2m = cfg.era5_t2m_csv()
    if skip_existing and out_ssrd.exists() and out_d2m.exists() and out_t2m.exists(): #skips only if all three already exist
        print("[era5] outputs exist, skipping")
        return out_ssrd, out_d2m, out_t2m

    import cdsapi #lazy import, only needed for this step

    P = 5 #grid width/height in points - a genuine 5x5 grid, unlike AIFS's single point
    HALF = P // 2 #how far the grid extends each side of center (2 points)
    DLAT = 0.25 #ERA5's native grid spacing in degrees
    DLON = 0.25

    client = cdsapi.Client()

    #builds the list of (row offset, col offset, lat, lon) for every point in the 5x5 grid
    points = []
    for r in range(-HALF, HALF + 1):
        for c in range(-HALF, HALF + 1):
            points.append((r, c, cfg.lat + r * DLAT, cfg.lon + c * DLON))

    ssrd_map, d2m_map, t2m_map = {}, {}, {} #dicts holding each grid point's timeseries per variable, keyed by "r{r}_c{c}"

    def find_column(df, candidates):
        #tries to find a variable's column by exact (case-insensitive) name match first
        lowcols = {col.lower(): col for col in df.columns}
        for cand in candidates:
            if cand.lower() in lowcols:
                return lowcols[cand.lower()]
        #falls back to substring match if CDS used a slightly different column name
        for cand in candidates:
            for lc, orig in lowcols.items():
                if cand.lower() in lc:
                    return orig
        return None

    #one API request per grid point, requesting all 3 variables together for the full date range
    for (r, c, lat, lon) in points:
        print(f"[era5] requesting r={r}, c={c} -> lat={lat:.5f}, lon={lon:.5f}")

        request = {
            "variable": [
                "surface_solar_radiation_downwards",
                "2m_dewpoint_temperature",
                "2m_temperature",
            ],
            "location": {"longitude": float(lon), "latitude": float(lat)},
            "date": [f"{cfg.start_date}/{cfg.end_date}"],
            "data_format": "csv",
        }

        filename = client.retrieve("reanalysis-era5-single-levels-timeseries", request).download() #downloads the returned csv
        df = pd.read_csv(filename)

        #figures out which column holds the timestamp - CDS's naming isn't always consistent
        if "valid_time" in df.columns:
            time_col = "valid_time"
        elif "time" in df.columns:
            time_col = "time"
        else:
            possible = [col for col in df.columns if "time" in col.lower()]
            if possible:
                time_col = possible[0]
            else:
                raise RuntimeError(f"Cannot find time column in CSV: {filename}")

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

        #locates the actual data column for each of the 3 variables
        ssrd_col = find_column(df, ["surface_solar_radiation_downwards", "ssrd"])
        d2m_col = find_column(df, ["2m_dewpoint_temperature", "d2m", "dewpoint2m", "dewpoint_temperature"])
        t2m_col = find_column(df, ["2m_temperature", "t2m", "temperature2m"])

        #stores each variable's series for this point, or warns and skips if the column wasn't found
        if ssrd_col is None:
            print(f"[era5] Warning: no SSRD column in {filename}; skipping point.")
        else:
            ssrd_map[f"r{r}_c{c}"] = df[ssrd_col].astype(float).sort_index()

        if d2m_col is None:
            print(f"[era5] Warning: no d2m column in {filename}; skipping point.")
        else:
            d2m_map[f"r{r}_c{c}"] = df[d2m_col].astype(float).sort_index()

        if t2m_col is None:
            print(f"[era5] Warning: no t2m column in {filename}; skipping point.")
        else:
            t2m_map[f"r{r}_c{c}"] = df[t2m_col].astype(float).sort_index()

        #deletes the temporary file CDS downloaded, now that its data is in memory
        try:
            Path(filename).unlink()
        except Exception:
            pass

    def build_union_index(maps):
        #combines every point's time index into one sorted union index (points may not all share identical timestamps)
        all_times = None
        for s in maps.values():
            all_times = s.index if all_times is None else all_times.union(s.index)
        return all_times.sort_values() if all_times is not None else pd.DatetimeIndex([])

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    def save_map(maps, out_path, label):
        #reindexes every point's series onto the shared union index (introduces NaN wherever a point lacks that timestamp)
        idx = build_union_index(maps)
        if idx.size == 0:
            print(f"[era5] No {label} data collected; skipping.")
            return
        df_out = pd.DataFrame(index=idx)
        for colname, ser in maps.items():
            df_out[colname] = ser.reindex(df_out.index)
        df_out.index.name = "valid_time"
        df_out.to_csv(out_path) #writes the wide 25-column csv (one column per grid point)
        print(f"[era5] saved {label} -> {out_path}")

    #builds and saves all three variable csvs
    save_map(ssrd_map, out_ssrd, "SSRD")
    save_map(d2m_map, out_d2m, "d2m")
    save_map(t2m_map, out_t2m, "t2m")
    return out_ssrd, out_d2m, out_t2m


# --------------------------------------------------------------------------
# MSG cloud mask
# --------------------------------------------------------------------------
#downloads 15-min MSG cloud mask data on a 13x13 pixel box around the site - resumable/crash-safe since this can be a long-running download
def step_msg(cfg: SiteConfig, skip_existing: bool = False) -> Path:
    #skip_existing is intentionally ignored here - msg always resumes from its own last completed file instead of hard-skipping
    out_path = cfg.msg_csv()

    import eumdac #EUMETSAT data access client
    import pygrib #reads the downloaded .grib cloud mask files

    #credentials come from env vars rather than being hardcoded in source
    consumer_key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    consumer_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        raise RuntimeError(
            "EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET env vars must be set to run the msg step."
        )

    LAT_C = cfg.lat
    LON_C = cfg.lon
    HALF_DEG = cfg.msg_half_deg #half-width in degrees of the box around the site
    tmp_csv = str(out_path) + ".tmp" #used as a scratch file while safely rewriting the csv

    def norm_lon_array(lonarr):
        #wraps longitudes into the -180..+180 convention
        return ((lonarr + 180) % 360) - 180

    def get_last_data_filename(csv_path):
        #reads the csv's very last row to find which source .grib file was being written when the process last stopped
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return None
        last = None
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header_seen = False
            for row in reader:
                if not header_seen: #skips header row
                    header_seen = True
                    continue
                if row and len(row) >= 5:
                    last = row[4] #filename column
        return last

    def filter_out_filename(csv_path, bad_fname):
        #rewrites the csv with every row belonging to bad_fname removed - used to strip a possibly half-written last file before resuming
        if bad_fname is None or not os.path.exists(csv_path):
            return
        with open(csv_path, "r", newline="") as fin, open(tmp_csv, "w", newline="") as fout:
            reader = csv.reader(fin)
            writer = csv.writer(fout)
            header = next(reader, None)
            if header:
                writer.writerow(header)
            for row in reader:
                if not row or len(row) < 5:
                    continue
                if row[4] == bad_fname:
                    continue
                writer.writerow(row)
        os.replace(tmp_csv, csv_path) #atomically swaps the cleaned file in

    def load_processed_filenames(csv_path):
        #builds the set of source .grib filenames that are already fully recorded in the csv, so they can be skipped on resume
        processed = set()
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return processed
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None) #skips header
            for row in reader:
                if row and len(row) >= 5:
                    processed.add(row[4])
        return processed

    #on startup: find and strip out the last file's rows in case the previous run crashed mid-write
    last_fname = get_last_data_filename(str(out_path))
    if last_fname:
        print(f"[msg] Found existing CSV. Last filename: {last_fname}; removing for safe restart")
        filter_out_filename(str(out_path), last_fname)

    processed_fnames = load_processed_filenames(str(out_path)) #files that can be safely skipped this run
    print(f"[msg] Files already completed in CSV: {len(processed_fnames)}")

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    csv_exists = out_path.exists() and out_path.stat().st_size > 0
    csv_file = open(out_path, "a", newline="") #opens in append mode so resuming doesn't clobber existing rows
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(["time", "lat", "lon", "value", "filename"]) #writes header only for a brand new file
        csv_file.flush()

    #authenticates and fetches the cloud mask product collection
    token = eumdac.AccessToken((consumer_key, consumer_secret))
    ds = eumdac.DataStore(token)
    collection = ds.get_collection("EO:EUM:DAT:MSG:CLM")

    start = datetime.strptime(cfg.msg_start, "%Y-%m-%d")
    end = datetime.strptime(cfg.msg_end, "%Y-%m-%d")
    products = list(collection.search(dtstart=start, dtend=end)) #all products (roughly one per 15 min) in the date window

    mask = None
    lats_global = None
    lons_global = None
    ii = jj = None

    #builds the spatial pixel mask once, from the very first available file - MSG's grid geometry is fixed across files so this only needs doing once
    for product in products:
        for entry in product.entries:
            fname = entry
            print("[msg] Preparing mask using", fname)

            #downloads just this one file to read its grid
            with product.open(entry) as stream:
                with open(fname, "wb") as fh:
                    fh.write(stream.read())

            grbs = pygrib.open(fname)
            grb = None
            for g in grbs: #finds the first proper 2D spatial field in the grib
                if g.values is not None and g.values.ndim == 2:
                    grb = g
                    break
            if grb is None: #no usable field in this file, try the next one
                grbs.close()
                os.remove(fname)
                continue

            lats_global, lons_global = grb.latlons() #full-disk lat/lon arrays
            lons_global = norm_lon_array(lons_global)
            lons_global = -1 * lons_global  # MSG longitude inversion

            #computes the box boundaries around the site
            lat_min = LAT_C - HALF_DEG
            lat_max = LAT_C + HALF_DEG
            lonc_norm = ((LON_C + 180) % 360) - 180
            lon_min = lonc_norm - HALF_DEG
            lon_max = lonc_norm + HALF_DEG

            #builds the longitude mask, handling the case where the box wraps around the +-180 line
            if lon_min <= lon_max:
                lon_mask = (lons_global >= lon_min) & (lons_global <= lon_max)
            else:
                lon_mask = (lons_global >= lon_min) | (lons_global <= lon_max)

            lat_mask = (lats_global >= lat_min) & (lats_global <= lat_max)
            mask = lat_mask & lon_mask
            ii, jj = np.where(mask) #the fixed pixel indices inside the box - reused for every subsequent file

            grbs.close()
            os.remove(fname)
            print(f"[msg] Mask created! Pixels in box: {ii.size}")
            break #stops once the mask is built
        if mask is not None:
            break

    FLUSH_EVERY = 20 #only flushes to disk every 20 files, for speed
    count_since_flush = 0

    try:
        #main download+extract pass over every product in the window
        for product in collection.search(dtstart=start, dtend=end):
            for entry in product.entries:
                fname = os.path.basename(entry).strip()
                lower = fname.lower()

                #skips manifest/metadata entries that aren't actual data files
                if lower.endswith("manifest.xml") or lower.endswith("eopmetadata.xml"):
                    continue
                if fname in processed_fnames: #already done on a previous run
                    continue

                print("[msg] Downloading", fname)

                with product.open(entry) as stream, open(fname, "wb") as fh:
                    shutil.copyfileobj(stream, fh)

                grbs = pygrib.open(fname)
                grb = None
                try:
                    if grb is None:
                        for g in grbs: #finds the first 2D field again for this file
                            vals = getattr(g, "values", None)
                            if vals is not None and getattr(vals, "ndim", 0) == 2:
                                grb = g
                                break
                    if grb is None: #no usable field, skip this file
                        grbs.close()
                        os.remove(fname)
                        continue

                    values = grb.values

                    #parses the timestamp out of the filename (MSG filenames embed a YYYYMMDDHHMMSS segment)
                    ts = None
                    for part in fname.split("-"):
                        if part.startswith("20") and len(part) >= 14:
                            try:
                                ts = datetime.strptime(part[:14], "%Y%m%d%H%M%S")
                                break
                            except Exception:
                                pass
                    time_str = ts.isoformat() if ts else ""

                    #reads out only the pixels inside the precomputed box
                    pix_lat = lats_global[ii, jj]
                    pix_lon = lons_global[ii, jj]
                    pix_val = values[ii, jj].astype(float)

                    #builds the rows for this file, dropping any NaN pixel values
                    rows = []
                    append = rows.append
                    for lat, lon, val in zip(pix_lat, pix_lon, pix_val):
                        if np.isnan(val):
                            continue
                        append((time_str, float(lat), float(lon), float(val), fname))

                    if rows:
                        writer.writerows(rows) #writes all rows for this file in one call

                    grbs.close()
                    os.remove(fname) #deletes the local grib once its data is safely written
                    processed_fnames.add(fname)
                    print("[msg] Saved & deleted", fname)

                    count_since_flush += 1
                    if count_since_flush >= FLUSH_EVERY:
                        csv_file.flush()
                        count_since_flush = 0

                except Exception:
                    #ensures the grib handle is closed and the partial download is removed even if something failed, then re-raises so the error isn't silently swallowed
                    try:
                        grbs.close()
                    except Exception:
                        pass
                    if os.path.exists(fname):
                        os.remove(fname)
                    raise
    finally:
        csv_file.close() #always closes the csv handle, even on error

    print("[msg] CSV written/appended to", out_path)

    #post-process: value 3 is MSG's "invalid/no data" sentinel code, replace it with NaN
    msg_df = pd.read_csv(out_path, parse_dates=["time"])
    msg_df["value"] = msg_df["value"].astype(float)
    msg_df.loc[msg_df["value"] == 3, "value"] = np.nan
    print("[msg] Number of invalid points replaced with NaN:", msg_df["value"].isna().sum())
    msg_df.to_csv(out_path, index=False)
    print("[msg] done")
    return out_path


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def _detect_pangaea_header_row(path: Path) -> int:
    """PANGAEA .tab files end their metadata block with a line containing
    only '*/'; the column header is the line right after it. Header block
    length varies file to file, so detect it instead of hardcoding it."""
    with open(path, "r", encoding="latin-1") as f:
        for i, line in enumerate(f): #scans line by line looking for the end-of-metadata marker
            if line.strip() == "*/":
                return i + 1 #the header row is the very next line
    raise ValueError(f"Could not find PANGAEA header marker '*/' in {path}")


#formats the raw manually-downloaded monthly station radiation files into one 15-min ground truth csv (J/m^2)
def step_ground_truth(cfg: SiteConfig, skip_existing: bool = False) -> Path:
    out_path = cfg.ground_csv()
    if skip_existing and out_path.exists():
        print(f"[ground] {out_path} exists, skipping")
        return out_path

    #globs for every monthly raw file matching this site, instead of a hardcoded list of filenames
    tab_files = sorted(cfg.raw_tab_dir.glob(f"{cfg.site_id}_radiation_*.tab"))
    if not tab_files: #these files have no API - they must be manually placed here first, so fail with a clear message
        raise FileNotFoundError(
            f"No raw radiation .tab files found matching {cfg.site_id}_radiation_*.tab in "
            f"{cfg.raw_tab_dir}. These are downloaded manually from PANGAEA/BSRN - place them "
            f"there before running the ground step."
        )

    #reads and concatenates every monthly file, auto-detecting each one's header row
    frames = []
    for f in tab_files:
        header_row = _detect_pangaea_header_row(f)
        frames.append(pd.read_csv(f, header=header_row, sep="\t"))

    ds = pd.concat(frames, ignore_index=True)

    ds["Date/Time"] = pd.to_datetime(ds["Date/Time"])
    ds = ds.set_index("Date/Time").sort_index()

    swd = ds["SWD [W/m**2]"].clip(lower=0.0) #shortwave downward radiation, clipping negative sensor noise to zero
    swd_15min_mean = swd.resample("15T").mean() #resamples to 15-minute mean W/m^2

    #drops the first bin since the interval preceding it is likely incomplete/out of range
    if len(swd_15min_mean) > 0:
        swd_15min_mean = swd_15min_mean[swd_15min_mean.index > swd_15min_mean.index[0]]

    swd_15min_j = swd_15min_mean * 900.0 #converts average W/m^2 over 15 min into energy J/m^2 (900 seconds in 15 min)
    swd_15min_j.name = "SWD [J/m**2]"

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    swd_15min_j.to_csv(out_path)
    print(f"[ground] saved 15-min ground truth ({len(tab_files)} monthly files) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Sample prep (.npz)
# --------------------------------------------------------------------------
#combines all downloaded/formatted sources into one .npz training sample per hourly t_init
def step_sample_prep(cfg: SiteConfig, skip_existing: bool = False) -> Path:
    site_id = cfg.site_id
    out_dir = cfg.samples_dir
    if skip_existing and out_dir.exists() and any(out_dir.glob("*.npz")): #skips only if the samples dir already has output
        print(f"[prep] {out_dir} already has .npz samples, skipping")
        return out_dir

    #derives the six input csv paths via the same cfg helpers every producing step writes to
    aifs_csv = cfg.aifs_csv()
    era5_ssrd_csv = cfg.era5_ssrd_csv()
    era5_d2m_csv = cfg.era5_d2m_csv()
    era5_t2m_csv = cfg.era5_t2m_csv()
    msg_csv = cfg.msg_csv()
    gt_csv = cfg.ground_csv()

    AIFS_CENTER_COL = "AIFS_6h_r0_c0" #the single AIFS column actually used as the baseline (matches step_aifs's P=1 center point)

    out_dir.mkdir(parents=True, exist_ok=True)

    #loads all six input tables, each indexed by time
    aifs_15 = pd.read_csv(aifs_csv, parse_dates=["valid_time"]).set_index("valid_time").sort_index()
    era5_ssrd = pd.read_csv(era5_ssrd_csv, parse_dates=["valid_time"]).set_index("valid_time").sort_index()
    era5_d2m = pd.read_csv(era5_d2m_csv, parse_dates=["valid_time"]).set_index("valid_time").sort_index()
    era5_t2m = pd.read_csv(era5_t2m_csv, parse_dates=["valid_time"]).set_index("valid_time").sort_index()

    msg = pd.read_csv(msg_csv, parse_dates=["time"]).set_index("time").sort_index()

    #loads ground truth, handling either a single numeric column or picking the first numeric column as a fallback
    ground = pd.read_csv(gt_csv, parse_dates=["Date/Time"]).set_index("Date/Time").sort_index()
    if ground.shape[1] == 1:
        ground_vals = ground.iloc[:, 0].astype(float)
    else:
        gcol = None
        for c in ground.columns:
            if np.issubdtype(ground[c].dtype, np.number):
                gcol = c
                break
        if gcol is None:
            raise RuntimeError("No numeric column found in ground CSV to use as 15-min J/m^2 values")
        ground_vals = ground[gcol].astype(float)

    def aifs_15min_vector_for_tinit(t_init):
        #looks up the AIFS baseline value at t_init+15/30/45/60 min, returning a length-4 array or None if any of the four is missing/invalid
        vals = []
        for i in range(1, 5):  # 1..4 -> 15,30,45,60
            t_lookup = t_init + timedelta(minutes=15 * i)
            if t_lookup not in aifs_15.index:
                return None
            row = aifs_15.loc[t_lookup]
            if AIFS_CENTER_COL in aifs_15.columns:
                val = row[AIFS_CENTER_COL]
                if isinstance(val, pd.Series): #handles the case of a duplicate index producing multiple rows
                    try:
                        v = float(val.iloc[0])
                    except Exception:
                        return None
                else:
                    v = float(val)
            else:
                #fallback: use the first numeric column if the expected center column is missing
                found = None
                for c in aifs_15.columns:
                    if np.issubdtype(aifs_15[c].dtype, np.number):
                        found = c
                        break
                if found is None:
                    return None
                if isinstance(row, pd.Series):
                    try:
                        v = float(row[found])
                    except Exception:
                        return None
                else:
                    v = float(row[found])
            if np.isnan(v):
                return None
            vals.append(v)
        return np.asarray(vals, dtype=np.float32)

    def sample_msg_to_grid(msg_df, time, center_lat, center_lon, nx=13, ny=13, dlat=None, dlon=None):
        #for one timestamp, builds a 13x13 target grid centered on the site and nearest-neighbor samples the MSG cloud mask points onto it
        df_t = msg_df[msg_df.index == time]
        if df_t.empty:
            return None
        df_t = df_t.drop_duplicates(subset=["lat", "lon"]).copy()
        pts = np.vstack([df_t["lat"].values, df_t["lon"].values]).T
        vals = df_t["value"].values
        if dlat is None:
            dlat = 0.03 #target grid spacing in degrees
        if dlon is None:
            dlon = 0.03
        half_x = (nx // 2) * dlon
        half_y = (ny // 2) * dlat
        lon_edges = np.linspace(center_lon - half_x, center_lon + half_x, nx)
        lat_edges = np.linspace(center_lat + half_y, center_lat - half_y, ny)
        tgt_lons, tgt_lats = np.meshgrid(lon_edges, lat_edges)
        tgt_pts = np.vstack([tgt_lats.ravel(), tgt_lons.ravel()]).T
        from scipy.spatial import cKDTree
        tree = cKDTree(pts) #builds a spatial index over the actual MSG points at this time
        _, idx = tree.query(tgt_pts, k=1) #finds the nearest actual point for every target grid cell
        return vals[idx].reshape((ny, nx))

    def era5_row_to_5x5_from_series(row):
        #reshapes the 25 r{r}_c{c} columns of one ERA5 row into a proper (5,5) spatial array
        arr = np.zeros((5, 5), dtype=float)
        rs = [-2, -1, 0, 1, 2]
        cs = [-2, -1, 0, 1, 2]
        for i_r, r in enumerate(rs):
            for i_c, c in enumerate(cs):
                col = f"r{r}_c{c}"
                if col not in row.index:
                    raise KeyError(f"ERA5 column {col} not found in DataFrame")
                arr[i_r, i_c] = float(row[col])
        return arr

    #primary strategy: candidate valid times are AIFS timestamps that fall exactly on the hour
    valid_times_all = aifs_15.index.unique().sort_values()
    valid_times = [t for t in valid_times_all if (t.minute == 0 and t.second == 0)]

    #fallback 1: if none are exactly on-hour, build an hourly range spanning the data and keep only timestamps actually present
    if len(valid_times) == 0:
        start = valid_times_all.min().floor("H")
        end = valid_times_all.max().ceil("H")
        hourly_range = pd.date_range(start=start, end=end, freq="1H")
        idx_set = set(valid_times_all)
        valid_times = [t for t in hourly_range if t in idx_set]

    #fallback 2: last resort, round each timestamp down to the hour and keep hours that have at least one matching 15-min mark
    if len(valid_times) == 0:
        cand = []
        for t in valid_times_all:
            t_hr = t.replace(minute=0, second=0, microsecond=0)
            if t_hr not in cand:
                cand.append(t_hr)
        valid_times = [h for h in cand if any(((h + pd.Timedelta(minutes=m)) in valid_times_all) for m in (0, 15, 30, 45))]

    valid_times = pd.Index(sorted(valid_times))
    print(f"[prep] Will attempt to create samples for {len(valid_times)} hourly valid times (first few): {list(valid_times[:5])}")

    msg.index = pd.to_datetime(msg.index, errors="coerce").round("15min") #rounds MSG timestamps onto the 15-min grid
    needed = [pd.Timestamp(t).round("15min") for t in valid_times]

    #diagnostic pass (not used by the main loop below) - checks how much usable MSG data exists at the needed timestamps and why any are unusable
    msg_groups = {}
    skipped_reasons = defaultdict(int)
    for t in sorted(set(msg.index.unique())):
        if t not in set(needed):
            skipped_reasons["not_needed"] += 1
            continue
        g = msg[msg.index == t]
        g = g.drop_duplicates(subset=["lat", "lon"])
        if len(g) < 10: #too sparse to be a usable grid at this time
            skipped_reasons["too_few_points"] += 1
            continue
        if g["value"].isna().any():
            skipped_reasons["has_nans"] += 1
            continue
        pts = g[["lat", "lon"]].values.astype(float)
        vals = g["value"].values.astype(float)
        msg_groups[t] = (pts, vals)

    print(f"[prep] Built groups: kept {len(msg_groups)}, skipped {sum(skipped_reasons.values())}")
    print("[prep] skip breakdown:", dict(skipped_reasons))

    #logs how well the needed timestamps and available MSG timestamps overlap
    needed_set = set(pd.to_datetime(needed).round("15min"))
    msg_set = set(pd.to_datetime(msg.index).round("15min").unique())
    inter = sorted(list(needed_set & msg_set))
    print("[prep] intersection count:", len(inter))
    missing = sorted(list(needed_set - msg_set))
    print("[prep] missing needed (examples):", missing[:10])

    def month_key(ts):
        return (ts.year, ts.month)

    #breaks missing/present coverage down by month, useful for spotting data gaps
    missing_months = Counter(month_key(t) for t in missing)
    present_months = Counter(month_key(t) for t in inter)
    print("[prep] present months (count):", dict(sorted(present_months.items())))
    print("[prep] missing months (count):", dict(sorted(missing_months.items())))

    n_created = 0
    total = len(valid_times)
    #main sample-building loop - one candidate .npz per hourly valid time
    for V in valid_times:
        t_init = V - pd.Timedelta(hours=6) #AIFS forecasts are anchored 6h before the target window

        #1) AIFS baseline vector at t_init+15/30/45/60 min
        aifs_baseline_15min = aifs_15min_vector_for_tinit(t_init)
        if aifs_baseline_15min is None:
            continue

        #2) ERA5 inputs at t_init+1h .. +6h, 3 variables on a 5x5 grid each
        era5_times = [t_init + timedelta(hours=(i + 1)) for i in range(6)]
        era5_patches_per_t = []
        skip = False
        for tt in era5_times:
            if tt not in era5_ssrd.index or tt not in era5_d2m.index or tt not in era5_t2m.index:
                skip = True
                break
            row_ssrd = era5_ssrd.loc[tt]
            row_d2m = era5_d2m.loc[tt]
            row_t2m = era5_t2m.loc[tt]
            if isinstance(row_ssrd, pd.DataFrame): #handles duplicate-index edge case
                row_ssrd = row_ssrd.iloc[0]
            if isinstance(row_d2m, pd.DataFrame):
                row_d2m = row_d2m.iloc[0]
            if isinstance(row_t2m, pd.DataFrame):
                row_t2m = row_t2m.iloc[0]
            try:
                p_ssrd = era5_row_to_5x5_from_series(row_ssrd)
                p_d2m = era5_row_to_5x5_from_series(row_d2m)
                p_t2m = era5_row_to_5x5_from_series(row_t2m)
            except KeyError:
                skip = True
                break
            era5_patches_per_t.append(np.stack([p_ssrd, p_d2m, p_t2m], axis=0)) #stacks the 3 variables into (3,5,5) for this timestep

        if skip:
            continue
        era5_inputs = np.stack(era5_patches_per_t, axis=0).astype(np.float32)  # (6,3,5,5)

        #3) MSG inputs - 8 frames spanning t_init-105min up to t_init in 15-min steps
        msg_times = [t_init - pd.Timedelta(minutes=105) + pd.Timedelta(minutes=15 * i) for i in range(8)]
        msg_arrays = []
        skip = False
        for tt in msg_times:
            grid = sample_msg_to_grid(msg, tt, cfg.lat, cfg.lon, nx=13, ny=13)
            if grid is None:
                skip = True
                break
            msg_arrays.append(grid)
        if skip:
            continue
        msg_inputs = np.stack(msg_arrays, axis=0).astype(np.float32)  # (8,13,13)

        #computes cloud fraction at t_init from the last MSG frame - fraction of valid pixels classified as cloudy (code 2)
        try:
            frame = msg_inputs[-1].astype(float)
            valid_mask = (frame == 0) | (frame == 1) | (frame == 2)
            if np.any(valid_mask):
                cloud_frac_at_t_init = float(np.sum(frame == 2) / np.sum(valid_mask))
            else:
                cloud_frac_at_t_init = float(np.nan)
        except Exception:
            cloud_frac_at_t_init = float(np.nan)

        #4) ground truth target vector at t_init+15/30/45/60 min - this is the actual training target
        ground_times = [t_init + timedelta(minutes=15 * i) for i in range(1, 5)]
        ground_vals_list = []
        missing_ground = False
        for tt in ground_times:
            if tt not in ground_vals.index:
                missing_ground = True
                break
            v = float(ground_vals.loc[tt])
            if np.isnan(v):
                missing_ground = True
                break
            ground_vals_list.append(v)
        if missing_ground:
            continue
        ground_15min = np.asarray(ground_vals_list, dtype=np.float32)  # (4,)

        #5) meta info describing this sample
        meta = {
            "t_init": t_init.isoformat(),
            "valid_times_15min": [t.isoformat() for t in ground_times],
            "era5_times": [t.isoformat() for t in era5_times],
            "msg_times": [t.isoformat() for t in msg_times],
            "ground_times": [t.isoformat() for t in ground_times],
            "cloud_frac_at_t_init": cloud_frac_at_t_init,
            "site_id": str(site_id),
            "center_lat": float(cfg.lat),
            "center_lon": float(cfg.lon),
        }

        tinit_str = t_init.isoformat().replace(":", "-")
        out_fname = cfg.sample_npz(tinit_str)

        #final sanity checks - skip and log rather than save a sample with any NaN in it, on top of the per-source skip checks above
        if np.isnan(aifs_baseline_15min).any():
            print(f"[prep] Skipping {t_init} because AIFS baseline (15min) contains NaN")
            continue
        if np.isnan(ground_15min).any():
            print(f"[prep] Skipping {t_init} because ground (15min) contains NaN")
            continue
        if np.isnan(era5_inputs).any():
            print(f"[prep] Skipping {t_init} because ERA5 contains NaN")
            continue
        if np.isnan(msg_inputs).any():
            print(f"[prep] Skipping {t_init} because MSG contains NaN")
            continue

        #6) saves the compressed .npz with all 4 data arrays plus meta
        np.savez_compressed(
            out_fname,
            era5=era5_inputs,
            msg=msg_inputs,
            aifs_baseline_quarter=aifs_baseline_15min,
            ground_quarter=ground_15min,
            meta=meta,
        )
        n_created += 1
        if n_created % 50 == 0 or n_created < 10: #progress logging
            print(f"[prep] {n_created}/{total} created -> {out_fname}")

    print(f"[prep] Done. Created {n_created} samples in {out_dir}")
    return out_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
#maps step name strings to their functions - used both for validating --steps and for dispatching
STEP_FNS = {
    "aifs": step_aifs,
    "era5": step_era5,
    "msg": step_msg,
    "ground": step_ground_truth,
    "prep": step_sample_prep,
}


def parse_args():
    #defines the CLI - every flag here replaces a constant that used to be hardcoded at the top of one of the five original scripts
    p = argparse.ArgumentParser(description="Run the full StFT raw-download -> .npz sample-prep pipeline for one site.")
    p.add_argument("--site", required=True, help="Site ID, e.g. TOR, BUD, PAY")
    p.add_argument("--lat", required=True, type=float, help="Station center latitude")
    p.add_argument("--lon", required=True, type=float, help="Station center longitude")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (AIFS/ERA5 valid-time range)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--msg-start", default=None, help="Override MSG cloud-mask start date (default: --start)")
    p.add_argument("--msg-end", default=None, help="Override MSG cloud-mask end date (default: --end)")
    p.add_argument("--msg-half-deg", type=float, default=0.30, help="Half-width in degrees of the MSG box")
    p.add_argument("--data-dir", default=None, help="Default: data{SITE}")
    p.add_argument("--samples-dir", default=None, help="Default: samples_{SITE}")
    p.add_argument("--raw-tab-dir", default=None, help="Where {SITE}_radiation_*.tab files live. Default: data-dir")
    p.add_argument("--aifs-source", default="aws", help="ecmwf.opendata.Client source")
    p.add_argument(
        "--steps",
        default="aifs,era5,msg,ground,prep", #default runs the full pipeline in order
        help=f"Comma-separated subset of: {','.join(STEP_FNS)}",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip a step if its output file(s) already exist")
    return p.parse_args()


def main():
    args = parse_args()
    site_id = args.site.upper() #uppercased for consistent filenames/dirs regardless of how it was typed

    #resolves the three directories - CLI override if given, else the standard {site}-based default
    data_dir = Path(args.data_dir) if args.data_dir else Path(f"data{site_id}")
    samples_dir = Path(args.samples_dir) if args.samples_dir else Path(f"samples_{site_id}")
    raw_tab_dir = Path(args.raw_tab_dir) if args.raw_tab_dir else data_dir

    #builds the single config object every step function receives
    cfg = SiteConfig(
        site_id=site_id,
        lat=args.lat,
        lon=args.lon,
        start_date=args.start,
        end_date=args.end,
        data_dir=data_dir,
        samples_dir=samples_dir,
        raw_tab_dir=raw_tab_dir,
        msg_start=args.msg_start,
        msg_end=args.msg_end,
        msg_half_deg=args.msg_half_deg,
        aifs_source=args.aifs_source,
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()] #parses the comma-separated --steps list
    for s in steps: #validates every requested step up front, so a typo fails fast instead of after earlier steps already ran
        if s not in STEP_FNS:
            raise ValueError(f"Unknown step '{s}'. Valid: {', '.join(STEP_FNS)}")

    for s in steps: #runs each requested step in order
        print(f"\n=== [{site_id}] running step: {s} ===")
        STEP_FNS[s](cfg, skip_existing=args.skip_existing)

    print(f"\n=== [{site_id}] pipeline complete ===")


if __name__ == "__main__":
    main()
