"""
Provide functions for writing metric results to a CSV file.

Includes handling both time series and static metrics using xarray DataArrays.
"""
# nemo_spinup_evaluation/metrics_io.py

from __future__ import annotations

import os
from typing import Any, Mapping

import pandas as pd
import xarray as xr  # type: ignore


def _build_timeseries_dataframe(results: Mapping[str, Any]) -> pd.DataFrame:
    """
    Convert metric results into a timeseries DataFrame.

    Returns DataFrame with 'timestamp' column plus all time-varying and scalar metrics.
    """
    ts_cols = {}
    scalar_cols = {}
    timestamp = None
    for k, v in results.items():
        if (
            isinstance(v, xr.DataArray)
            and "time_counter" in v.dims
            and v.ndim == 1
            and "depth" not in v.dims
        ):
            if timestamp is None:  # pick canonical time axis
                timestamp = [str(t) for t in v["time_counter"].values]
            ts_cols[k] = v.values
        elif (isinstance(v, xr.DataArray) and "depth" in v.dims):
            continue
        elif isinstance(v, xr.DataArray) and v.ndim == 0:
            scalar_cols[k] = v.item()
        elif isinstance(v, xr.DataArray):
            # Remaining DataArrays (e.g. pointwise_ssh with dims
            # (time_counter, y, x)) have no CSV representation here.
            continue
        elif not k.endswith(("_mae", "_rmse")):
            scalar_cols[k] = v

    if timestamp is None:  # restart-only case
        ts_df = pd.DataFrame({
            k: [v] for k, v in scalar_cols.items()
        })
        ts_df.insert(0, "timestamp", [None])
    else:
        #Repeat scalar values for each timestamp
        for k, v in scalar_cols.items():
            ts_cols[k] = [v]*len(timestamp)
        ts_df = pd.DataFrame(ts_cols, index=timestamp).reset_index()
        ts_df = ts_df.rename(columns={"index": "timestamp"})

    return ts_df


def _build_summary_dataframe(results: Mapping[str, Any]) -> pd.DataFrame:
    """
    Extract MAE and RMSE statistics from results into a summary DataFrame.

    Returns DataFrame with columns: metric, mae, rmse.
    """
    rows = []
    for k, v in results.items():
        if k.startswith("diff_") and k.endswith(("_mae", "_rmse")):
            if isinstance(v, xr.DataArray):
                continue  
            if k.endswith("_mae"):
                metric = k[len("diff_") : -4]  # strip "_mae"
                stat = "mae"
            elif k.endswith("_rmse"):
                metric = k[len("diff_") : -5]  # strip "_rmse"
                stat = "rmse"
            else:
                continue
            rows.append((metric, stat, v))

    return (
        pd.DataFrame(rows, columns=["metric", "stat", "value"])
        .pivot(index="metric", columns="stat", values="value")
        .reset_index()
    )

def _build_depth_dataframe(results: Mapping[str, Any]) -> pd.DataFrame:
    """
    Convert depth-resolved metrics to a long-format DataFrame.

    Supports metrics with different depth lengths, such as:
      - density_horizontal_mean: 36 levels
      - density_delta: 35 adjacent-level differences
      - density_violation_fraction_by_depth: 35 adjacent-level pairs

    Returns a DataFrame with timestamp, depth, and one column per metric.
    Missing metric/depth combinations are left as NaN.
    """
    depth_cols = {}
    rows_by_key = {}

    for k, v in results.items():
        if (
            isinstance(v, xr.DataArray)
            and "time_counter" in v.dims
            and "depth" in v.dims
            and v.ndim == 2
        ):
            # These metrics are expected to be 2D: time_counter x depth.
            v = v.transpose("time_counter", "depth")

            timestamps = [str(t) for t in v["time_counter"].values]

            if "depth" in v.coords:
                depths = v["depth"].values
            else:
                depths = range(v.sizes["depth"])

            values = v.values

            for i, timestamp in enumerate(timestamps):
                for j, depth in enumerate(depths):
                    # Convert NumPy scalar coordinates to native scalars.
                    depth = depth.item() if hasattr(depth, "item") else depth

                    key = (timestamp, depth)

                    if key not in rows_by_key:
                        rows_by_key[key] = {
                            "timestamp": timestamp,
                            "depth": depth,
                        }

                    rows_by_key[key][k] = values[i, j]

    if not rows_by_key:
        return pd.DataFrame()

    return pd.DataFrame(list(rows_by_key.values()))


def _format_metadata_header(paths: Mapping[str, str]) -> str:
    """
    Generate CSV comment header with file path metadata.

    Returns multi-line string with # comment prefix for each line.
    """
    lines = []
    lines.append("# Metric Evaluation")
    lines.append(f"# Base Directory: {paths.get('base', 'N/A')}")
    lines.append(f"# Mesh Mask: {paths.get('mesh_mask', 'N/A')}")

    if paths.get("restart"):
        lines.append(f"# Restart File: {paths['restart']}")

    output_files = paths.get("output_files", [])
    if output_files:
        lines.append("# Output Files:")
        for f in output_files:
            lines.append(f"#   - {f}")

    lines.append("#")  # blank comment line before data

    return "\n".join(lines) + "\n"


def write_metric_results(
    results: Mapping[str, Any], out_path: str, paths: Mapping[str, str]
) -> None:
    """
    Write metric results to CSV files.

    Writes two CSVs:
      - {out_path}.csv : timestamp + sim/ref/diff/diff_ae series
      - {out_path}_summary.csv : metric, mae, rmse
    """
    # Build dataframes
    ts_df = _build_timeseries_dataframe(results)
    summary_df = _build_summary_dataframe(results)
    depth_df = _build_depth_dataframe(results)

    # Prepare paths
    ts_path = f"{out_path}.csv"
    sm_path = f"{out_path}_summary.csv"
    depth_path = f"{out_path}_depth.csv"
    os.makedirs(os.path.dirname(os.path.abspath(ts_path)), exist_ok=True)

    # Generate metadata header
    metadata_header = _format_metadata_header(paths)

    # Write timeseries
    print(f"Writing timeseries metrics to '{ts_path}'")
    with open(ts_path, "w") as f:
        f.write(metadata_header)
        ts_df.to_csv(f, float_format="%.6f", index=False)

    # Write summary
    print(f"Writing summary metrics to '{sm_path}'")
    with open(sm_path, "w") as f:
        f.write(metadata_header)
        summary_df.to_csv(f, float_format="%.6f", index=False)
    
    if not depth_df.empty:
        print(f"Writing depth-resolved metrics to '{depth_path}' ")
        with open(depth_path, "w") as f:
            f.write(metadata_header)
            depth_df.to_csv(f, float_format="%.6f", index= False)