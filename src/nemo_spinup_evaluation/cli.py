#!/usr/bin/env python3
"""
CLI to compute climate model diagnostics from NEMO restart/output files.

- Loads a DINO setup YAML for variable mapping.
- Runs selected metric suites (restart and/or output).
- Optionally compares against a reference simulation and reports diffs/stats.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, cast

import numpy as np
import xarray as xr
import yaml

from nemo_spinup_evaluation.loader import load_dino_data
from nemo_spinup_evaluation.metrics_io import write_metric_results
from nemo_spinup_evaluation.utils import get_density, get_depth

# Metrics: support both package and direct execution
try:
    from .metrics import (  # type: ignore
        ACC_Drake_metric,
        ACC_Drake_metric_2,
        NASTG_BSF_max,
        check_density,
        temperature_500m_30NS_metric,
        temperature_BWbox_metric,
        temperature_DWbox_metric,
        density_horizontal_mean,
        density_delta,
        density_violation_fraction_by_depth,
        density_500m,
        density_500_3000m,
        density_3000m_bottom,
        density_global,
        check_density_volume_weighted,
        temperature_horizontal_mean,
        salinity_horizontal_mean,
        pointwise_temperature,
        pointwise_salinity,
        pointwise_density,
        pointwise_ssh,
        pointwise_velocity_u,
        pointwise_velocity_v,
    )
except (TypeError, ValueError, KeyError, AttributeError):
    from nemo_spinup_evaluation.metrics import (  # type: ignore
        ACC_Drake_metric,
        ACC_Drake_metric_2,
        NASTG_BSF_max,
        check_density,
        temperature_500m_30NS_metric,
        temperature_BWbox_metric,
        temperature_DWbox_metric,
        density_horizontal_mean,
        density_delta,
        density_violation_fraction_by_depth,
        density_500m,
        density_500_3000m,
        density_3000m_bottom,
        density_global,
        check_density_volume_weighted,
        temperature_horizontal_mean,
        salinity_horizontal_mean,
        pointwise_temperature,
        pointwise_salinity,
        pointwise_density,
        pointwise_ssh,
        pointwise_velocity_u,
        pointwise_velocity_v,
    )


# -----------------------------
# CLI helpers
# -----------------------------
def parse_args(argv) -> argparse.Namespace:
    """Parse command-line arguments for the CLI."""
    parser = argparse.ArgumentParser(
        description="Compute climate model diagnostics from restart or output files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/DINO-setup.yaml",
        help="Path to the DINO setup YAML config file",
    )
    parser.add_argument(
        "--sim-path",
        type=str,
        help="Path to the NEMO simulation directory",
    )
    parser.add_argument(
        "--ref-sim-path",
        type=str,
        help="Path to the reference NEMO simulation directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["output", "restart", "both"],
        default="both",
        help="Process mode: 'output' for output files, 'restart' for restart files, \
        'both' for both",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save output metric values (CSV files).",
    )
    parser.add_argument(
        "--result-file-prefix",
        type=str,
        default="metrics_results",
        help="Prefix to use for the result files.",
    )
    args = parser.parse_args(argv)

    if not args.sim_path and not args.ref_sim_path:
        parser.error("You must provide --sim-path and/or --ref-sim-path.")

    return args


def load_config(config_file: str, sim_path: Optional[str]) -> Mapping[str, Any]:
    """
    Load the DINO-setup.yaml config file for variable mapping.

    Currently uses a fixed path: configs/DINO-setup.yaml
    """
    if not sim_path:
        sys.exit("Error: --sim-path must be provided to load config.")

    if not os.path.exists(config_file):
        sys.exit(f"Error: The file {config_file} does not exist.")

    with open(config_file, "r") as fh:
        return yaml.safe_load(fh)


def ensure_results_dir(results_dir: str) -> None:
    """Ensure the results directory exists."""
    os.makedirs(results_dir, exist_ok=True)


# -----------------------------
# Metric runners
# -----------------------------
def apply_metrics_restart(data: xr.Dataset, mask: xr.Dataset) -> Dict[str, Any]:
    """
    Apply metrics to a standardized restart dataset.

    Returns a dict of {metric_name: result or 'Error: ...'}.
    """
    results: Dict[str, Any] = {}

    metric_fns = {
        "check_density_from_file": lambda d: check_density(d["density"][0]),
        "check_density_computed": lambda d: check_density(
            get_density(
                d["temperature"], d["salinity"], get_depth(d, mask), mask["tmask"]
            )[0]
        ),
        "temperature_horizontal_mean": lambda d: temperature_horizontal_mean(
                    d["temperature"], mask
                ),
        "salinity_horizontal_mean": lambda d: salinity_horizontal_mean(
                    d["salinity"], mask
                ),
        "temperature_500m_30NS_metric": lambda d: temperature_500m_30NS_metric(
            d["temperature"][0], mask
        ),
        "temperature_BWbox_metric": lambda d: temperature_BWbox_metric(
            d["temperature"][0], mask
        ),
        "temperature_DWbox_metric": lambda d: temperature_DWbox_metric(
            d["temperature"][0], mask
        ),
        "ACC_Drake_metric": lambda d: ACC_Drake_metric(d["velocity_u"][0], mask),
        "ACC_Drake_metric_2": lambda d: ACC_Drake_metric_2(
            d["velocity_u"][0], d["ssh"][0], mask
        ),
        "NASTG_BSF_max": lambda d: NASTG_BSF_max(d["velocity_v"][0], d["ssh"], mask),
        "density_horizontal_mean": lambda d: density_horizontal_mean(
            d["density"], mask
        ),
        "density_delta": lambda d: density_delta(
            d["density"], mask
        ),
        "density_violation_fraction_by_depth": lambda d: (
            density_violation_fraction_by_depth(d["density"])
        ),
        "density_500m": lambda d: density_500m(
            d["density"], mask
        ),
        "density_500_3000m": lambda d: density_500_3000m(
            d["density"], mask
        ),
        "density_3000m_bottom": lambda d: density_3000m_bottom(
            d["density"], mask
        ),
        "density_global": lambda d: density_global(
            d["density"], mask
        ),
        "check_density_volume_weighted": lambda d: check_density_volume_weighted(
            get_density(
                d["temperature"], d["salinity"], get_depth(d, mask), mask["tmask"]
            )[0], mask
        ),
        "pointwise_temperature": lambda d: pointwise_temperature(
            d["temperature"][0], mask
        ),
        "pointwise_salinity": lambda d: pointwise_salinity(
            d["salinity"][0], mask
        ),
        "pointwise_density": lambda d: pointwise_density(
            d["density"][0], mask
        ),
        "pointwise_ssh": lambda d: pointwise_ssh(
            d["ssh"][0], mask
        ),
        "pointwise_velocity_u": lambda d: pointwise_velocity_u(
            d["velocity_u"][0], mask
        ),
        "pointwise_velocity_v": lambda d: pointwise_velocity_v(
            d["velocity_v"][0], mask
        ),
        
    
    }

    for name, fn in metric_fns.items():
        try:
            results[name] = fn(data)
        except (ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
            msg = f"Error in metric {name}: {e}"
            warnings.warn(msg, stacklevel=2)
            results[name] = f"Error: {e}"

    return results


def apply_metrics_output(
    grid_output: xr.Dataset,
    restart: Optional[xr.Dataset],
    mask: xr.Dataset,
) -> Dict[str, Any]:
    """
    Apply metrics to standardized output datasets.

    If restart is supplied, adds a computed density check using restart depth.
    """
    results: Dict[str, Any] = {}

    metric_fns = {
        "check_density_from_file": lambda: check_density(grid_output["density"]),
        "temperature_500m_30NS_metric": lambda: temperature_500m_30NS_metric(
            grid_output["temperature"], mask
        ),
        "temperature_BWbox_metric": lambda: temperature_BWbox_metric(
            grid_output["temperature"], mask
        ),
        "temperature_DWbox_metric": lambda: temperature_DWbox_metric(
            grid_output["temperature"], mask
        ),
        "temperature_horizontal_mean": lambda grid_output: temperature_horizontal_mean(
            grid_output["temperature"], mask
        ),
        "salinity_horizontal_mean": lambda grid_output: salinity_horizontal_mean(
            grid_output["salinity"], mask
        ),
        "ACC_Drake_metric": lambda: ACC_Drake_metric(grid_output["velocity_u"], mask),
        "ACC_Drake_metric_2": lambda: ACC_Drake_metric_2(
            grid_output["velocity_u"], grid_output["ssh"], mask
        ),
        "NASTG_BSF_max": lambda: NASTG_BSF_max(
            grid_output["velocity_v"], grid_output["ssh"], mask
        ),
                "density_horizontal_mean": lambda: density_horizontal_mean(
            grid_output["density"], mask
        ),
        "density_delta": lambda: density_delta(
            grid_output["density"], mask
        ),
        "density_violation_fraction_by_depth": lambda: (
            density_violation_fraction_by_depth(grid_output["density"])
        ),
        "density_500m": lambda: density_500m(
            grid_output["density"], mask
        ),
        "density_500_3000m": lambda: density_500_3000m(
            grid_output["density"], mask
        ),
        "density_3000m_bottom": lambda: density_3000m_bottom(
            grid_output["density"], mask
        ),
        "density_global": lambda: density_global(
            grid_output["density"], mask
        ),
        "pointwise_temperature": lambda: pointwise_temperature(
            grid_output["temperature"], mask
        ),
        "pointwise_salinity": lambda: pointwise_salinity(
            grid_output["salinity"], mask
        ),
        "pointwise_density": lambda: pointwise_density(
            grid_output["density"], mask
        ),
        "pointwise_ssh": lambda: pointwise_ssh(
            grid_output["ssh"], mask
        ),
        "pointwise_velocity_u": lambda: pointwise_velocity_u(
            grid_output["velocity_u"], mask
        ),
        "pointwise_velocity_v": lambda: pointwise_velocity_v(
            grid_output["velocity_v"], mask
        ),
    }

    if restart is not None:
        metric_fns["check_density_computed"] = lambda: check_density(
            get_density(
                grid_output["temperature"],
                grid_output["salinity"],
                get_depth(restart, mask),
                mask["tmask"],
            )[0]
        )

    for name, fn in metric_fns.items():
        try:
            results[name] = fn()
        except (ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
            msg = f"Error in metric {name}: {e}"
            warnings.warn(msg, stacklevel=2)
            results[name] = f"Error: {e}"

    return results


# -----------------------------
# Diff & stats
# -----------------------------


def _compute_means(
    a: xr.DataArray, b: xr.DataArray, keep_dims: tuple = ()
) -> Tuple[xr.DataArray, xr.DataArray, float, float]:
    """
    Compare two 1D time series DataArrays.

    Returns
    -------
      - delta: signed difference (a - b) [DataArray]
      - abs_err: absolute error |a - b| [DataArray]
      - mae: mean absolute error (float)
      - rmse: root mean square error (float)
    """
    # Align along time (inner join ensures only overlapping points)
    a_aln, b_aln = xr.align(a, b, join="inner")

    # Compute diffs
    delta = a_aln - b_aln
    abs_err = abs(delta)

    reduce_dims = [d for d in delta.dims if d not in keep_dims]

    if reduce_dims:
        mae  = abs_err.mean(dim=reduce_dims, skipna=True)
        rmse = (delta ** 2).mean(dim=reduce_dims, skipna=True) ** 0.5
    else:
        # Nothing to reduce: MAE and RMSE of a single sample are both |delta|.
        # This is mathematically correct, but a warning fires so a caller
        # passing an over-broad keep_dims notices the accidental no-op.
        warnings.warn(
            f"_compute_means called with keep_dims={keep_dims!r} covering "
            f"all of {delta.dims!r}; MAE and RMSE both equal |delta|.",
            stacklevel=2,
        )
        mae = rmse = abs_err

    if not keep_dims:
        mae = float(abs_err.mean(skipna=True).values.item())
        rmse = float((delta**2).mean(skipna=True).values.item() ** 0.5)

    return delta, abs_err, mae, rmse


def compute_diffs_and_stats(
    results: Mapping[str, xr.DataArray],
    ref_results: Mapping[str, xr.DataArray],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compute signed diff, absolute error, and MAE/RMSE vs reference for xarray inputs.

    Expect `ref_results_with_prefix` to have keys like 'ref_<key>'.

    Returns
    -------
      - diffs: {'diff_<key>': DataArray, 'diff_<key>_ae': DataArray}
      - stats: {'diff_<key>_mae': float, 'diff_<key>_rmse': float}
    """
    diffs: Dict[str, Any] = {}
    stats: Dict[str, Any] = {}

    for key, val in results.items():
        ref_key = f"ref_{key}"
        if ref_key not in ref_results:
            continue

        ref_val = ref_results[ref_key]

        if not isinstance(val, xr.DataArray) or not isinstance(ref_val, xr.DataArray):
            err = "Error: expected xarray.DataArray on both sides"
            diffs[f"diff_{key}"] = err
            stats[f"diff_{key}_mae"] = err
            stats[f"diff_{key}_rmse"] = err
            continue

        try:
            # Pointwise fields keep time_counter to get a per-quarter
            # MAE/RMSE time series. All other metrics reduce over time to a
            # single scalar, as before.
            keep = ("time_counter",) if key.startswith("pointwise_") else ()
            delta, abs_err, mae, rmse = _compute_means(
                val, ref_val, keep_dims=keep
            )
            delta = delta.rename(f"diff_{key}")

            # Don't store the 4D diff array for pointwise metrics
            # (1.4 GB each, recomputable as sim - ref).
            if not key.startswith("pointwise_"):
                diffs[f"diff_{key}"] = delta

            # Pointwise: mae/rmse are DataArrays over time_counter
            # handled by _build_timeseries_dataframe.
            # Non-pointwise: mae/rmse are scalars
            # handled by _build_summary_dataframe.
            if isinstance(mae, xr.DataArray):
                stats[f"diff_{key}_mae"] = mae
                stats[f"diff_{key}_rmse"] = rmse
            else:
                stats[f"diff_{key}_mae"] = mae if np.isfinite(mae) else np.nan
                stats[f"diff_{key}_rmse"] = rmse if np.isfinite(rmse) else np.nan

        except (ValueError, TypeError, KeyError, AttributeError, ImportError) as e:
            err = f"Error: {e}"
            if not key.startswith("pointwise_"):
                diffs[f"diff_{key}"] = err
                diffs[f"diff_{key}_ae"] = err
            stats[f"diff_{key}_mae"] = err
            stats[f"diff_{key}_rmse"] = err

    return diffs, stats


# restart mode
def run_restart_metrics(
    data: Mapping[str, Any], data_ref: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Run restart metrics and compute diffs/stats if reference provided."""
    results = apply_metrics_restart(data["restart"], data["mesh_mask"])
    if data_ref is not None:
        ref = apply_metrics_restart(data_ref["restart"], data_ref["mesh_mask"])
        ref_prefixed = {f"ref_{k}": v for k, v in ref.items()}
        diffs, stats = compute_diffs_and_stats(results, ref_prefixed)
        results.update(ref_prefixed)
        results.update(diffs)
        results.update(stats)
    return results


# output mode
def run_output_metrics(
    data: Mapping[str, Any], data_ref: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Run output metrics and compute diffs/stats if reference provided."""
    results = apply_metrics_output(data["grid"], data.get("restart"), data["mesh_mask"])
    if data_ref is not None:
        ref = apply_metrics_output(
            data_ref["grid"], data_ref.get("restart"), data_ref["mesh_mask"]
        )
        ref_prefixed = {f"ref_{k}": v for k, v in ref.items()}
        diffs, stats = compute_diffs_and_stats(results, ref_prefixed)
        results.update(ref_prefixed)
        results.update(diffs)
        results.update(stats)
    return results


def main(argv=None) -> int:
    """Command-line interface for computing climate model diagnostics."""
    args = parse_args(argv)
    dino_setup = load_config(args.config, args.sim_path)
    ensure_results_dir(args.results_dir)
    prefix: str = args.result_file_prefix

    # Load primary and (optional) reference datasets
    data = load_dino_data(args.mode, args.sim_path, dino_setup)

    data_ref = None
    if args.ref_sim_path:
        data_ref = load_dino_data(args.mode, args.ref_sim_path, dino_setup)

    # Extract paths with proper type cast
    paths = cast(dict[str, str], data.get("paths", {}))
    print("Data paths:", paths)

    # Run suites
    if args.mode in {"restart", "both"}:
        results_restart = run_restart_metrics(data, data_ref)
        out_path = os.path.join(args.results_dir, f"{prefix}_restart")
        write_metric_results(results_restart, out_path, paths)

    if args.mode in {"output", "both"}:
        results_grid = run_output_metrics(data, data_ref)
        out_path = os.path.join(args.results_dir, f"{prefix}_grid")
        write_metric_results(results_grid, out_path, paths)

    return 0


##################################
### Running spinup-evaluation ####
##################################
# python3 main.py --sim-path /path/to/simulation_directory
#                 --ref-sim-path /path/to/reference_simulation_directory
#                 --results-dir /path/to/results_directory
#                 --result-file-prefix my_results
#                 --mode output
#
# The code now uses a configuration YAML (e.g., configs/DINO-setup.yaml)
# to determine which files to load and which variables to map.

if __name__ == "__main__":
    sys.exit(main())
