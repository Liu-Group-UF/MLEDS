#!/usr/bin/env python3
"""
Compute the charge-density prediction error of an ML-predicted density
against a ground-truth DFT density, for a single structure.

    abs_error_e       = sum |rho_ML - rho_GT| * dV           [electrons]
    mean_abs_e_per_A3 = abs_error_e / V                      [e / A^3]
    rmse_e_per_A3     = sqrt( sum (rho_ML - rho_GT)^2 * dV / V )
    nmae_percent      = abs_error_e / (sum |rho_GT| * dV) * 100

The ML density is rescaled to integrate to the same electron count as the
ground-truth density before comparison, so the error reflects redistribution
of charge rather than a global electron-count offset.

This is a bare-minimum, single-structure example distilled from the
production multi-category pipeline used for the full validation set in the
manuscript; it keeps the same metric definitions but drops the batch
dispatch and structure-type bookkeeping.

Usage:
    python3 compute_density_error.py \
        --ml-density prediction_lmax_4/cubes/CHGCAR.npy \
        --gt-chgcar CHGCAR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def get_charge_density(density: np.ndarray) -> np.ndarray:
    if density.ndim == 4:
        return np.asarray(density[..., 0], dtype=np.float64)
    return np.asarray(density, dtype=np.float64)


def load_gt_chgcar(chgcar_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth charge density (e/A^3) and cell, parsed directly from CHGCAR."""
    with open(chgcar_path, "r") as fh:
        fh.readline()
        scale = float(fh.readline().strip())
        cell = np.array(
            [list(map(float, fh.readline().split())) for _ in range(3)],
            dtype=np.float64,
        ) * scale
        species_line = fh.readline().split()
        counts_line = fh.readline().split()
        try:
            counts = list(map(int, counts_line))
        except ValueError:
            counts = list(map(int, species_line))
        natoms = sum(counts)
        fh.readline()
        for _ in range(natoms):
            fh.readline()
        fh.readline()
        grid_line = fh.readline().split()
        nx, ny, nz = int(grid_line[0]), int(grid_line[1]), int(grid_line[2])
        n_vals = nx * ny * nz
        remaining = fh.read()

    tokens = remaining.split()[:n_vals]
    density_flat = np.array(tokens, dtype=np.float64)
    if density_flat.size != n_vals:
        raise RuntimeError(
            f"Expected {n_vals} density values in {chgcar_path}, got {density_flat.size}"
        )
    cell_volume = abs(np.linalg.det(cell))
    density = density_flat.reshape((nx, ny, nz), order="F") / cell_volume
    return density, cell


def compute_density_error(ml_density: np.ndarray, gt_density: np.ndarray, gt_cell: np.ndarray) -> dict:
    if ml_density.shape != gt_density.shape:
        raise ValueError(f"Grid mismatch: ml={ml_density.shape} gt={gt_density.shape}")

    volume = abs(np.linalg.det(gt_cell))
    dv = volume / float(ml_density.size)

    # Rescale ML so it integrates to the same electron count as GT.
    n_gt = float(gt_density.sum() * dv)
    n_ml = float(ml_density.sum() * dv)
    ml_scaled = ml_density * (n_gt / n_ml) if n_ml != 0.0 else ml_density

    diff = ml_scaled - gt_density
    abs_err_e = float(np.abs(diff).sum() * dv)
    gt_abs_e = float(np.abs(gt_density).sum() * dv)

    return {
        "abs_error_e": abs_err_e,
        "mean_abs_e_per_A3": abs_err_e / volume,
        "rmse_e_per_A3": float(np.sqrt((diff ** 2).sum() * dv / volume)),
        "nmae_percent": abs_err_e / gt_abs_e * 100.0,
        "gt_electrons": n_gt,
        "ml_electrons_raw": n_ml,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-density", type=Path, required=True,
                     help="predicted density .npy (charge3net cube output)")
    ap.add_argument("--gt-chgcar", type=Path, required=True, help="ground-truth CHGCAR")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    ml_density = get_charge_density(np.load(args.ml_density))
    gt_density, gt_cell = load_gt_chgcar(args.gt_chgcar)
    result = compute_density_error(ml_density, gt_density, gt_cell)

    for k, v in result.items():
        print(f"{k:20s} {v:.6f}")

    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
