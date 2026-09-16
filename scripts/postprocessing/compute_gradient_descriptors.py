#!/usr/bin/env python3
"""
Compute electron-density GRADIENT descriptors for a single structure, and
quantities that relate the gradient to property-prediction errors.

Motivation
----------
The density gradient marks where the charge density is structured (nuclei,
bonds, surfaces). Property errors are not uniform over the density: forces
(Hellmann-Feynman) sample the density where |grad rho| is largest, whereas
the dipole is a smooth first moment. We compute:

Ground-truth inhomogeneity descriptors (how "hard" the density is):
    gt_grad_mean      <|grad rho_GT|>                       [e / A^4]
    gt_grad_integral  sum |grad rho_GT| dV                  [e / A]
    gt_rdg_mean       density-weighted reduced gradient s   [dimensionless]
                      s = |grad rho| / (2 (3 pi^2)^{1/3} rho^{4/3})

Predicted-density inhomogeneity descriptors (GT-FREE; the same descriptors
evaluated on the normalized ML density, usable as predictors when the
ground-truth density is unknown). The ML density is normalized to the
nearest integer of its own electron count (no GT electron count used):
    pred_grad_mean, pred_grad_integral, pred_rdg_mean

Gradient-resolved error descriptors (where the ML error lives):
    grad_weighted_err   sum |grad rho_GT| |rho_ML-rho_GT| dV / sum |grad rho_GT| dV
    error_grad_integral sum |grad(rho_ML - rho_GT)| dV       [e / A]
                        (expected to track force errors)
    abs_error_e         sum |rho_ML-rho_GT| dV               [electrons] (reference)

The ML density is rescaled to the GT electron count first (same convention
as compute_density_error.py). Gradients use periodic central differences in
fractional coordinates mapped to Cartesian via the reciprocal cell.

This is a bare-minimum, single-structure example distilled from the
production multi-category pipeline used for the full validation set in the
manuscript.

Usage:
    python3 compute_gradient_descriptors.py \
        --ml-density prediction_lmax_4/cubes/CHGCAR.npy \
        --gt-chgcar CHGCAR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compute_density_error import get_charge_density, load_gt_chgcar

_C_TF = 2.0 * (3.0 * np.pi ** 2) ** (1.0 / 3.0)  # reduced-gradient prefactor


def cartesian_gradient(rho: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """|grad rho| (Cartesian) via periodic central differences in fractional coords.

    r = f @ cell  =>  d/dr_i = sum_a (cell^{-1})_{i,a} d/df_a, with f_a spacing 1/N_a.
    Computed in float32 to bound memory.
    """
    r = rho.astype(np.float32, copy=False)
    shape = r.shape
    dfa = []
    for a in range(3):
        d = (np.roll(r, -1, axis=a) - np.roll(r, 1, axis=a)) * np.float32(shape[a] / 2.0)
        dfa.append(d)
    inv = np.linalg.inv(cell)
    grad_sq = np.zeros(shape, dtype=np.float32)
    for i in range(3):
        gi = inv[i, 0] * dfa[0] + inv[i, 1] * dfa[1] + inv[i, 2] * dfa[2]
        grad_sq += gi ** 2
    return np.sqrt(grad_sq)


def compute_gradient_descriptors(ml: np.ndarray, gt: np.ndarray, cell: np.ndarray) -> dict:
    if ml.shape != gt.shape:
        raise ValueError(f"Grid mismatch: ml={ml.shape} gt={gt.shape}")

    volume = abs(np.linalg.det(cell))
    dv = volume / float(gt.size)
    n_gt = float(gt.sum() * dv)
    n_ml = float(ml.sum() * dv)

    # --- GT-free predicted-density descriptors -----------------------
    n_pred_target = round(n_ml) if n_ml != 0.0 else 0.0
    ml_self = ml * (n_pred_target / n_ml) if n_ml != 0.0 else ml
    grad_ml = cartesian_gradient(ml_self, cell)
    grad_ml_sum = float(grad_ml.sum())
    ml_pos = np.clip(ml_self, 1e-6, None)
    rdg_ml = grad_ml / (_C_TF * ml_pos.astype(np.float32) ** np.float32(4.0 / 3.0))
    ml_sum = float(ml_self.sum())
    pred_rdg_mean = float((ml_self * rdg_ml).sum() / ml_sum) if ml_sum else float("nan")

    # --- GT descriptors and error descriptors -------------------------
    ml_scaled = ml * (n_gt / n_ml) if n_ml != 0.0 else ml
    err = ml_scaled - gt

    grad_gt = cartesian_gradient(gt, cell)
    grad_err = cartesian_gradient(err, cell)

    grad_gt_sum = float(grad_gt.sum())
    rho_pos = np.clip(gt, 1e-6, None)
    rdg = grad_gt / (_C_TF * rho_pos.astype(np.float32) ** np.float32(4.0 / 3.0))
    rdg_mean = float((gt * rdg).sum() / gt.sum())

    abs_err_e = float(np.abs(err).sum() * dv)
    grad_weighted_err = float((grad_gt * np.abs(err).astype(np.float32)).sum() / grad_gt_sum)

    return {
        "gt_grad_mean": grad_gt_sum / grad_gt.size,
        "gt_grad_integral": grad_gt_sum * dv,
        "gt_rdg_mean": rdg_mean,
        "pred_grad_mean": grad_ml_sum / grad_ml.size,
        "pred_grad_integral": grad_ml_sum * dv,
        "pred_rdg_mean": pred_rdg_mean,
        "grad_weighted_err": grad_weighted_err,
        "error_grad_integral": float(grad_err.sum() * dv),
        "abs_error_e": abs_err_e,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-density", type=Path, required=True)
    ap.add_argument("--gt-chgcar", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    ml = get_charge_density(np.load(args.ml_density))
    gt, cell = load_gt_chgcar(args.gt_chgcar)
    result = compute_gradient_descriptors(ml, gt, cell)

    for k, v in result.items():
        print(f"{k:20s} {v:.6f}")

    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
