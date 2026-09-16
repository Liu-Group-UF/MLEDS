#!/usr/bin/env python3
"""
Compute the dipole moment and Bader charges of an ML-predicted electron
density vs. a ground-truth DFT density, for a single structure.

The predicted density is normalized once to the GT-integrated electron count,
and that single normalized density is reused for both the dipole integration
and the Bader analysis, so both numbers describe the same density.

Bader charges are computed with the external `bader` executable (Henkelman
group, http://theory.cm.utexas.edu/henkelman/code/bader/). ASE's own
VaspChargeDensity writer produces a CHGCAR that `bader` cannot parse
("Divide by zero in matrix inverse"), so the density is instead written using
the reference CHGCAR's verbatim header/augmentation with values scaled by
cell volume (`write_chgcar_for_bader`).

This is a bare-minimum, single-structure example distilled from the
production multi-category pipeline used for the full validation set in the
manuscript (which also aggregates per-case JSON results into summary CSVs
and plots; that aggregation step is omitted here).

Usage:
    python3 compute_dipole_and_bader.py \
        --ml-density prediction_lmax_4/cubes/CHGCAR.npy \
        --gt-chgcar CHGCAR \
        --poscar POSCAR --potcar POTCAR \
        --bader-executable bader
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase.calculators.vasp import VaspChargeDensity


def get_charge_density(density: np.ndarray) -> np.ndarray:
    if density.ndim == 4:
        return np.asarray(density[..., 0], dtype=np.float64)
    return np.asarray(density, dtype=np.float64)


def compute_dipole_moment(
    density: np.ndarray,
    cell: np.ndarray,
    reference_fractional: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    charge_density = get_charge_density(density)
    nx, ny, nz = charge_density.shape
    frac_x = np.arange(nx, dtype=np.float64) / nx
    frac_y = np.arange(ny, dtype=np.float64) / ny
    frac_z = np.arange(nz, dtype=np.float64) / nz
    frac_grid = np.stack(np.meshgrid(frac_x, frac_y, frac_z, indexing="ij"), axis=-1)
    reference_cart = np.dot(np.asarray(reference_fractional, dtype=np.float64), cell)
    cart_grid = np.dot(frac_grid, cell) - reference_cart
    voxel_volume = abs(np.linalg.det(cell)) / float(nx * ny * nz)
    return np.sum(charge_density[..., None] * cart_grid, axis=(0, 1, 2)) * voxel_volume


def compute_electron_count(density: np.ndarray, cell: np.ndarray) -> float:
    voxel_volume = abs(float(np.linalg.det(cell))) / float(np.prod(density.shape))
    return float(np.sum(density) * voxel_volume)


def parse_poscar_species_counts(poscar_path: Path) -> tuple[list[str], list[int]]:
    with open(poscar_path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    if len(lines) < 7:
        raise ValueError(f"POSCAR appears too short: {poscar_path}")
    return lines[5].split(), [int(v) for v in lines[6].split()]


def parse_potcar_zvals(potcar_path: Path) -> list[float]:
    zvals: list[float] = []
    pattern = re.compile(r"ZVAL\s*=\s*([0-9.]+)")
    with open(potcar_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                zvals.append(float(m.group(1)))
    if not zvals:
        raise ValueError(f"No ZVAL entries in POTCAR: {potcar_path}")
    return zvals


def load_species_and_pseudo_charges(poscar: Path, potcar: Path) -> tuple[list[str], np.ndarray]:
    species, counts = parse_poscar_species_counts(poscar)
    zvals = parse_potcar_zvals(potcar)
    if len(zvals) < len(species):
        raise ValueError("POTCAR has fewer ZVAL blocks than POSCAR species")
    expanded_species: list[str] = []
    charges: list[float] = []
    for symbol, count, zval in zip(species, counts, zvals[: len(species)]):
        expanded_species.extend([symbol] * count)
        charges.extend([zval] * count)
    return expanded_species, np.asarray(charges, dtype=np.float64)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _find_grid_line(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return index
    raise ValueError("Could not find the density grid-dimensions line")


def _find_augmentation_start(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if "augmentation occupancies" in line.lower():
            return index
    for index in range(len(lines) - 1, -1, -1):
        parts = lines[index].strip().split()
        if parts and not all(_is_float(p) for p in parts):
            return index
    return None


def write_chgcar_for_bader(density: np.ndarray, reference_chgcar: Path, output_path: Path) -> None:
    """Write a CHGCAR for bader using the reference file's header and augmentation.

    ASE's VaspChargeDensity writer produces a CHGCAR `bader` cannot parse; the
    workaround is to keep the reference header/augmentation verbatim and only
    replace the density values (in VASP's cell-volume-scaled format).
    """
    ref_lines = reference_chgcar.read_text().splitlines(keepends=True)
    grid_line = _find_grid_line(ref_lines)
    aug_start = _find_augmentation_start(ref_lines)

    ref_rec = VaspChargeDensity(filename=str(reference_chgcar))
    cell_volume = abs(float(np.linalg.det(ref_rec.atoms[-1].cell.array)))
    flat = (np.asarray(density, dtype=np.float64) * cell_volume).flatten(order="F")

    lines_out = list(ref_lines[: grid_line + 1])
    for i in range(0, len(flat), 5):
        chunk = flat[i : i + 5]
        lines_out.append(" " + " ".join(f"{v:.11E}" for v in chunk) + "\n")
    if aug_start is not None:
        lines_out.extend(ref_lines[aug_start:])

    output_path.write_text("".join(lines_out))


def run_bader_get_electrons(chgcar_path: Path, bader_exe: str) -> np.ndarray:
    result = subprocess.run(
        [bader_exe, chgcar_path.name], cwd=str(chgcar_path.parent),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bader failed (exit {result.returncode}):\n{result.stderr}")
    acf_path = chgcar_path.parent / "ACF.dat"
    if not acf_path.exists():
        raise FileNotFoundError(f"bader ran but produced no ACF.dat in {chgcar_path.parent}")
    electrons: list[float] = []
    with open(acf_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0].isdigit() and len(parts) >= 5:
                electrons.append(float(parts[4]))
    return np.asarray(electrons, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-density", type=Path, required=True)
    ap.add_argument("--gt-chgcar", type=Path, required=True)
    ap.add_argument("--poscar", type=Path, required=True)
    ap.add_argument("--potcar", type=Path, required=True,
                     help="VASP POTCAR for this structure (not redistributed; obtain your own licensed copy)")
    ap.add_argument("--bader-executable", type=str, default="bader")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    predicted = get_charge_density(np.load(args.ml_density))
    ground_truth_rec = VaspChargeDensity(filename=str(args.gt_chgcar))
    ground_truth = np.asarray(ground_truth_rec.chg[-1], dtype=np.float64)
    atoms = ground_truth_rec.atoms[-1]
    cell = np.asarray(atoms.cell.array, dtype=np.float64)

    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Grid mismatch: pred={predicted.shape} gt={ground_truth.shape}")

    pred_electrons_raw = compute_electron_count(predicted, cell)
    gt_electrons = compute_electron_count(ground_truth, cell)
    pred_scale = gt_electrons / pred_electrons_raw
    pred_normalized = predicted * pred_scale

    pred_dipole = compute_dipole_moment(pred_normalized, cell)
    gt_dipole = compute_dipole_moment(ground_truth, cell)
    dipole_delta = pred_dipole - gt_dipole

    species, pseudo_charges = load_species_and_pseudo_charges(args.poscar, args.potcar)

    with tempfile.TemporaryDirectory(prefix="dipole_bader_") as tmp:
        tmp_path = Path(tmp)
        pred_chgcar = tmp_path / "pred_CHGCAR"
        write_chgcar_for_bader(pred_normalized, args.gt_chgcar, pred_chgcar)
        pred_electrons = run_bader_get_electrons(pred_chgcar, args.bader_executable)
        gt_electrons_bader = run_bader_get_electrons(args.gt_chgcar, args.bader_executable)

    if len(pred_electrons) != len(gt_electrons_bader) or len(pred_electrons) != len(pseudo_charges):
        raise ValueError("Atom-count mismatch between Bader output and POSCAR/POTCAR")

    delta_electrons = pred_electrons - gt_electrons_bader
    pred_net_charge = pseudo_charges - pred_electrons
    gt_net_charge = pseudo_charges - gt_electrons_bader

    result = {
        "num_atoms": int(len(pseudo_charges)),
        "pred_electron_count_raw": float(pred_electrons_raw),
        "gt_electron_count": float(gt_electrons),
        "pred_density_scale_to_gt": float(pred_scale),
        "pred_dipole": pred_dipole.tolist(),
        "gt_dipole": gt_dipole.tolist(),
        "delta_dipole": dipole_delta.tolist(),
        "abs_delta_dipole_magnitude": float(np.linalg.norm(dipole_delta)),
        "mean_abs_delta_bader_electrons": float(np.mean(np.abs(delta_electrons))),
        "atoms": [
            {
                "index": i, "element": species[i],
                "pseudo_ion_charge": float(pseudo_charges[i]),
                "pred_bader_electrons": float(pred_electrons[i]),
                "gt_bader_electrons": float(gt_electrons_bader[i]),
                "pred_net_charge": float(pred_net_charge[i]),
                "gt_net_charge": float(gt_net_charge[i]),
            }
            for i in range(len(pseudo_charges))
        ],
    }

    print(f"|delta dipole|             {result['abs_delta_dipole_magnitude']:.6f} e*Ang")
    print(f"mean |delta Bader e-|/atom {result['mean_abs_delta_bader_electrons']:.6f}")
    for row in result["atoms"]:
        print(f"  {row['index']:3d} {row['element']:2s}  "
              f"pred_net_q={row['pred_net_charge']:+.3f}  gt_net_q={row['gt_net_charge']:+.3f}")

    if args.out_json:
        args.out_json.write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
