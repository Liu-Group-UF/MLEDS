#!/usr/bin/env python3
"""
Local + global electron-density-error descriptors for interpreting ENERGY and
FORCE errors of an ML-predicted density, for a single structure.

Motivation
----------
The error proxies here are the non-self-consistent (Harris) deviations between
a single-point SCF calculation on the DFT ground-truth density and a
non-self-consistent (fixed-density) calculation seeded with the ML-predicted
density:

    energy_dev = E_nscf - E_scf            (meV/atom)
    dF_I       = F_nscf,I - F_scf,I        (per atom, eV/Ang)

Both are driven by the same density error  drho = rho_ML - rho_scf(GT).

  * ENERGY is variationally stationary -> the error is 2nd order in drho. The
    right descriptor is a *quadratic* functional of drho. We compute the
    Hartree self-energy  E_H[drho] = 1/2 integral integral drho drho'/|r-r'|
    (leading electrostatic 2nd-order penalty) and the L2 norm integral drho^2.

  * FORCE error is 1st order and *local*: the Hellmann-Feynman force is a
    linear functional of the density, so dF_I is driven by the near-nuclear,
    1/r^2-weighted *dipole* of drho around atom I. Per atom, over a sphere of
    radius r_c about the nucleus (minimum-image / periodic):
        abs_q   = integral |drho| dV                       (scalar)
        q       = integral drho dV                          (scalar, signed)
        dip     = integral drho (r-R_I) dV                  (vector)
        hf      = integral drho (r-R_I)/|r-R_I|^3 dV        (vector, HF integrand)

This is a bare-minimum, single-structure example distilled from the
production multi-category pipeline used for the full validation set in the
manuscript (which also computes dataset-wide Pearson/Spearman correlations
between these descriptors and the SCF/NSCF force-energy deviations; that
aggregation step is omitted here).

Usage:
    python3 compute_force_energy_descriptors.py \
        --ml-density prediction_lmax_4/cubes/CHGCAR.npy \
        --gt-chgcar CHGCAR \
        --scf-json scf_energy_forces.json \
        --nscf-json nscf_energy_forces.json

scf_energy_forces.json / nscf_energy_forces.json each look like:
    {"energy_fr": -1234.5, "n_atoms": 32, "forces": [[fx, fy, fz], ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.calculators.vasp import VaspChargeDensity

from compute_density_error import get_charge_density, load_gt_chgcar
from compute_gradient_descriptors import _C_TF, cartesian_gradient

# Coulomb constant: energy (eV) of two electrons 1 Ang apart, i.e. e^2/Ang.
K_E = 14.399645
# PAW pseudopotential valence (ZVAL) for the elements in this study.
ZVAL = {"Cu": 11.0, "O": 6.0}


def hartree_self_energy(drho: np.ndarray, cell: np.ndarray) -> float:
    """E_H[drho] = 1/2 int int drho(r) drho(r') / |r-r'|  in eV (periodic, G!=0).

    drho should be ~charge-neutral (the G=0 term is dropped); both densities
    are normalized to the same electron count upstream, so int drho ~ 0.
    """
    nx, ny, nz = drho.shape
    ngrid = nx * ny * nz
    volume = abs(np.linalg.det(cell))

    f = np.fft.fftn(drho)
    recip = np.linalg.inv(cell).T
    mx = np.fft.fftfreq(nx) * nx
    my = np.fft.fftfreq(ny) * ny
    mz = np.fft.fftfreq(nz) * nz

    g2 = np.zeros((nx, ny, nz), dtype=np.float64)
    for c in range(3):
        gc = (mx * recip[0, c])[:, None, None] \
           + (my * recip[1, c])[None, :, None] \
           + (mz * recip[2, c])[None, None, :]
        g2 += gc * gc
    g2 *= (2.0 * np.pi) ** 2
    g2[0, 0, 0] = np.inf  # drop G=0

    e_h = K_E * 2.0 * np.pi * volume / (ngrid ** 2) * np.sum(np.abs(f) ** 2 / g2)
    return float(e_h)


def atom_sphere_descriptors(
    drho: np.ndarray, cell: np.ndarray, scaled_pos: np.ndarray, r_c: float, r_min: float,
) -> dict:
    """Integrate drho-moments in a periodic sphere of radius r_c about one atom."""
    nx, ny, nz = drho.shape
    ns = np.array([nx, ny, nz])
    dv = abs(np.linalg.det(cell)) / drho.size
    inv_cell = np.linalg.inv(cell)

    hw = r_c * np.linalg.norm(inv_cell, axis=0)
    steps = np.ceil(hw * ns).astype(int) + 1
    center = np.round(scaled_pos * ns).astype(int)

    offs = [np.arange(-steps[d], steps[d] + 1) for d in range(3)]
    idx = [(center[d] + offs[d]) % ns[d] for d in range(3)]
    df = [((center[d] + offs[d]) / ns[d]) - scaled_pos[d] for d in range(3)]
    df = [d - np.round(d) for d in df]

    IX, IY, IZ = np.meshgrid(idx[0], idx[1], idx[2], indexing="ij")
    DX, DY, DZ = np.meshgrid(df[0], df[1], df[2], indexing="ij")
    dfrac = np.stack([DX.ravel(), DY.ravel(), DZ.ravel()], axis=1)
    dcart = dfrac @ cell
    dist = np.linalg.norm(dcart, axis=1)

    in_sphere = dist <= r_c
    flat = (IX.ravel(), IY.ravel(), IZ.ravel())
    vals = drho[flat][in_sphere]
    dcart = dcart[in_sphere]
    dist = dist[in_sphere]

    w = vals * dv
    q = float(w.sum())
    abs_q = float(np.abs(w).sum())
    dip = (w[:, None] * dcart).sum(axis=0)

    core = dist > r_min
    inv_r3 = np.zeros_like(dist)
    inv_r3[core] = 1.0 / dist[core] ** 3
    hf = (w[:, None] * dcart * inv_r3[:, None]).sum(axis=0)

    return {"q": q, "abs_q": abs_q, "dip": dip, "hf": hf}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-density", type=Path, required=True)
    ap.add_argument("--gt-chgcar", type=Path, required=True)
    ap.add_argument("--scf-json", type=Path, required=True)
    ap.add_argument("--nscf-json", type=Path, required=True)
    ap.add_argument("--r-c", type=float, default=2.0, help="sphere radius (Ang) for per-atom descriptors")
    ap.add_argument("--r-min", type=float, default=0.1, help="inner cutoff (Ang) for the 1/r^2 HF integrand")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    ml_density = get_charge_density(np.load(args.ml_density))
    gt_density, gt_cell = load_gt_chgcar(args.gt_chgcar)
    if ml_density.shape != gt_density.shape:
        raise ValueError(f"Grid mismatch: ml={ml_density.shape} gt={gt_density.shape}")

    atoms = VaspChargeDensity(filename=str(args.gt_chgcar)).atoms[-1]
    scf = json.loads(args.scf_json.read_text())
    nscf = json.loads(args.nscf_json.read_text())

    dv = abs(np.linalg.det(gt_cell)) / float(ml_density.size)
    n_gt = float(gt_density.sum() * dv)
    n_ml = float(ml_density.sum() * dv)
    ml_scaled = ml_density * (n_gt / n_ml) if n_ml != 0.0 else ml_density
    drho = ml_scaled - gt_density

    sf = np.array(scf["forces"])
    nf = np.array(nscf["forces"])
    if not (len(atoms) == len(sf) == len(nf)):
        raise ValueError(f"Atom-count mismatch: atoms={len(atoms)} scf={len(sf)} nscf={len(nf)}")

    e_dev = abs(nscf["energy_fr"] - scf["energy_fr"]) / scf["n_atoms"] * 1000.0
    f_dev_mean = float(np.mean(np.linalg.norm(nf - sf, axis=1)))
    struct = {
        "n_atoms": int(len(atoms)),
        "energy_dev_meV_per_atom": e_dev,
        "force_dev_mean_eVA": f_dev_mean,
        "EH_drho_eV": hartree_self_energy(drho, gt_cell),
        "L2_drho": float(np.sqrt((drho ** 2).sum() * dv)),
        "L1_drho_e": float(np.abs(drho).sum() * dv),
        "net_drho_e": float(drho.sum() * dv),
    }
    print("Structure-level descriptors:")
    for k, v in struct.items():
        print(f"  {k:26s} {v:.6f}" if isinstance(v, float) else f"  {k:26s} {v}")

    scaled = (atoms.get_positions() @ np.linalg.inv(gt_cell)) % 1.0
    symbols = atoms.get_chemical_symbols()
    dF = nf - sf

    atom_rows = []
    for i in range(len(atoms)):
        d = atom_sphere_descriptors(drho, gt_cell, scaled[i], args.r_c, args.r_min)
        dfi = dF[i]
        atom_rows.append({
            "atom_index": i, "symbol": symbols[i],
            "dF_mag": float(np.linalg.norm(dfi)),
            "q_sphere": d["q"], "absq_sphere": d["abs_q"],
            "dip_mag": float(np.linalg.norm(d["dip"])),
            "cos_dip_dF": _cos(d["dip"], dfi),
            "hf_mag": float(np.linalg.norm(d["hf"])),
            "cos_hf_dF": _cos(d["hf"], dfi),
        })

    print("\nPer-atom descriptors:")
    for row in atom_rows:
        print(f"  {row['atom_index']:3d} {row['symbol']:2s}  dF={row['dF_mag']:.4f}"
              f"  hf={row['hf_mag']:.4f}  cos(hf,dF)={row['cos_hf_dF']:.3f}")

    if args.out_json:
        args.out_json.write_text(json.dumps({"structure": struct, "atoms": atom_rows}, indent=2))
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
