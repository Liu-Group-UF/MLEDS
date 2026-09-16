import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from src.utils.data import compute_electron_count, load_atoms_file, load_density_file


@dataclass
class CheckResult:
    label: str
    density_file: Path
    atoms_file: Path
    target_electrons: float
    integrated_electrons: float
    abs_error: float
    passed: bool


def find_prediction_cubes(root: Path) -> list[Path]:
    return sorted(root.rglob("prediction_lmax_4/cubes/*.npy"))


def resolve_atoms_file(cube_file: Path) -> tuple[Path | None, str]:
    if cube_file.name == "CHGCAR.npy":
        structure_dir = cube_file.parents[2]
        atoms_file = structure_dir / "CHGCAR_atoms.pkl"
        return (atoms_file if atoms_file.exists() else None), structure_dir.name

    dataset_dir = cube_file.parents[2]
    atoms_file = dataset_dir / f"{cube_file.stem}_atoms.pkl"
    return (atoms_file if atoms_file.exists() else None), cube_file.stem


def check_cube(cube_file: Path, tolerance: float) -> CheckResult:
    atoms_file, label = resolve_atoms_file(cube_file)
    if atoms_file is None:
        raise FileNotFoundError(f"Could not find matching atoms pickle for {cube_file}")

    density = load_density_file(str(cube_file))
    atoms = load_atoms_file(str(atoms_file))
    target_electrons = atoms.info.get("charge3net_target_electrons")
    if target_electrons is None:
        raise KeyError(f"{atoms_file} is missing atoms.info['charge3net_target_electrons']")

    integrated_electrons = compute_electron_count(density, atoms.get_cell())
    abs_error = abs(integrated_electrons - float(target_electrons))
    return CheckResult(
        label=label,
        density_file=cube_file,
        atoms_file=atoms_file,
        target_electrons=float(target_electrons),
        integrated_electrons=integrated_electrons,
        abs_error=abs_error,
        passed=abs_error <= tolerance,
    )


def write_csv(results: list[CheckResult], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "density_file",
                "atoms_file",
                "target_electrons",
                "integrated_electrons",
                "abs_error",
                "passed",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "label": result.label,
                    "density_file": str(result.density_file),
                    "atoms_file": str(result.atoms_file),
                    "target_electrons": f"{result.target_electrons:.12f}",
                    "integrated_electrons": f"{result.integrated_electrons:.12f}",
                    "abs_error": f"{result.abs_error:.12e}",
                    "passed": result.passed,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that predicted density cubes integrate to the cached target electron count."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan for prediction_lmax_4/cubes/*.npy files.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Maximum allowed absolute error in integrated electron count.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-density results.",
    )
    parser.add_argument(
        "--max-failures-to-print",
        type=int,
        default=20,
        help="Maximum number of failing cases to print.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cube_files = find_prediction_cubes(args.root)
    if not cube_files:
        raise FileNotFoundError(f"No prediction cubes found under {args.root}")

    results = [check_cube(cube_file, args.tolerance) for cube_file in cube_files]
    failures = [result for result in results if not result.passed]

    print(f"Scanned {len(results)} predicted density cubes under {args.root}")
    print(f"Tolerance: {args.tolerance:.6g} electrons")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed: {len(failures)}")

    if results:
        max_abs_error = max(result.abs_error for result in results)
        mean_abs_error = sum(result.abs_error for result in results) / len(results)
        print(f"Mean absolute electron-count error: {mean_abs_error:.12e}")
        print(f"Max absolute electron-count error:  {max_abs_error:.12e}")

    if failures:
        print("Largest failures:")
        for result in sorted(failures, key=lambda item: item.abs_error, reverse=True)[: args.max_failures_to_print]:
            print(
                f"  {result.label}: integrated={result.integrated_electrons:.12f}, "
                f"target={result.target_electrons:.12f}, abs_error={result.abs_error:.12e}"
            )

    if args.output_csv is not None:
        write_csv(results, args.output_csv)
        print(f"Wrote per-density results to {args.output_csv}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
