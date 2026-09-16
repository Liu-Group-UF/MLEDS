# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import argparse
import pandas as pd
from pathlib import Path
from src.utils.data import _load_pickled_atoms, load_numpy_density
from multiprocessing import Pool
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data/mp", help="path to the directory containing filelist.txt and pickled atoms/density files")
parser.add_argument("--n_test", type=int, default=70, help="number of structures to hold out for the test split")
parser.add_argument("--n_val", type=int, default=80, help="number of structures to hold out for the validation split")

def get_atoms(member, mp_dataset_path):
    return member, _load_pickled_atoms(mp_dataset_path, member)

def _get_atoms_stats_multi(mpid_atoms):
    mpid, atoms = mpid_atoms
    return mpid, get_atoms_stats(atoms)


def get_atoms_stats(atoms_obj):
    # get info from atoms_obj
    chemical_formula = atoms_obj.get_chemical_formula()

    # get info from structure
    struc = AseAtomsAdaptor.get_structure(atoms_obj)

    # get info from composition
    composition = struc.composition
    num_atoms = composition.num_atoms
    chemical_system = composition.chemical_system

    # get info from reduced composition
    reduced_composition, repeats = composition.get_reduced_composition_and_factor()
    reduced_formula, repeats = composition.get_reduced_formula_and_factor()
    num_atoms_reduced = reduced_composition.num_atoms

    # get info from space group analyzer
    sga = SpacegroupAnalyzer(struc)
    sgn = sga.get_space_group_number()
    sgs = sga.get_space_group_symbol()

    return {
            "chemical_formula":chemical_formula,
            "reduced_chemical_formula": reduced_formula,
            "chemical_system": chemical_system,
            "num_atoms": num_atoms,
            "reduced_num_atoms": num_atoms_reduced,
            "factor": repeats,
            "space_group_number": sgn,
            "space_group_symbol": sgs,
            "atomic_weight":composition.weight
            }


def main(mp_dataset_path, n_test, n_val):
    # Get file names / mpids
    filename = mp_dataset_path / "filelist.txt"
    with open(filename, "r") as f:
        lines = f.readlines()
    member_list = [line.strip() for line in lines]

    # load the atoms objects using multiprocessing
    with Pool(20) as p:
        out = p.starmap(get_atoms, [(member, mp_dataset_path) for member in member_list])
    mpid_to_atoms = dict(out)

    # get statistics on each atoms object
    with Pool(20) as p:
        out = p.map(_get_atoms_stats_multi, list(mpid_to_atoms.items()))

    # Create output dataframe
    out_dicts = [dict(mpid=o[0], **o[1]) for o in out]
    df = pd.DataFrame(out_dicts)

    # Remove duplicates
    df_deduped = df.sort_values("mpid", ascending=False).sort_values("num_atoms").drop_duplicates(subset=["reduced_chemical_formula", "space_group_number"])

    # save to csv
    # row indices (df.loc) can be associated with the indices in the split files
    # to recover metadata for individual subsets
    df_deduped.sort_index().to_csv(mp_dataset_path / "material_metadata.csv")

    # stats on full dataset:
    print("Number of materials in directory: ", len(df))
    print("Number of de-duped materials: ", len(df_deduped))
    print("Number of distinct chemical formulae: ", len(df_deduped.chemical_formula.unique()))
    print("Number of distinct reduced formulae: ", len(df_deduped.reduced_chemical_formula.unique()))

    # shuffle
    df_deduped = df_deduped.sample(frac=1.0, random_state=42)  # shuffle

    # split dataset
    test_set = df_deduped.iloc[:n_test]
    train_val_sets = df_deduped.iloc[n_test:]
    val_set = train_val_sets.iloc[:n_val]
    train_set = train_val_sets.iloc[n_val:]

    # write json split file for entire dataset
    split_dict = dict(train=train_set.index.tolist(),
                      test=test_set.index.tolist(),
                      validation=val_set.index.tolist())

    with open(mp_dataset_path / "split.json", 'w') as f:
        json.dump(split_dict, f)


if __name__ == "__main__":
    args = parser.parse_args()
    main(Path(args.data_dir), args.n_test, args.n_val)
