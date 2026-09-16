<p align="center">
  <img src="assets/banner.svg" alt="MLEDS banner" width="900">
</p>

# MLEDS

**Assessing the Physical Fidelity of Machine-Learned Electron Densities for
Surface and Interfacial Systems**

Configs, data-conversion scripts, and driver-code patches used to train and
evaluate a charge3net-based model for predicting electron density on CuxO
(CuO / Cu2O) bulk structures, as reported in the manuscript above.

## Repository structure

```
mleds/
├── configs/                 # Hydra configs
│   ├── train.yaml
│   ├── test_probes.yaml     # NMAE on sampled probes
│   ├── test_chgcar.yaml     # full-grid density cube prediction
│   ├── model/e3_density.yaml
│   └── data/
│       ├── mp_data.yaml
│       └── graph_constructor/kdtree.yaml
├── run/                      # SLURM launch scripts
│   ├── preprocess_data.sh
│   ├── train.sh
│   └── test_probes.sh
├── scripts/                  # data conversion / split / QC
└── src_patch/                 # overlay onto a charge3net checkout's src/
```

## Dependency

This code is built on top of [charge3net](https://github.com/AIforGreatGood/charge3net)
(MIT License, Copyright (c) 2023 Massachusetts Institute of Technology). It is
not a redistribution of that package — only the parts specific to the CuxO
bulk workflow are included here. To use this repo:

1. Clone charge3net and follow its README to set up the `charge3net` conda
   environment.
2. Copy `src_patch/` on top of that checkout's `src/` directory (it patches
   `trainer.py`, `test.py`, `test_from_config.py`,
   `charge3net/data/dataset.py`, `utils/data.py`, and `utils/predictions.py`
   — see "What's patched" below).
3. Copy `configs/` and `scripts/` into the checkout root (or point the
   commands below at wherever you placed them).

## What's patched, and why

`src_patch/` carries a small number of changes on top of stock charge3net:

- `trainer.py`, `test.py`, `test_from_config.py`: support for
  `defer_cube_materialization` (skip combining per-shard prediction cubes
  until a later step, useful for very large grids) and a SLURM-aware
  distributed init port (`resolve_master_port`) so concurrent jobs on a
  shared cluster don't collide on the default port.
- `charge3net/data/dataset.py`: a `filelist.txt` data root is always treated
  as a pickled-density directory (`DensityPickleDir`), matching the data
  layout used here.
- `utils/data.py`, `utils/predictions.py`: electron-count bookkeeping
  (`compute_electron_count`, `get_density_normalization_metadata`,
  `normalize_density_to_electron_count`) used by the CHGCAR conversion
  scripts and by `scripts/check_predicted_density_normalization.py` to
  sanity-check that predicted density cubes integrate to the correct number
  of electrons.

Model architecture (`e3.py`, `densitymodel.py`) is unmodified from charge3net
and is not duplicated here — install it from the upstream repo.

Everything else in `scripts/` and `src_patch/` carries at least one real
change from upstream (a new flag, a bug fix, or new functionality this
workflow relies on) — the exception is `scripts/write_dummy_split.py`, which
is included unmodified purely because `convert_chgcar_dir_to_pkl_dir.py`
imports it directly and it's easier to keep the repo self-contained.

## Workflow

1. **Raw CHGCAR -> pickled density/atoms + filelist** (`run/preprocess_data.sh`,
   `scripts/batch_convert_to_npy.py`, `scripts/write_mp_probe_count_file.py`).
   For converting or reconverting a specific subset of structures (rather
   than an entire raw-data directory), use
   `scripts/convert_chgcar_dir_to_pkl_dir.py --input-list <path>` instead.
2. **Train/val/test split**: `scripts/write_mp_datasplits.py` writes
   `split.json` from `filelist.txt`, stratified by de-duplicated
   composition/space group. `configs/train.yaml` and the test
   configs expect the split at `./data/mp/split_v2.json`, so rename/copy the
   script's output accordingly (or pass a different `split:` value in the
   config). The split is index-based (positions into `filelist.txt`), so
   re-converting individual entries in place (step 1) does not require
   regenerating the split.
3. **Train**: `run/train.sh` (`configs/train.yaml`).
4. **Evaluate on sampled probes**: `run/test_probes.sh`
   (`configs/test_probes.yaml`), reports NMAE on 1000 probes/structure.
5. **Full-grid inference**: `configs/test_chgcar.yaml`, for
   producing a complete predicted density cube per structure (input
   directory needs its own `filelist.txt`, `split.json`, `probe_counts.csv`).
6. **Cube -> CHGCAR** (for DFT-comparison / visualization):
   `scripts/convert_pkl_to_chgcar.py`.
7. **QC**: `scripts/check_predicted_density_normalization.py` checks that
   predicted cubes integrate to the expected electron count within
   tolerance.

Set `--account`/`--qos`/`--partition` in the `run/*.sh` scripts for your
cluster before submitting.

## License

MIT — see [LICENSE](LICENSE). Carried over from charge3net; this repo's own
additions are released under the same terms.
