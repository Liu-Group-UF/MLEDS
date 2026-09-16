#!/bin/bash
#SBATCH --job-name=cuxo_bulk_dft
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --distribution=cyclic:cyclic
# Set --account/--qos/--partition for your cluster.

# Single-structure ground-truth SCF calculation for a CuxO bulk structure.
# Expects POSCAR (and INCAR/KPOINTS, e.g. copied from configs/dft/) in the
# working directory. Set POTCAR_DIR before submitting, or generate POTCAR
# ahead of time with scripts/dft/generate_potcar.sh.

module purge
module load vasp   # adjust module name/path for your cluster
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=1
export SLURM_MPI_TYPE=pmix_v5

if [ ! -f "POTCAR" ]; then
    echo "Generating POTCAR..."
    ../../scripts/dft/generate_potcar.sh Cu O
fi

echo "Running VASP job in $(pwd)"
srun vasp_std > vasp.log
