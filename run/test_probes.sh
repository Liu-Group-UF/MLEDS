#!/bin/bash
#SBATCH --job-name=test_cuxo_bulk
#SBATCH --output=slurmoutputs/R-%x.%j.out
#SBATCH --error=slurmoutputs/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24gb
#SBATCH --gpus=1
#SBATCH --time=00:30:00
# Set --account/--qos/--partition for your cluster.

module load conda
conda activate charge3net
pwd; hostname; date

python src/test_from_config.py \
    -cd configs \
    -cn test_probes.yaml

date
