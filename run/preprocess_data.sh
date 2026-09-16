#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --mem-per-cpu=4gb

module load conda
conda activate charge3net
pwd; hostname; date

# Convert raw CHGCARs under ./data/mp_raw into pickled (density, atoms) pairs
# under ./data/mp, and build the filelist used by the data module.
python scripts/batch_convert_to_npy.py --raw_data_dir ./data/mp_raw --pkl_data_dir ./data/mp/ --my_task_id 0 --num_tasks 1 --num_cores 24 --s False

python scripts/write_mp_probe_count_file.py --filelist ./data/mp/filelist.txt --workers 24
