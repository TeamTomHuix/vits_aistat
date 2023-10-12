#!/bin/bash
#SBATCH --job-name=lens
##SBATCH --nodes=1       
#SBATCH --time=02:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --gres=gpu:1
#SBATCH -A nek@v100
#SBATCH -C v100
#SBATCH --array=0-19

module load python/3.10.4
export PATH="/linkhome/rech/genuey01/ujz54ib/.local/bin:$PATH"
export TMP=$SCRATCH
nvidia-smi

srun python3 run.py --seed=${SLURM_ARRAY_TASK_ID}         
