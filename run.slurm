#!/bin/bash
#SBATCH --job-name=reb
#SBATCH --nodes=1       
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=11
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -A nek@v100
#SBATCH --array=0-0
module load python/3.10.4
export PATH="/linkhome/rech/genuey01/ujz54ib/.local/bin:$PATH"
export TMP=$SCRATCH
nvidia-smi

srun python3 main.py  --param_file ./parameters/linear/ts.json                                                                                 
