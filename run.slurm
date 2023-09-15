#!/bin/bash
#SBATCH --job-name=reb
##SBATCH --nodes=1       
#SBATCH --partition=prepost
#SBATCH --time=02:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -A nek@v100
#SBATCH --array=0-0
module load python/3.10.4
export PATH="/linkhome/rech/genuey01/ujz54ib/.local/bin:$PATH"
export TMP=$SCRATCH
nvidia-smi

srun python3 run_linear.py -m eta=0.1,0.5,1,5 lbd=0.01,0.1,1 agent_name='ts'                                                                        
