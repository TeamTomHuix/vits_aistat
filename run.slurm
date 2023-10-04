#!/bin/bash
#SBATCH --job-name=reb
##SBATCH --nodes=1       
#SBATCH --partition=prepost
#SBATCH --time=02:00:00
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH -A nek@v100
#SBATCH --array=0-19

module load python/3.10.4
export PATH="/linkhome/rech/genuey01/ujz54ib/.local/bin:$PATH"
export TMP=$SCRATCH
nvidia-smi

srun python3 run_linear.py -m eta=1, 10, 100, 300 lbd=1, 10, 0.1 agent_name='ts' key.agent=${SLURM_ARRAY_TASK_ID}         
#srun python3 run_linear.py -m eta=1, 10, 100, 300 lbd=1, 10, 0.1 vits.step_size=0.01, 0.001, 0.005 agent_name='vits' key.agent=${SLURM_ARRAY_TASK_ID}   
#srun python3 run_linear.py -m eta=1, 10, 100, 300 lbd=1, 10, 0.1 lmcts.step_size=0.01, 0.001, 0.005 agent_name='lmcts' key.agent=${SLURM_ARRAY_TASK_ID}                                                          
