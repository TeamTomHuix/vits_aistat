#!/bin/bash
#SBATCH --job-name=RSAC              
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                   
#SBATCH --time=05:00:00             
#SBATCH --output=lmcts.out
#SBATCH --error=error_lmcts.out     
#SBATCH --account=iks@v100
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-15
##SBATCH -C a100

export TMP=$SCRATCH
export PATH="/linkhome/rech/genuey01/urb47su/.local/bin:$PATH"

nvidia-smi

srun python3 run_linear.py -m eta=10,100,300,1000 lbd=1,10,0.1, vits.step_size=0.01, 0.001, 0.005 agent_name='ts' key.agent=${SLURM_ARRAY_TASK_ID}         
#srun python3 run_linear.py -m eta=1, 10, 100, 300 lbd=1, 10, 0.1 vits.step_size=0.01, 0.001, 0.005 agent_name='vits' key.agent=${SLURM_ARRAY_TASK_ID}   
#srun python3 run_linear.py -m eta=1, 10, 100, 300 lbd=1, 10, 0.1 lmcts.step_size=0.01, 0.001, 0.005 agent_name='lmcts' key.agent=${SLURM_ARRAY_TASK_ID}                                                          
