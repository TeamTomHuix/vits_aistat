#!/bin/bash

#SBATCH --job-name=vts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=output.txt
#SBATCH --error=error.txt


module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate myenv
srun python3 main.py  --param_file ./parameters/test_linear/rvits.json
