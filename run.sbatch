#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -A zhangmlgroup
#SBATCH --mem=64G
#SBATCH -p gpu

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-02:00:00

# 设置环境
module load miniforge
source activate ProtoLens
chmod +x /scratch/nkw3mr/intepre_clinical_long_doc/ProtoLens/experiment.py

python experiment.py

conda deactivate