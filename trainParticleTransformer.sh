#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

cd /home/joosep/tau/ml-tau-reco

# PyTorch training with the full shuffled dataset
uv run python3 src/trainParticleTransformer.py num_dataloader_workers=0 max_num_files=-1
