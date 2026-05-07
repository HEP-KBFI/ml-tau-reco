#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

cd /home/joosep/tau/ml-tau-reco

# PyTorch training
uv run  python3 src/trainParticleTransformer.py
