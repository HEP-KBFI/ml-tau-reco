#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

cd /home/joosep/tau/ml-tau-reco
uv run python3 src/trainParticleTransformer.py \
    max_num_files=-1 \
    num_dataloader_workers=4 \
    cache_size=4
