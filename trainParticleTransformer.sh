#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/ml-tau-reco

# PyTorch training
singularity exec -B /scratch-persistent --nv $IMG \
  python3 src/trainParticleTransformer.py
