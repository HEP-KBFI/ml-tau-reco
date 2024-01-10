#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=15G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2023-03-16
cd ~/ml-tau-reco

# PyTorch training
singularity exec -B /scratch/persistent --nv $IMG \
  python3 src/endtoend_simple.py
