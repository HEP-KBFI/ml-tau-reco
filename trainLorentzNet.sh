#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=18G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/ml-tau-reco

# PyTorch training
singularity exec -B /scratch/persistent --nv $IMG \
  python3 src/trainLorentzNet.py
