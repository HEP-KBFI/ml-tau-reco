#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd ~/ml-tau-reco

#pytorch training
#input data files should be in /scratch-persistent or /home, NOT in /local (too slow)
singularity exec -B /scratch-persistent --nv $IMG \
  python3 src/endtoend_simple.py
