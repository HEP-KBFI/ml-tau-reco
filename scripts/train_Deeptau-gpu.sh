#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N_wconv.out

IMG=/home/software/singularity/pytorch.simg
cd ~/mltaureco_paper/ml-tau-reco

#pytorch training
#input data files should be in /scratch-persistent or /home, NOT in /local (too slow)
singularity exec -B /scratch/persistent -B /local --nv $IMG \
    python3 src/deeptauTraining.py
    #python3 src/endtoend_simple.py
