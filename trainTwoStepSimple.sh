#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg
cd /home/tolange/CLIC_paper/ml-tau-reco

# PyTorch training
singularity exec -B /scratch-persistent --nv $IMG \
  python3 src/tauTwoStepSimple.py runName=gnn_complete_addNN outpath=/home/tolange/CLIC_paper/ml-tau-reco/twostep_gpu/gnn_complete_addNN extras="addNN"
