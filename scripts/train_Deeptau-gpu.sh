#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu=10G
#SBATCH -o slurm-conv_fl2_tanhlifetime.out

IMG=/home/software/singularity/pytorch.simg
cd $1
echo $(pwd)

#pytorch training
#input data files should be in /scratch-persistent or /home, NOT in /local (too slow)
singularity exec -B /scratch/persistent -B /local --nv $IMG \
    python3 src/deeptauTraining.py
    #python3 src/endtoend_simple.py
