#!/bin/bash
singularity exec -B /local/ -B /scratch-persistent /home/software/singularity/pytorch.simg:2023-01-25 "$@"