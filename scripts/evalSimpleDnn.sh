#!/bin/bash

#change this for a new model
export MODELPATH=/home/joosep/ml-tau-reco/outputs/2024-03-28/16-23-26
export NUM_FILES=500

#don't change these
export DATAPATH=/scratch/persistent/joosep/ml-tau/20240328_hepmc_genjets
export TAU_MODEL_FILE=$MODELPATH/model_best.pt
export OUTPATH=$MODELPATH/evaluation

./scripts/run-env.sh python3 src/runBuilder.py \
    hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
    n_files=$NUM_FILES samples_to_process=[ZH_Htautau] \
    samples.ZH_Htautau.output_dir=$DATAPATH/ZH_Htautau/ \
    output_dir=$OUTPATH \
    builder=SimpleDNN \
    use_multiprocessing=False

./scripts/run-env.sh python3 src/runBuilder.py \
    hydra.run.dir=. hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
    n_files=$NUM_FILES samples_to_process=[QCD] \
    samples.QCD.output_dir=$DATAPATH/QCD/ \
    output_dir=$OUTPATH \
    builder=SimpleDNN \
    use_multiprocessing=False
