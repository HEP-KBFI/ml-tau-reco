#!/bin/bash

./scripts/run-env.sh python3 src/runBuilder.py n_files=100 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=/scratch/persistent/joosep/ml-tau/2023/ZH_Htautau/ builder=SimpleDNN use_multiprocessing=False output_dir=simplednn
./scripts/run-env.sh python3 src/runBuilder.py n_files=100 samples_to_process=[QCD] samples.QCD.output_dir=/scratch/persistent/joosep/ml-tau/2023/QCD/ builder=SimpleDNN use_multiprocessing=False output_dir=simplednn


