#!/bin/bash
set -e
set -x

mkdir -p ntuple
mkdir -p data
cd ntuple

#Download test files if they don't exist

INFILE_TAU_DIR=/local/joosep/clic_edm4hep_2023_02_27/p8_ee_ZH_Htautau_ecm380
if [ ! -d "$INFILE_TAU_DIR" ]; then
    wget --directory-prefix ZH_Htautau -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/clic_edm4hep_2023_02_27/p8_ee_ZH_Htautau_ecm380/reco_p8_ee_ZH_Htautau_ecm380_200001.root
    INFILE_TAU_DIR=$PWD/ZH_Htautau
else
    mkdir -p root_input/p8_ee_ZH_Htautau_ecm380
    cp /local/joosep/clic_edm4hep_2023_02_27/p8_ee_ZH_Htautau_ecm380/reco_p8_ee_ZH_Htautau_ecm380_200001.root root_input/p8_ee_ZH_Htautau_ecm380/
    INFILE_TAU_DIR=$PWD/root_input/p8_ee_ZH_Htautau_ecm380
fi;

INFILE_QCD_DIR=/local/joosep/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380
if [ ! -d "$INFILE_QCD_DIR" ]; then
    wget --directory-prefix QCD -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100001.root
    INFILE_QCD_DIR=$PWD/QCD
else
    mkdir -p root_input/p8_ee_qq_ecm380
    cp /local/joosep/clic_edm4hep_2023_02_27/p8_ee_qq_ecm380/reco_p8_ee_qq_ecm380_100001.root root_input/p8_ee_qq_ecm380/
    INFILE_QCD_DIR=$PWD/root_input/p8_ee_qq_ecm380
fi;


#process EDM4HEP to training ntuple in .parquet format
python3 ../src/edm4hep_to_ntuple.py samples_to_process=[ZH_Htautau] samples.ZH_Htautau.input_dir=$INFILE_TAU_DIR samples.ZH_Htautau.output_dir=$PWD test_run=True
python3 ../src/edm4hep_to_ntuple.py samples_to_process=[QCD] samples.QCD.input_dir=$INFILE_QCD_DIR samples.QCD.output_dir=$PWD test_run=True

python3 ../src/weight_tools.py samples.ZH_Htautau.output_dir=$PWD samples.QCD.output_dir=$PWD  # Currently the weights are calculated 2 times here since the two samples are in the same directory

find . -type f -name "*.parquet"

TAU_FILENAME=reco_p8_ee_ZH_Htautau_ecm380_200001.parquet
QCD_FILENAME=reco_p8_ee_qq_ecm380_100001.parquet

TAU_FILES=( $TAU_FILENAME )
python3 ../src/test_ntuple_shape.py -f "$TAU_FILES"
QCD_FILES=( $QCD_FILENAME )
python3 ../src/test_ntuple_shape.py -f "$QCD_FILES"

cd ..
ls

#Load the dataset in pytorch
python3 src/taujetdataset.py ./ntuple/

#Prepare training inputs with just one file per split
cat <<EOF > train.yaml
train:
  paths:
  - ./ntuple/reco_p8_ee_ZH_Htautau_ecm380_200001.parquet
EOF

cat <<EOF > val.yaml
validation:
  paths:
  - ./ntuple/reco_p8_ee_qq_ecm380_100001.parquet
EOF

#Train a simple pytorch model
python3 src/endtoend_simple.py epochs=2 train_files=train.yaml validation_files=val.yaml

# run oracle -> oracle.parquet
mkdir -p oracle
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=Oracle use_multiprocessing=False output_dir=oracle

# run fastCMSTauBuilder
mkdir -p fastCMSTaus
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=FastCMSTau use_multiprocessing=False output_dir=fastCMSTaus

#run HPS -> hps.parquet
mkdir -p hps
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=HPS use_multiprocessing=False output_dir=hps

#run GridBuilder
mkdir -p grid
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] builder=Grid use_multiprocessing=False output_dir=grid samples.ZH_Htautau.output_dir=hps/HPS/ZH_Htautau/

#run simple DNN reco
mkdir -p simplednn
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=SimpleDNN use_multiprocessing=False output_dir=simplednn

#run LorentzNet algo
mkdir -p LorentzNet
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=LorentzNet use_multiprocessing=False output_dir=LorentzNet

#run ParticleTransformer algo
mkdir -p ParticleTransformer
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=ParticleTransformer use_multiprocessing=False output_dir=ParticleTransformer

#list all files
find . -type f -name "*.parquet"

#run HPS + DeepTau -> hps_deeptau.parquet
#python3 reco_hps_deeptau.py

#run ML reco + DeepTau -> mlreco_deeptau.parquet
#python3 reco_ml_deeptau.py

#run end-to-end ML reco+id -> endtoend_ml.parquet
#python3 reco_endtoend_ml.py

#run metrics script
#python3 metrics.py hps.parquet hps_deeptau.parquet mlreco_deeptau.parquet endtoend_ml.parquet
