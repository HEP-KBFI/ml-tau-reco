#!/bin/bash
set -e
set -x

mkdir -p ntuple
mkdir -p data
cd ntuple

#Download test files if they don't exist
INFILE_TAU_DIR=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380
if [ ! -d "$INFILE_TAU_DIR" ]; then
    wget --directory-prefix ZH_Htautau -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_ZH_Htautau_ecm380_1.root
    INFILE_TAU_DIR=$PWD/ZH_Htautau
else
    INFILE_TAU_DIR=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380
fi;

INFILE_QCD_DIR=/local/joosep/clic_edm4hep/p8_ee_qcd_ecm380
if [ ! -d "$INFILE_QCD_DIR" ]; then
    wget --directory-prefix QCD -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_qcd_ecm380_1.root
    INFILE_QCD_DIR=$PWD/QCD
else
    INFILE_QCD_DIR=/local/joosep/clic_edm4hep/p8_ee_qcd_ecm380
fi;


#process EDM4HEP to training ntuple in .parquet format
python3 ../src/edm4hep_to_ntuple.py samples_to_process=[ZH_Htautau] samples.ZH_Htautau.input_dir=$INFILE_TAU_DIR samples.ZH_Htautau.output_dir=$PWD test_run=True
python3 ../src/edm4hep_to_ntuple.py samples_to_process=[QCD] samples.QCD.input_dir=$INFILE_QCD_DIR samples.QCD.output_dir=$PWD test_run=True

find . -type f -name "*.parquet"

TAU_FILENAME=reco_p8_ee_ZH_Htautau_ecm380_*.parquet
QCD_FILENAME=reco_p8_ee_qcd_ecm380_*.parquet

TAU_FILES=( $TAU_FILENAME )
python3 ../src/test_ntuple_shape.py -f "$TAU_FILES"
QCD_FILES=( $QCD_FILENAME )
python3 ../src/test_ntuple_shape.py -f "$QCD_FILES"

cd ..
ls

#Load the dataset in pytorch
python3 src/taujetdataset.py ./ntuple/

cat <<EOF > train.yaml
train:
  paths:
  - ./ntuple/reco_p8_ee_ZH_Htautau_ecm380_1.parquet
EOF

cat <<EOF > val.yaml
train:
  paths:
  - ./ntuple/reco_p8_ee_qcd_ecm380_1.parquet
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

#run simple DNN reco
mkdir -p simplednn
python3 src/runBuilder.py n_files=1 samples_to_process=[ZH_Htautau] samples.ZH_Htautau.output_dir=ntuple builder=SimpleDNN use_multiprocessing=False output_dir=simplednn

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
