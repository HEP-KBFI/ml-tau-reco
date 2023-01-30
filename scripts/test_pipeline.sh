#!/bin/bash


cd src
INFILE_TAU=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380/reco_p8_ee_ZH_Htautau_ecm380_1.root
if [ ! -f "$INFILE_TAU" ]; then
    wget --directory-prefix ZH_Htautau -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_ZH_Htautau_ecm380_1.root
    INFILE_TAU_DIR=$PWD/ZH_Htautau
    TAU_FILENAME=reco_p8_ee_tt_ecm365_*.parquet
else
    INFILE_TAU_DIR=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380
    TAU_FILENAME=reco_p8_ee_ZH_Htautau_ecm380_*.parquet
fi;

INFILE_QCD=/local/joosep/clic_edm4hep/p8_ee_qcd_ecm380/reco_p8_ee_qcd_ecm380_1.root
if [ ! -f "$INFILE_QCD" ]; then
    wget --directory-prefix QCD -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_qcd_ecm380_1.root
    INFILE_QCD_DIR=$PWD/QCD
    QCD_FILENAME=reco_p8_ee_qcd_ecm380_*.parquet
else
    INFILE_QCD_DIR=/local/joosep/clic_edm4hep/p8_ee_qcd_ecm380
    QCD_FILENAME=reco_p8_ee_qcd_ecm380_*.parquet
fi;

#process EDM4HEP to training ntuple in .parquet format
python3 edm4hep_to_ntuple.py input_dir=$INFILE_TAU_DIR output_dir=$PWD test_run=True
python3 edm4hep_to_ntuple.py input_dir=$INFILE_QCD_DIR output_dir=$PWD test_run=True
TAU_FILES=( $TAU_FILENAME )
python3 test_ntuple_shape.py -f "$TAU_FILES"
QCD_FILES=( $QCD_FILENAME )
python3 test_ntuple_shape.py -f $QCD_FILES

#Load generated dataset in pytorch
python3 taujetdataset.py

#run HPS -> hps.parquet
#python3 reco_hps.py

#run HPS + DeepTau -> hps_deeptau.parquet
#python3 reco_hps_deeptau.py

#run ML reco + DeepTau -> mlreco_deeptau.parquet
#python3 reco_ml_deeptau.py

#run end-to-end ML reco+id -> endtoend_ml.parquet
#python3 reco_endtoend_ml.py

#run metrics script
#python3 metrics.py hps.parquet hps_deeptau.parquet mlreco_deeptau.parquet endtoend_ml.parquet
