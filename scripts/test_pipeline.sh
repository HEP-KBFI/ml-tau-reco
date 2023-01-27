#!/bin/bash

set -e
set -x

cd src
INFILE_TAU=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380/reco_p8_ee_ZH_Htautau_ecm380_1.root
if [ ! -f "$INFILE_TAU" ]; then
    find .
    wget --directory-prefix ZH_Htautau -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_1.root
    INFILE_TAU=$PWD/ZH_Htautau
    du -ach INFILE_TAU
else
    INFILE_TAU=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380/
fi;

INFILE_QCD=/local/joosep/clic_edm4hep/p8_ee_qcd_ecm380/reco_p8_ee_qcd_ecm380_1.root
if [ ! -f "$INFILE_QCD" ]; then
    find .
    wget --directory-prefix QCD -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_qcd_ecm380_1.root
    INFILE_QCD=$PWD/QCD
    du -ach INFILE_QCD
else
    INFILE_QCD=/local/joosep/clic_edm4hep/p8_ee_ZH_Htautau_ecm380/
fi;

#process EDM4HEP to training ntuple in .parquet format
find .
python3 edm4hep_to_ntuple.py $INFILE_TAU $PWD test
find .
python3 edm4hep_to_ntuple.py $INFILE_QCD $PWD test

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
