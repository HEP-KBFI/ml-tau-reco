#!/bin/bash

cd src

wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_1.root

#process EDM4HEP to training ntuple in .parquet format
python3 edm4hep_to_ntuple.py reco_p8_ee_tt_ecm365_1.root reco_p8_ee_tt_ecm365_1.parquet

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
