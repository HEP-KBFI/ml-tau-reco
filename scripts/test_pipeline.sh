#!/bin/bash

cd src

wget https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_1.root

#process EDM4HEP to training ntuple in .parquet format
python3 edm4hep_to_ntuple.py reco_p8_ee_tt_ecm365_1.root reco_p8_ee_tt_ecm365_1.parquet
