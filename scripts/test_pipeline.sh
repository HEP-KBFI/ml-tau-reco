#!/bin/bash

cd src

wget https://jpata.web.cern.ch/jpata/mlpf/clic_edm4hep/reco_p8_ee_tt_ecm365_1.root
python3 test.py reco_p8_ee_tt_ecm365_1.root
