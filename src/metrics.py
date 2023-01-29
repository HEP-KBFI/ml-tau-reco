"""
https://github.com/HEP-KBFI/ml-tau-reco/issues/10

src/metrics.py  \
  --model outputs/hps/signal.parquet:outputs/hps/bkg.parquet:HPS \
  --model outputs/hps_deeptau/signal.parquet:outputs/hps_deeptau/bkg.parquet:HPS-DeepTau \
  ...
"""
# import os
# import glob
# import sys
# import uproot
# import awkward as ak
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
from general import load_all_data


def plot_eff_fake(sig_data, bkg_data):
    pass


def plot_roc(sig_data, bkg_data):
    pass


def plot_response(sig_data, bkg_data):
    pass


def plot_decaymode_reconstruction(sig_data, bkg_data):
    pass


def plot_all_metrics(input_sig_dir, input_bkg_dir, output_dir):
    sig_data = load_all_data(input_sig_dir)
    bkg_data = load_all_data(input_bkg_dir)
    plot_eff_fake(sig_data, bkg_data)
    plot_roc(sig_data, bkg_data)
    plot_response(sig_data, bkg_data)
    plot_decaymode_reconstruction(sig_data, bkg_data)


# if __name__ == '__main__':
#     input_bkg_dir = sys.argv[1]
#     input_sig_dir = sys.argv[2]
#     output_dir = sys.argv[3]
#     plot_all_metrics(input_sig_dir, input_bkg_dir, output_dir)
