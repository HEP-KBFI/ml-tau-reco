import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob
import json

from part_var import *

class TauJetDatasetWithGrid():
    def __init__(self, filelist=[], cfgFileName="./config/grid_builder.json"):
        self.tau_features = ["x", "y", "z", "tau"]
        self.filelist = filelist
        self.all_data = []

        cfgFile = open(cfgFileName, "r")
        cfg = json.load(cfgFile)
        self._builderConfig = cfg["GridAlgo"]
        cfgFile.close()

        for fn in self.filelist:
            data = ak.from_parquet(fn)
            self.all_data += self.process_file_data(data)

    def process_file_data(self, data):
        
        tau_features = self.get_tau_features(data)
        ele_block_features = self.get_part_block_features(data, 'ele')
        gamma_block_features = self.get_part_block_features(data, 'gamma')
        ele_gamma_features = self.get_ele_gamma_block_features(ele_block_features, gamma_block_features)
        muon_block_features = self.get_part_block_features(data, 'mu')
        charged_cand_block_features = self.get_part_block_features(data, 'charged_cand')
        neutral_cand_block_features = self.get_part_block_features(data, 'neutral_cand')
        ret_data = [
            Data(
                tau_features = tau_features[itau : itau+1, :],
                inner_grid_ele_gamma_block = ele_gamma_features['inner_grid'][itau : itau+1, :],
                outer_grid_ele_gamma_block = ele_gamma_features['outer_grid'][itau : itau+1, :],
                inner_grid_mu_block = muon_block_features['inner_grid'][itau : itau+1, :],
                outer_grid_mu_block = muon_block_features['outer_grid'][itau : itau+1, :],
                inner_grid_charged_cand_block = muon_block_features['inner_grid'][itau : itau+1, :],
                outer_grid_charged_cand_block = muon_block_features['outer_grid'][itau : itau+1, :],
                inner_grid_neutral_cand_block = muon_block_features['inner_grid'][itau : itau+1, :],
                outer_grid_neutral_cand_block = muon_block_features['outer_grid'][itau : itau+1, :],
            )
            for itau in range(len(tau_features))
        ]

        return ret_data

    def get_tau_features(self, data: ak.Record) -> torch.Tensor:
        taus = {}
        for k in data["tau_p4s"].fields:
            taus[k] = data["tau_p4s"][k]
        # collect tau features in a specific order to an (Njet x Nfeatjet) torch tensor 
        tau_feature_tensors = []
        for feat in self.tau_features:
            tau_feature_tensors.append(torch.tensor(taus[feat], dtype=torch.float32))
        tau_features = torch.stack(tau_feature_tensors, axis=-1)
        self.data_len = len(tau_features)

        return tau_features.to(dtype=torch.float32)

    def get_part_block_features(self, data: ak.Record, parttype: str) -> dict:
        part_block_frs = {}
        for cone in ["inner_grid", "outer_grid"]:
            block_name = f'{cone}_{parttype}_block'
            len_part_features = len(part_var_list[parttype])
            part_block_features = torch.tensor(data[block_name], dtype=torch.float32)
            part_block = part_block_features.reshape(self.data_len, self._builderConfig[cone]["n_cells"], self._builderConfig[cone]["n_cells"], len_part_features)
            part_block_frs[cone] = part_block
        return part_block_frs

    def get_ele_gamma_block_features(self, eleblocks: dict, gammablocks: dict) -> dict:
        ele_gamma_frs = {}
        for block in ['inner_grid', 'outer_grid']:
            ele_feature = eleblocks[block]
            gamma_feature = gammablocks[block]
            ele_gamma_frs[f'{block}'] = torch.concatenate((ele_feature, gamma_feature), axis=-1)
        return ele_gamma_frs

    def __getitem__(self, idx):
        return self.all_data[idx]

    def __len__(self):
        return len(self.all_data)

if __name__ == "__main__":
    filelist = list(glob(osp.join(sys.argv[1], "*.parquet")))
    ds = TauJetDatasetWithGrid(filelist)
    print("Loaded TauJetDataset with {} files".format(len(ds)))

    # treat each input file like a batch
    for ibatch in range(len(ds)):
        batch = ds[ibatch]
    print('shape of ele_gamma inner grid ', batch['inner_grid_ele_gamma_block'].shape)
    print('shape of ele_gamma inner grid ', batch['outer_grid_ele_gamma_block'].shape)
    print('shape of muon inner grid ', batch['inner_grid_mu_block'].shape)
    print('shape of muon inner grid ', batch['outer_grid_mu_block'].shape)
    print('shape of charged cand inner grid ', batch['inner_grid_charged_cand_block'].shape)
    print('shape of charged cand inner grid ', batch['outer_grid_charged_cand_block'].shape)
    print('shape of neutral cand inner grid ', batch['inner_grid_neutral_cand_block'].shape)
    print('shape of neutral cand inner grid ', batch['outer_grid_neutral_cand_block'].shape)
