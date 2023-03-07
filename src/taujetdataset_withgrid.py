import sys
import awkward as ak
import torch
from torch_geometric.data import Data
import os.path as osp
from glob import glob
import json
import numpy as np
import time

from build_grid import GridBuilder

from part_var import Var


class TauJetDatasetWithGrid:
    def __init__(self, filelist=[], cfgFileName="./config/grid_builder.json"):
        self.tau_p4_features = ["x", "y", "z", "tau"]
        self.filelist = filelist
        self.all_data = []
        cfgFile = open(cfgFileName, "r")
        cfg = json.load(cfgFile)
        self._builderConfig = cfg["GridAlgo"]
        self.tauftrs = cfg["tau_features"]
        cfgFile.close()
        self.buildGrid = GridBuilder(verbosity=False)
        self.len_part_features = Var.max_value()
        self.num_particles_in_grid = cfg["num_particles_in_grid"]

        for fn in self.filelist:
            print(f'Processing file: {fn} at ', time.strftime('%H:%M'))
            data = ak.from_parquet(fn)
            self.all_data += self.process_file_data(data)

    def process_file_data(self, data):
        if 'inner_grid' not in data.fields:
            data = self.buildGrid.processJets(data)
        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = data["gen_jet_tau_p4s"]
        gen_tau_p4 = torch.tensor(np.stack([p4.x, p4.y, p4.z, p4.tau], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]
        tau_features = self.get_tau_features(data)
        part_block_features = self.get_part_block_features(data)
        ret_data = [
            Data(
                gen_tau_decaymode=gen_tau_decaymode[itau : itau + 1],
                gen_tau_p4=gen_tau_p4[itau : itau + 1],
                tau_features=tau_features[itau : itau + 1, :],
                inner_grid=part_block_features["inner_grid"][itau : itau + 1, :],
                outer_grid=part_block_features["outer_grid"][itau : itau + 1, :],
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
        for feat in self.tau_p4_features:
            tau_feature_tensors.append(torch.tensor(taus[feat], dtype=torch.float32))
        for feat in self.tauftrs:
            tau_feature_tensors.append(torch.tensor(data[feat], dtype=torch.float32))
        tau_features = torch.stack(tau_feature_tensors, axis=-1)
        self.data_len = len(tau_features)

        return tau_features.to(dtype=torch.float32)

    def get_part_block_features(self, data: ak.Record) -> dict:
        part_block_frs = {}
        for cone in ["inner_grid", "outer_grid"]:
            block_name = f"{cone}"
            part_block_features = torch.tensor(data[block_name].to_numpy(), dtype=torch.float32)
            part_block = part_block_features.reshape(
                self.data_len, self.len_part_features*self.num_particles_in_grid,
                self._builderConfig[cone]["n_cells"], self._builderConfig[cone]["n_cells"]
            )
            part_block_frs[cone] = part_block
        return part_block_frs

    def get_concatenated_part_block_features(self, first_part_blocks: dict, second_part_blocks: dict) -> dict:
        concatenated_part_frs = {}
        for block in ["inner_grid", "outer_grid"]:
            first_block_feature = first_part_blocks[block]
            second_block_feature = second_part_blocks[block]
            concatenated_part_frs[f"{block}"] = torch.concatenate((first_block_feature, second_block_feature), axis=1)
        return concatenated_part_frs

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
        print("shape of inner grid ", batch["inner_grid"].shape)
        print("shape of outer grid ", batch["outer_grid"].shape)
        break
