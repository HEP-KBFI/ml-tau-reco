import sys
import awkward as ak
import torch
from torch_geometric.data import Data
import os.path as osp
import os
from glob import glob
import json
import numpy as np
import time
import multiprocessing
from build_grid import GridBuilder

from part_var import Var
from auxiliary import *


class TauJetDatasetWithGrid:
    def __init__(self, processed_dir="", filelist=[], cfgFileName="./config/grid_builder.json"):
        self._processed_dir = processed_dir
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

    @property
    def raw_file_names(self):
        return self.filelist

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted(proc_list)

    @property
    def processed_dir(self):
        return self._processed_dir
    '''
    for fn in self.filelist:
    print(f"Processing file: {fn} at ", time.strftime("%H:%M"))
    data = ak.from_parquet(fn)
    self.all_data += self.process_file_data(data)
    '''
    def process_file_data(self, data):
        if "inner_grid" not in data.fields:
            data = self.buildGrid.processJets(data)
        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = data["gen_jet_tau_p4s"]
        gen_tau_p4 = torch.tensor(np.stack([p4.x, p4.y, p4.z, p4.tau], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]
        tau_features = self.get_tau_features(data)
        part_block_features = self.get_part_block_features(data)
        weights = torch.tensor(data["weight"]).to(dtype=torch.float32)
        ret_data = [
            Data(
                gen_tau_decaymode=gen_tau_decaymode[itau : itau + 1],
                gen_tau_p4=gen_tau_p4[itau : itau + 1],
                tau_features=tau_features[itau : itau + 1, :],
                inner_grid=part_block_features["inner_grid"][itau : itau + 1, :],
                outer_grid=part_block_features["outer_grid"][itau : itau + 1, :],
                weight=weights[itau : itau + 1],
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
                self.data_len,
                self.len_part_features * self.num_particles_in_grid,
                self._builderConfig[cone]["n_cells"],
                self._builderConfig[cone]["n_cells"],
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

    def process_multiple_files(self, filenames, idx_file):
        datas = []
        for fn in filenames:
            data = ak.from_parquet(fn)
            x = self.process_file_data(data)
            if x is None:
                continue
            datas.append(x)

        assert len(datas) > 0
        datas = sum(datas[1:], datas[0])
        p = osp.join(self.processed_dir, "data_{}.pt".format(idx_file))
        print("saved {} samples to {}".format(len(datas), p))
        torch.save(datas, p)

    def process(self, num_files_to_batch):
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            self.process_multiple_files(fns, idx_file)
            idx_file += 1

    def process_parallel(self, num_files_to_batch, num_proc):
        pars = []
        idx_file = 0
        for fns in chunks(self.raw_file_names, num_files_to_batch):
            pars += [(self, fns, idx_file)]
            idx_file += 1
        pool = multiprocessing.Pool(num_proc)
        pool.map(process_func, pars)
        # for p in pars:
        #     process_func(p)

    def get(self, idx):
        fn = "data_{}.pt".format(idx)
        p = osp.join(self.processed_dir, fn)
        data = torch.load(p, map_location="cpu")
        print("loaded {}, N={}".format(fn, len(data)))
        return data

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return len(self.processed_file_names)


if __name__ == "__main__":

    infile = sys.argv[1]
    ds = osp.basename(infile).split(".")[0]
    sig_ntuples_dir = '/scratch-persistent/snandan/CLIC_tau_ntuples/Grid/ZH_Htautau'
    bkg_ntuples_dir = '/scratch-persistent/snandan/CLIC_tau_ntuples/Grid/QCD'

    filelist = get_split_files(infile, ds, sig_ntuples_dir, bkg_ntuples_dir)
    outp = "data/dataset_{}".format(ds)
    os.makedirs(outp, exist_ok=True)
    ds = TauJetDatasetWithGrid(outp, filelist)

    # merge 50 files, run 16 processes
    ds.process_parallel(50, 8)
    #filelist = list(glob(osp.join(sys.argv[1], "*.parquet")))
    #ds = TauJetDatasetWithGrid(filelist)
    #print("Loaded TauJetDataset with {} files".format(len(ds)))

    # treat each input file like a batch
    for ibatch in range(len(ds)):
        data = ds[ibatch]
        print("shape of inner grid ", data[0]["inner_grid"].shape)
        print("shape of outer grid ", data[0]["outer_grid"].shape)
        break
