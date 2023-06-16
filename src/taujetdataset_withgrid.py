# ./scripts/run-env.sh python3 src/taujetdataset_withgrid.py config/datasets/train.yaml /local/snandan/CLIC_data
import sys
import vector
import awkward as ak
import torch
from torch_geometric.data import Data
import os.path as osp
import os
import random
from glob import glob
import json
import numpy as np
import multiprocessing
from build_grid import GridBuilder

from part_var import Var
from auxiliary import get_split_files, chunks, process_func

# np.set_printoptions(threshold=sys.maxsize)


class TauJetDatasetWithGrid:
    def __init__(self, processed_dir="", filelist=[], outputdir="", cfgFileName="./config/grid_builder.json"):
        self._processed_dir = processed_dir
        self.tau_p4_features = ["pt", "theta", "phi", "mass"]
        self.filelist = filelist
        self.od = outputdir
        random.shuffle(self.filelist)
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
        taup4s = vector.awk(
            ak.zip(
                {
                    "px": data["tau_p4s"].x,
                    "py": data["tau_p4s"].y,
                    "pz": data["tau_p4s"].z,
                    "mass": data["tau_p4s"].tau,
                }
            )
        )
        taus["pt"] = taup4s.pt
        taus["theta"] = taup4s.theta
        taus["mass"] = taup4s.mass
        taus["phi"] = taup4s.phi
        # collect tau features in a specific order to an (Njet x Nfeatjet) torch tensor
        tau_feature_tensors = []
        for feat in self.tau_p4_features:
            tau_feature_tensors.append(torch.tensor(taus[feat], dtype=torch.float32))
        for feat in self.tauftrs:
            if "multiplicity" in feat:
                tau_feature_tensors.append(self.calculate_multiplicuty(data, feat))
                continue
            tau_feature_tensors.append(torch.tensor(data[feat], dtype=torch.float32))
        tau_features = torch.stack(tau_feature_tensors, axis=-1)
        return tau_features.to(dtype=torch.float32)

    def calculate_multiplicuty(self, data: ak.Record, part: str) -> torch.tensor:
        grid = "inner_grid" if "inner" in part else "outer_grid"
        if "ele" in part:
            idx1 = Var.isele.value - 1
            idx2 = (Var.max_value() + Var.isele.value) - 1
        elif "mu" in part:
            idx1 = Var.ismu.value - 1
            idx2 = (Var.max_value() + Var.ismu.value) - 1
        elif "ch" in part:
            idx1 = Var.isch.value - 1
            idx2 = (Var.max_value() + Var.isch.value) - 1
        elif "nh" in part:
            idx1 = Var.isnh.value - 1
            idx2 = (Var.max_value() + Var.isnh.value) - 1
        elif "gamma" in part:
            idx1 = Var.isgamma.value - 1
            idx2 = (Var.max_value() + Var.isgamma.value) - 1
        else:
            print("provide correct particle type")
            assert 0
        return torch.tensor(
            np.sum(data[grid].to_numpy()[:, idx1, :, :], axis=(1, 2))
            + np.sum(data[grid].to_numpy()[:, idx2, :, :], axis=(1, 2)),
            dtype=torch.int,
        )

    def get_part_block_features(self, data: ak.Record) -> dict:
        part_block_frs = {}
        for cone in ["inner_grid", "outer_grid"]:
            block_name = f"{cone}"
            part_block_features = torch.tensor(data[block_name].to_numpy(), dtype=torch.float32)
            part_block = part_block_features
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
        torch.save(datas, p)
        print("saved {} samples to {}".format(len(datas), p))
        os.makedirs(self.od, exist_ok=True)
        os.system(f"mv {p} {self.od}")

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
    sig_ntuples_dir = "/local/snandan/DeepTau_wd0/Grid/ZH_Htautau/"
    bkg_ntuples_dir = "/local/snandan/DeepTau_wd0/Grid/QCD/"

    filelist = get_split_files(infile, ds, sig_ntuples_dir, bkg_ntuples_dir)
    outp = "data/dataset_{}".format(ds)
    os.makedirs(outp, exist_ok=True)
    outputdir = os.path.join(sys.argv[2], "dataset_{}".format(ds))
    ds = TauJetDatasetWithGrid(outp, filelist, outputdir)

    # merge 50 files, run 16 processes
    ds.process_parallel(50, 8)

    for ibatch in range(len(ds)):
        data = ds[ibatch]
        print("shape of inner grid ", data[0]["inner_grid"].shape)
        print("shape of outer grid ", data[0]["outer_grid"].shape)
        break
