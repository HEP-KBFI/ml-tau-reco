import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os
import os.path as osp
from glob import glob
import vector
from torch.utils.data import IterableDataset
import multiprocessing
import random
import yaml


def get_split_files(config_path, split):
     with open(config_path, "r") as fi:
         data = yaml.safe_load(fi)
         paths = data[split]["paths"]
         return paths


def process_func(args):
     self, fns, idx_file = args
     return self.process_multiple_files(fns, idx_file)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# 
class AwkwardDataset:
    def __init__(self, values, value_cols = None, label_cols='label', data_format='channel_last'):
        self.value_cols = value_cols if value_cols is not None else {
            'points': ['pf_etarel', 'pf_phirel'],
            'features': ['pf_pt_log', 'pf_e_log', 'pf_etarel', 'pf_phirel', 'pf_signed_dxy', 'pf_signed_dz', 'pf_sigma_dxy', 'pf_sigma_dz', 'pf_charge', 'pf_pdg', 'pf_jet_px', 'pf_jet_py', 'pf_jet_pz','pf_jet_E'],
            'features_test': ['pf_npf', 'pf_etarel', 'pf_phirel', 'pf_px', 'pf_py', 'pf_pz', 'pf_e', 'pf_signed_dxy', 'pf_signed_dz', 'pf_sigma_dxy', 'pf_sigma_dz', 'pf_charge', 'pf_pdg', 'pf_jet_px', 'pf_jet_py', 'pf_jet_pz','pf_jet_E'],
            'mask': ['pf_pt_log'],
            'weight': ['pf_weights'],
            'weight_test': ['pf_weights_test'],
            'fracs': ['pf_fracs']
        }
        self.label_cols = label_cols
        assert data_format in ('channel_first', 'channel_last')
        self.stack_axis = 1 if data_format=='channel_first' else -1
        # Here we make the arrays which will keep out data and load the first batch
        self._values = {}
        self._label = None
        self._load(values)

    def _load(self, values):
        for k in self.value_cols:
            cols = self.value_cols[k]
            arrs = []
            for col in cols:
                arrs.append(values[col])
            self._values[k] = np.stack(arrs, axis=self.stack_axis)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        return self._label if key == self.label_cols else self._values[key]

    @property
    def X(self):
        return self._values

    def shuffle(self, seed=None):
        # Get a random permutation
        if seed is not None: np.random.seed(seed)
        shuffle_indices = np.random.permutation(self.__len__())
        # Reorder the table
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]

class TauJetDataset(Dataset):
    # def __init__(self, filelist=[]):
    def __init__(self, processed_dir="", filelist=[]):
        self._processed_dir = processed_dir
        self.filelist = filelist
        self.npfmax =20
        random.shuffle(self.filelist)
        # # just load all data to memory
        # self.all_data = []
        # for fn in self.processed_file_names:
        #     print(fn)
        #     data = ak.from_parquet(fn)
        #     self.all_data += self.process_file_data(data)
        
    # @property
    # def processed_file_names(self):
    #     return self.filelist


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

    def __len__(self):
        return len(self.processed_file_names)
        # return len(self.all_data)

    def asP4(self, p4):
        P4=vector.awk(
            ak.zip(
                {
                    "mass": p4.tau,
                    "x": p4.x,
                    "y": p4.y,
                    "z": p4.z,
                }
            )
        )
        return P4

    def pad_array(self,jagged_array, max_len, value=0., dtype='float32'):
        rectangluar_array = np.full(shape=(len(jagged_array), max_len), fill_value=value, dtype=dtype)
        for idx, jagged_element in enumerate(jagged_array):
            if len(jagged_element) != 0:
                trunc = ak.to_numpy(jagged_element[:max_len]).astype(dtype)
                rectangluar_array[idx, :len(trunc)] = trunc
        return ak.Array(rectangluar_array)

    def get_gnn_feats(self, data: ak.Record) -> (AwkwardDataset):
        jet_p4 = self.asP4( data["reco_jet_p4s"] )
        p4s = self.asP4( data["reco_cand_p4s"] )
        genmatched_E = data["reco_cand_matched_gen_energy"]
        pf_signed_dxy = np.abs(data["reco_cand_dxy"])
        pf_signed_dz = np.abs(data["reco_cand_dz"])
        pf_sigma_dxy = np.abs(data["reco_cand_dxy"])/data["reco_cand_dxy_err"]
        pf_sigma_dz = np.abs(data["reco_cand_dz"])/data["reco_cand_dz_err"]
        pf_charge = data["reco_cand_charge"]
        pf_pdg = np.abs(data["reco_cand_pdg"])

        p4s_new = []
        genFracs = []
        weights = []
        weights_test = []
        pfs_signed_dxy_new = []
        pfs_signed_dz_new = []
        pfs_sigma_dxy_new = []
        pfs_sigma_dz_new = []
        pfs_charge_new = []
        pfs_pdg_new = []
        pfs_jet_px = []
        pfs_jet_py = []
        pfs_jet_pz = []
        pfs_jet_E = []
        pfs_npf = []
        for ip4, p4 in enumerate(p4s):
            # sort by pt as we later want to take the 20 highest pt cands
            p4new, genmatched_new, pf_signed_dxy_new, pf_signed_dz_new, pf_sigma_dxy_new, pf_sigma_dz_new, pf_charge_new, pf_pdg_new   = zip(*sorted(zip(p4,genmatched_E[ip4], pf_signed_dxy[ip4], pf_signed_dz[ip4], pf_sigma_dxy[ip4],pf_sigma_dz[ip4], pf_charge[ip4],pf_pdg[ip4]), key=lambda x: x[0].pt, reverse=True))
            p4new = self.asP4(ak.Array(p4new))
            p4s_new.append(p4new)
            genFracs_temp = genmatched_new/p4new.energy
            # get rid of -1s in BG
            genFracs_temp = ak.max([genFracs_temp,ak.zeros_like(genFracs_temp)], axis=0)
            genFracs_temp = ak.min([genFracs_temp,ak.ones_like(genFracs_temp)], axis=0)
            genFracs.append(genFracs_temp)
            if data["gen_jet_tau_decaymode"][ip4]>=0:
                weights.append(p4new.energy/jet_p4[ip4].energy)
                weights_test.append(np.sqrt(p4new.energy/(jet_p4[ip4].energy/len(p4new.energy))))
            else:
                weights.append(ak.zeros_like(p4new.energy))
                weights_test.append(ak.zeros_like(p4new.energy))
            pfs_signed_dxy_new.append(ak.Array(pf_signed_dxy_new))
            pfs_signed_dz_new.append(ak.Array(pf_signed_dz_new))
            pfs_sigma_dxy_new.append(ak.Array(pf_sigma_dxy_new))
            pfs_sigma_dz_new.append(ak.Array(pf_sigma_dz_new))
            pfs_charge_new.append(ak.Array(pf_charge_new))
            pfs_pdg_new.append(ak.Array(pf_pdg_new))
            pfs_jet_px.append( jet_p4[ip4].px * ak.ones_like(genFracs_temp))
            pfs_jet_py.append( jet_p4[ip4].py * ak.ones_like(genFracs_temp))
            pfs_jet_pz.append( jet_p4[ip4].pz * ak.ones_like(genFracs_temp))
            pfs_jet_E.append( jet_p4[ip4].energy * ak.ones_like(genFracs_temp))
            pfs_npf.append(len(genFracs_temp) * ak.ones_like(genFracs_temp))
        p4s_new = ak.Array(p4s_new)
        genFracs = ak.Array(genFracs)
        weights = ak.Array(weights)
        weights_test = ak.Array(weights_test)
        pfs_signed_dxy_new = ak.Array(pfs_signed_dxy_new)
        pfs_signed_dz_new = ak.Array(pfs_signed_dz_new)
        pfs_sigma_dxy_new = ak.Array(pfs_sigma_dxy_new)
        pfs_sigma_dz_new = ak.Array(pfs_sigma_dz_new)
        pfs_charge_new = ak.Array(pfs_charge_new)
        pfs_pdg_new = ak.Array(pfs_pdg_new)
        pfs_jet_px_new = ak.Array(pfs_jet_px)
        pfs_jet_py_new = ak.Array(pfs_jet_py)
        pfs_jet_pz_new = ak.Array(pfs_jet_pz)
        pfs_jet_E_new = ak.Array(pfs_jet_E)
        pfs_npf = ak.Array(pfs_npf)
        rets = {}
        rets['pf_pt_log'] = self.pad_array(np.log(p4s.pt), self.npfmax)
        rets['pf_e_log'] = self.pad_array(np.log(p4s.energy), self.npfmax)
        rets['pf_e'] = self.pad_array(p4s.energy, self.npfmax)
        rets['pf_px'] = self.pad_array(p4s.px, self.npfmax)
        rets['pf_py'] = self.pad_array(p4s.py, self.npfmax)
        rets['pf_pz'] = self.pad_array(p4s.pz, self.npfmax)
        # rets['pf_pt_log'] = np.log(p4s.pt)
        # rets['pf_e_log'] = np.log(p4s.energy)
        _jet_etasign = np.sign((jet_p4.eta).to_numpy())
        _jet_etasign[_jet_etasign==0] = 1
        _jet_etasign_forPF = []
        _jet_eta_forPF = []
        for ij, j in enumerate(_jet_etasign):
            _jet_etasign_forPF.append(ak.Array(j*np.ones(len(p4s_new[ij]))))
            _jet_eta_forPF.append(ak.Array((jet_p4[ij].eta*np.ones(len(p4s_new[ij])))))
        _jet_etasign_forPF = ak.Array(_jet_etasign_forPF)
        _jet_eta_forPF = ak.Array(_jet_eta_forPF)
        rets['pf_etarel'] = self.pad_array((p4s.eta - _jet_eta_forPF) * _jet_etasign_forPF, self.npfmax)
        rets['pf_phirel'] = self.pad_array(p4s.deltaphi(jet_p4),self.npfmax)
        rets['pf_fracs'] = self.pad_array( genFracs,self.npfmax )
        rets['pf_weights'] = self.pad_array( weights,self.npfmax )
        rets['pf_weights_test'] = self.pad_array( weights_test,self.npfmax )
        rets['pf_signed_dxy'] = self.pad_array( pfs_signed_dxy_new,self.npfmax )
        rets['pf_signed_dz'] = self.pad_array( pfs_signed_dz_new,self.npfmax )
        rets['pf_sigma_dxy'] = self.pad_array( pfs_sigma_dxy_new,self.npfmax )
        rets['pf_sigma_dz'] = self.pad_array( pfs_sigma_dz_new,self.npfmax )
        rets['pf_charge'] = self.pad_array( pfs_charge_new,self.npfmax )
        rets['pf_pdg'] = self.pad_array( pfs_pdg_new,self.npfmax )
        rets['pf_jet_px'] = self.pad_array( pfs_jet_px_new, self.npfmax )
        rets['pf_jet_py'] = self.pad_array( pfs_jet_py_new, self.npfmax )
        rets['pf_jet_pz'] = self.pad_array( pfs_jet_pz_new, self.npfmax )
        rets['pf_jet_E'] = self.pad_array( pfs_jet_E_new, self.npfmax )
        rets['pf_npf'] = self.pad_array( pfs_npf, self.npfmax )
        # rets['pf_etarel'] = (p4s.eta - _jet_eta_forPF) * _jet_etasign_forPF
        # rets['pf_phirel'] = p4s.deltaphi(jet_p4)
        # rets['pf_fracs'] =  genFracs
        # rets['pf_weights'] =  weights
        # rets['pf_signed_dxy'] =  pfs_signed_dxy_new
        # rets['pf_signed_dz'] =  pfs_signed_dz_new
        # rets['pf_sigma_dxy'] =  pfs_sigma_dxy_new
        # rets['pf_sigma_dz'] =  pfs_sigma_dz_new
        # rets['pf_charge'] =  pfs_charge_new
        # rets['pf_pdg'] =  pfs_pdg_new

        return AwkwardDataset(rets, data_format='channel_last')

    def process_file_data(self, data):
        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = self.asP4(data["gen_jet_tau_p4s"])
        gen_tau_p4 = torch.tensor(np.stack([p4.px, p4.py, p4.pz, p4.energy], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]
        gnndset = self.get_gnn_feats(data)
        perjet_weight = torch.tensor(data["weight"]).to(dtype=torch.float)
        tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.float)
        tau_charge = torch.tensor(data["gen_jet_tau_charge"]).to(dtype=torch.float)
        ##print(gnndset.X["weight"][0])
        #print(gnndset.X["weight_test"][0])
        ret_data = [
            Data(
                gen_tau_decaymode=gen_tau_decaymode[ijet : ijet + 1],
                gen_tau_p4=gen_tau_p4[ijet : ijet + 1],
                perjet_weight = perjet_weight[ijet : ijet + 1],
                gnnfeats = torch.tensor(gnndset.X["features"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                gnnfeats_test = torch.tensor(gnndset.X["features_test"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                gnnpos = torch.tensor(gnndset.X["points"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                gnnfracs = torch.tensor(gnndset.X["fracs"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                gnnweights = torch.tensor(gnndset.X["weight"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                gnnweights_test = torch.tensor(gnndset.X["weight_test"][ijet:ijet + 1].to_numpy(), dtype=torch.float).squeeze(0),
                tau_decaymode = tau_decaymode[ijet : ijet + 1],
                tau_charge = tau_charge[ijet : ijet + 1]
            )
            for ijet in range(len(gen_tau_decaymode))
        ]
        return ret_data


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
    # def __getitem__(self, idx):
    #     return self.all_data[idx]

if __name__ == "__main__":
    for ds in ["train", "validation"]:
        conf = "config/datasets/{}.yaml".format(ds)
        filelist = get_split_files(conf, ds)
        outp = "data_test/dataset_{}".format(ds)
        os.makedirs(outp)
        ds = TauJetDataset(outp, filelist)

        # merge 50 files, run 16 processes
        ds.process_parallel(50, 16)
