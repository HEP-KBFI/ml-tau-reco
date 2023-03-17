import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob
import vector

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



class AwkwardDataset:
    def __init__(self, values, value_cols = None, label_cols='label', data_format='channel_last'):
        self.value_cols = value_cols if value_cols is not None else {
            'points': ['pf_etarel', 'pf_phirel'],
            'features': ['pf_pt_log', 'pf_e_log', 'pf_etarel', 'pf_phirel', 'pf_signed_dxy', 'pf_signed_dz', 'pf_sigma_dxy', 'pf_sigma_dz', 'pf_charge', 'pf_pdg'],
            'mask': ['pf_pt_log'],
            'weight': ['pf_weights'],
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
    def __init__(self, filelist=[]):

        self.filelist = filelist

        # just load all data to memory
        self.all_data = []
        for fn in self.processed_file_names:
            print(fn)
            data = ak.from_parquet(fn)
            self.all_data += self.process_file_data(data)

    @property
    def processed_file_names(self):
        return self.filelist

    def __len__(self):
        return len(self.all_data)

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
        pfs_signed_dxy_new = []
        pfs_signed_dz_new = []
        pfs_sigma_dxy_new = []
        pfs_sigma_dz_new = []
        pfs_charge_new = []
        pfs_pdg_new = []
        for ip4, p4 in enumerate(p4s):
            # sort by pt as we later want to take the 20 highest pt cands
            p4new, genmatched_new, pf_signed_dxy_new, pf_signed_dz_new, pf_sigma_dxy_new, pf_sigma_dz_new, pf_charge_new, pf_pdg_new   = zip(*sorted(zip(p4,genmatched_E[ip4], pf_signed_dxy[ip4], pf_signed_dz[ip4], pf_sigma_dxy[ip4],pf_sigma_dz[ip4], pf_charge[ip4],pf_pdg[ip4]), key=lambda x: x[0].pt, reverse=True))
            p4new = self.asP4(ak.Array(p4new))
            p4s_new.append(p4new)
            genFracs.append(genmatched_new/p4new.energy)
            weights.append((data["gen_jet_tau_decaymode"][ip4]>0/jet_p4[ip4].energy)*p4new.energy)
            pfs_signed_dxy_new.append(ak.Array(pf_signed_dxy_new))
            pfs_signed_dz_new.append(ak.Array(pf_signed_dz_new))
            pfs_sigma_dxy_new.append(ak.Array(pf_sigma_dxy_new))
            pfs_sigma_dz_new.append(ak.Array(pf_sigma_dz_new))
            pfs_charge_new.append(ak.Array(pf_charge_new))
            pfs_pdg_new.append(ak.Array(pf_pdg_new))
        p4s_new = ak.Array(p4s_new)
        genFracs = ak.Array(genFracs)
        weights = ak.Array(weights)
        pfs_signed_dxy_new = ak.Array(pfs_signed_dxy_new)
        pfs_signed_dz_new = ak.Array(pfs_signed_dz_new)
        pfs_sigma_dxy_new = ak.Array(pfs_sigma_dxy_new)
        pfs_sigma_dz_new = ak.Array(pfs_sigma_dz_new)
        pfs_charge_new = ak.Array(pfs_charge_new)
        pfs_pdg_new = ak.Array(pfs_pdg_new)
        rets = {}
        rets['pf_pt_log'] = self.pad_array(np.log(p4s.pt), 20)
        rets['pf_e_log'] = self.pad_array(np.log(p4s.energy), 20)
        _jet_etasign = np.sign((jet_p4.eta).to_numpy())
        _jet_etasign[_jet_etasign==0] = 1
        _jet_etasign_forPF = []
        _jet_eta_forPF = []
        for ij, j in enumerate(_jet_etasign):
            _jet_etasign_forPF.append(ak.Array(j*np.ones(len(p4s_new[ij]))))
            _jet_eta_forPF.append(ak.Array((jet_p4[ij].eta*np.ones(len(p4s_new[ij])))))
        _jet_etasign_forPF = ak.Array(_jet_etasign_forPF)
        _jet_eta_forPF = ak.Array(_jet_eta_forPF)
        rets['pf_etarel'] = self.pad_array((p4s.eta - _jet_eta_forPF) * _jet_etasign_forPF, 20)
        rets['pf_phirel'] = self.pad_array(p4s.deltaphi(jet_p4),20)
        rets['pf_fracs'] = self.pad_array( genFracs,20 )
        rets['pf_weights'] = self.pad_array( weights,20 )
        rets['pf_signed_dxy'] = self.pad_array( pfs_signed_dxy_new,20 )
        rets['pf_signed_dz'] = self.pad_array( pfs_signed_dz_new,20 )
        rets['pf_sigma_dxy'] = self.pad_array( pfs_sigma_dxy_new,20 )
        rets['pf_sigma_dz'] = self.pad_array( pfs_sigma_dz_new,20 )
        rets['pf_charge'] = self.pad_array( pfs_charge_new,20 )
        rets['pf_pdg'] = self.pad_array( pfs_pdg_new,20 )
        return AwkwardDataset(rets, data_format='channel_last')

    def process_file_data(self, data):
        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = data["gen_jet_tau_p4s"]
        gen_tau_p4 = torch.tensor(np.stack([p4.x, p4.y, p4.z, p4.tau], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]

        gnndset = self.get_gnn_feats(data)
        perjet_weight = torch.tensor(data["weight"]).to(dtype=torch.int32)
        ret_data = [
            Data(
                gen_tau_decaymode=gen_tau_decaymode[ijet : ijet + 1],
                gen_tau_p4=gen_tau_p4[ijet : ijet + 1],
                perjet_weight = perjet_weight[ijet : ijet + 1],
                gnnfeats = torch.tensor(gnndset.X["features"][ijet:ijet + 1].to_numpy(), dtype=torch.float),
                gnnpos = torch.tensor(gnndset.X["points"][ijet:ijet + 1].to_numpy(), dtype=torch.float),
                gnnfracs = torch.tensor(gnndset.X["fracs"][ijet:ijet + 1].to_numpy(), dtype=torch.float),
                gnnweights = torch.tensor(gnndset.X["weight"][ijet:ijet + 1].to_numpy(), dtype=torch.float),
            )
            for ijet in range(len(gen_tau_decaymode))
        ]
        return ret_data

    def __getitem__(self, idx):
        return self.all_data[idx]


if __name__ == "__main__":
    filelist = list(glob(osp.join(sys.argv[1], "*.parquet")))
    ds = TauJetDataset(filelist)
    print("Loaded TauJetDataset with {} files".format(len(ds)))

    # treat each input file like a batch
    for ibatch in range(len(ds)):
        batch = ds[ibatch]
        print(ibatch, batch.gen_tau_decaymode.shape, batch.gnngnnfeats.shape)
        assert batch.gen_tau_decaymode.shape[0] == batch.gnnfeats.shape[0]
        assert batch.gen_tau_decaymode.shape[0] == batch.gnnpos.shape[0]
