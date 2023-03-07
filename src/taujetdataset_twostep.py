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


class TauJetDataset(Dataset):
    def __init__(self, filelist=[]):

        self.filelist = filelist

        # The order of features in the jet feature tensor
        self.reco_jet_features = ["x", "y", "z", "tau"]

        # The order of features in the PF feature tensor
        self.pf_features = ["Energy", "x", "y", "z", "tau", "charge", "pdg", "reco_cand_signed_dxy", "reco_cand_signed_dz", "reco_cand_sigma_dxy", "reco_cand_sigma_dz"]

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

    def get_jet_features(self, data: ak.Record) -> torch.Tensor:
        jets = {}
        for k in data["reco_jet_p4s"].fields:
            jets[k] = data["reco_jet_p4s"][k]
        # collect jet features in a specific order to an (Njet x Nfeatjet) torch tensor
        jet_feature_tensors = []
        for feat in self.reco_jet_features:
            jet_feature_tensors.append(torch.tensor(jets[feat], dtype=torch.float32))
        jet_features = torch.stack(jet_feature_tensors, axis=-1)
        return jet_features.to(dtype=torch.float32)

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

    def get_pf_adds(self, jet_features: torch.Tensor, data: ak.Record) -> (torch.Tensor, torch.Tensor):
        p4s = self.asP4( data["reco_cand_p4s"] )
        reco_E = p4s.energy
        perjet_weights = data["weight"]
        weights = []
        jet_E = self.asP4( data["reco_jet_p4s"] ).energy
        for i in range(len(jet_E)):
            e_j = jet_E[i]
            perJet_weight = perjet_weights[i]
            weights.append((perJet_weight/e_j)*reco_E[i])
        weights = ak.flatten(ak.Array(weights))
        genmatched_E = data["reco_cand_matched_gen_energy"]
        fracs = ak.flatten(genmatched_E)/ak.flatten(reco_E)
        fracs_feature_tensor = [torch.tensor(fracs,dtype=torch.float32)]
        fracs_feature = torch.stack(fracs_feature_tensor, axis=1)
        weights_feature_tensor = [torch.tensor(weights, dtype=torch.float32)]
        weights_feature = torch.stack(weights_feature_tensor, axis=1)

        # create a tensor with (Ncand x 1) which assigns each PF candidate to the jet it belongs to
        # this can be treated like batch_index in downstream algos

        pf_per_jet = ak.num(genmatched_E, axis=1)
        pf_to_jet = torch.tensor(np.repeat(np.arange(len(jet_features)), pf_per_jet))

        return fracs_feature.to(dtype=torch.float32), weights_feature.to(dtype=torch.float32) , pf_to_jet.to(dtype=torch.long)
 
    def get_pf_features(self, jet_features: torch.Tensor, data: ak.Record) -> (torch.Tensor, torch.Tensor):
        pfs = {}
        for k in data["reco_cand_p4s"].fields:
            pfs[k] = data["reco_cand_p4s"][k]
        pfs["charge"] = data["reco_cand_charge"]
        pfs["pdg"] = np.abs(data["reco_cand_pdg"])
        pfs["reco_cand_signed_dxy"] = np.abs(data["reco_cand_dxy"])
        pfs["reco_cand_signed_dz"] = np.abs(data["reco_cand_dz"])
        pfs["reco_cand_sigma_dxy"] = np.abs(data["reco_cand_dxy"])/data["reco_cand_dxy_err"]
        pfs["reco_cand_sigma_dz"] = np.abs(data["reco_cand_dz"])/data["reco_cand_dz_err"]
        p4s = self.asP4( data["reco_cand_p4s"] )
        pfs["Energy"] = p4s.energy
        # collect PF features in a specific order to an (Ncand x Nfeatcand) torch tensor
        pf_feature_tensors = []
        for feat in self.pf_features:
            pf_feature_tensors.append(torch.tensor(ak.flatten(pfs[feat]), dtype=torch.float32))
        pf_features = torch.stack(pf_feature_tensors, axis=-1)

        # create a tensor with (Ncand x 1) which assigns each PF candidate to the jet it belongs to
        # this can be treated like batch_index in downstream algos

        pf_per_jet = ak.num(pfs["pdg"], axis=1)
        pf_to_jet = torch.tensor(np.repeat(np.arange(len(jet_features)), pf_per_jet))

        return pf_features.to(dtype=torch.float32), pf_to_jet.to(dtype=torch.long)

    
    def process_file_data(self, data):
        # collect all jet features
        jet_features = self.get_jet_features(data)

        # collect all jet PF candidate features
        pf_features, pf_to_jet = self.get_pf_features(jet_features, data)
        pf_efrac, pf_weights ,pf_to_jet2 = self.get_pf_adds(jet_features, data)

        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        p4 = data["gen_jet_tau_p4s"]
        gen_tau_p4 = torch.tensor(np.stack([p4.x, p4.y, p4.z, p4.tau], axis=-1)).to(dtype=torch.float32)
        assert gen_tau_p4.shape[0] == gen_tau_decaymode.shape[0]
        
        perjet_weight = torch.tensor(data["weight"]).to(dtype=torch.int32)
        # Data object with:
        #   - reco jet (jet_features, jet_pf_features)
        #   - jet PF candidates (jet_pf_features, pf_to_jet)
        #   - generator level target (gen_tau_decaymode, gen_tau_p4)
        ret_data = [
            Data(
                jet_features=jet_features[ijet : ijet + 1, :],
                jet_pf_features=pf_features[pf_to_jet == ijet],
                gen_tau_decaymode=gen_tau_decaymode[ijet : ijet + 1],
                gen_tau_p4=gen_tau_p4[ijet : ijet + 1],
                jet_pf_efrac = pf_efrac[pf_to_jet2 == ijet],
                pf_weights = pf_weights[pf_to_jet2 == ijet],
                perjet_weight = perjet_weight[ijet : ijet + 1]
            )
            for ijet in range(len(jet_features))
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
        print(ibatch, batch.jet_features.shape, batch.jet_pf_features.shape)
        assert batch.jet_features.shape[0] == batch.gen_tau_decaymode.shape[0]
        assert batch.jet_features.shape[0] == batch.gen_tau_p4.shape[0]
