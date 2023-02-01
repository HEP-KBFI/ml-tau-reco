import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob


class TauJetDataset(Dataset):
    def __init__(self, path=""):
        self.path = path

        # The order of features in the jet feature tensor
        self.reco_jet_features = ["x", "y", "z", "tau"]

        # The order of features in the PF feature tensor
        self.pf_features = ["x", "y", "z", "tau", "charge", "pdg"]

    @property
    def processed_file_names(self):
        raw_list = glob(osp.join(self.path, "*.parquet"))
        return sorted(raw_list)

    def __len__(self):
        return len(self.processed_file_names)

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

    def get_pf_features(self, jet_features: torch.Tensor, data: ak.Record) -> (torch.Tensor, torch.Tensor):
        pfs = {}
        for k in data["reco_cand_p4s"].fields:
            pfs[k] = data["reco_cand_p4s"][k]
        pfs["charge"] = data["reco_cand_charge"]
        pfs["pdg"] = np.abs(data["reco_cand_pdg"])
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

        gen_tau_decaymode = torch.tensor(data["gen_jet_tau_decaymode"]).to(dtype=torch.int32)
        gen_tau_vis_energy = torch.tensor(data["gen_jet_tau_vis_energy"]).to(dtype=torch.float32)

        # Data object with:
        #   - reco jet (jet_features, jet_pf_features)
        #   - jet PF candidates (jet_pf_features, pf_to_jet)
        #   - generator level target (gen_tau_decaymode, gen_tau_vis_energy)

        ret_data = Data(
            jet_features=jet_features,  # (Njet x Nfeat_jet) of jet features
            jet_pf_features=pf_features,  # (Ncand x Nfeat_cand) of PF features
            pf_to_jet=pf_to_jet,  # (Ncand x 1) index of PF candidate to jet
            gen_tau_decaymode=gen_tau_decaymode,  # (Njet x 1) of gen tau decay mode or -1
            gen_tau_vis_energy=gen_tau_vis_energy,  # (Njet x 1) of gen tau visible energy or -1
        )
        return ret_data

    def __getitem__(self, idx):
        # Load the n-th file
        data = ak.from_parquet(self.processed_file_names[idx])
        ret_data = self.process_file_data(data)
        return ret_data


if __name__ == "__main__":
    ds = TauJetDataset(sys.argv[1])
    print("Loaded TauJetDataset with {} files".format(len(ds)))

    # treat each input file like a batch
    for ibatch in range(len(ds)):
        batch = ds[ibatch]
        n_tau = torch.sum(batch.gen_tau_decaymode != -1)
        print(ibatch, batch.jet_features.shape, batch.jet_pf_features.shape, batch.pf_to_jet.shape, n_tau)
        assert batch.jet_features.shape[0] == batch.gen_tau_decaymode.shape[0]
        assert batch.jet_features.shape[0] == batch.gen_tau_vis_energy.shape[0]
        assert batch.jet_pf_features.shape[0] == batch.pf_to_jet.shape[0]
