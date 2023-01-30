import sys
import awkward as ak
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob


class TauJetDataset(Dataset):
    def __init__(self, path):
        # replace this with the actual generated files
        self.path = path

    @property
    def processed_file_names(self):
        raw_list = glob(osp.join(self.path, "*.parquet"))
        assert(len(raw_list)>0)
        return sorted(raw_list)

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = ak.from_parquet(self.processed_file_names[idx])
        print(data.fields)

        # collect all jet features to a single dict
        jets = {}
        for k in data["reco_jet_p4s"].fields:
            jets[k] = data["reco_jet_p4s"][k]
        print(jets.keys())
        # collect all jet PF candidate features to a single dict
        pfs = {}
        for k in data["reco_cand_p4s"].fields:
            pfs[k] = data["reco_cand_p4s"][k]
        print(pfs.keys())

        # collect jet features in a specific order to an (Njet x Nfeatjet) torch tensor
        reco_jet_features = ["x", "y", "z", "tau"]
        jet_feature_tensors = []
        for feat in reco_jet_features:
            jet_feature_tensors.append(torch.tensor(jets[feat], dtype=torch.float32))
        jet_features = torch.stack(jet_feature_tensors, axis=-1)

        # collect PF features in a specific order to an (Ncand x Nfeatcand) torch tensor
        pf_features = ["x", "y", "z", "tau"]
        pf_feature_tensors = []
        for feat in pf_features:
            pf_feature_tensors.append(torch.tensor(ak.flatten(pfs[feat]), dtype=torch.float32))
        pf_features = torch.stack(pf_feature_tensors, axis=-1)

        # create a tensor with (Ncand x 1) which assigns each PF candidate to the jet it belongs to
        # this can be treated like batch_index in downstream algos
        pf_per_jet = ak.num(pfs["tau"], axis=1)
        pf_to_jet = torch.tensor(np.repeat(np.arange(len(jet_features)), pf_per_jet))

        # Data object with jet_features=(Njet x Nfeatjet), pf_features=(Ncand x Nfeatcand), pf_to_jet=(Ncand x 1)
        # Njet is the number of jets in the input file
        data = Data(jet_features=jet_features, jet_pf_features=pf_features, pf_to_jet=pf_to_jet)
        return data


if __name__ == "__main__":
    ds = TauJetDataset(sys.argv[1])
    # treat each input file like a batch
    for ibatch in range(len(ds)):
        batch = ds[ibatch]
        print(batch.jet_features.shape, batch.jet_pf_features.shape, batch.pf_to_jet.shape)
