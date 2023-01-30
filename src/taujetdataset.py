import awkward as ak
import sys
import torch
import glob
import numpy as np
import torch_geometric
import torch_geometric.loader
import tqdm
from torch_geometric.data import Data, Dataset
import os.path as osp
from glob import glob 

# Temporary: generate an example input file with the correct format
njet = 1000
jet_pfs = [] 
for ijet in range(njet):
    ncand = np.random.randint(5,20)
    pf = {
        "rho": np.zeros(ncand, dtype=np.float32),
        "eta": np.zeros(ncand, dtype=np.float32),
        "phi": np.zeros(ncand, dtype=np.float32),
        "tau": np.zeros(ncand, dtype=np.float32),
    }
    jet_pfs.append(pf)
jet_pfs = ak.from_iter(jet_pfs)

data = ak.Record({"reco_jet_p4s": {
        "rho": np.zeros(njet, dtype=np.float32),
        "eta": np.zeros(njet, dtype=np.float32),
        "phi": np.zeros(njet, dtype=np.float32),
        "tau": np.zeros(njet, dtype=np.float32),
    }, "jet_pfs": jet_pfs})

ak.to_parquet(data, "data/test1.parquet")
ak.to_parquet(data, "data/test2.parquet")
ak.to_parquet(data, "data/test3.parquet")

class TauJetDataset(Dataset):
    def __init__(self, path):
        #replace this with the actual generated files
        self.path = "data/"

    @property
    def processed_file_names(self):
        raw_list = glob(osp.join(self.path, "*.parquet"))
        return sorted(raw_list)

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = ak.from_parquet(self.processed_file_names[idx])

        #collect all jet features to a single dict
        jets = {}
        for k in data["reco_jet_p4s"].fields:
            jets[k] = data["reco_jet_p4s"][k]
        #collect all jet PF candidate features to a single dict
        pfs = {}
        for k in data["jet_pfs"].fields:
            pfs[k] = data["jet_pfs"][k]

        #collect jet features in a specific order to an (Njet x Nfeatjet) torch tensor
        reco_jet_features = ["rho", "eta", "phi", "tau"]
        jet_feature_tensors = []
        for feat in reco_jet_features:
            jet_feature_tensors.append(torch.tensor(jets[feat], dtype=torch.float32))
        jet_features = torch.stack(jet_feature_tensors, axis=-1)

        #collect PF features in a specific order to an (Ncand x Nfeatcand) torch tensor
        pf_features = ["rho", "eta", "phi", "tau"]
        pf_feature_tensors = []
        for feat in pf_features:
            pf_feature_tensors.append(torch.tensor(ak.flatten(pfs[feat]), dtype=torch.float32))
        pf_features = torch.stack(pf_feature_tensors, axis=-1)

        #create a tensor with (Ncand x 1) which assigns each PF candidate to the jet it belongs to
        #this can be treated like batch_index in downstream algos
        pf_per_jet = ak.num(pfs["rho"], axis=1)
        pf_to_jet = torch.tensor(np.repeat(np.arange(len(jet_features)), pf_per_jet))

        #Data object with jet_features=(Njet x Nfeatjet), pf_features=(Ncand x Nfeatcand), pf_to_jet=(Ncand x 1)
        #Njet is the number of jets in the input file
        data = Data(jet_features=jet_features, jet_pf_features=pf_features, pf_to_jet=pf_to_jet)

        return data

if __name__ == "__main__":
    ds = TauJetDataset()
    #treat each input file like a batch 
    for batch in ds:
        print(batch.jet_features.shape, batch.jet_pf_features.shape)
     

