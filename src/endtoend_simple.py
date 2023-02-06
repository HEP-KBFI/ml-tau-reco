import os
import hydra
import vector
import awkward as ak
import numpy as np
import tqdm
import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.loader import DataLoader
from taujetdataset import TauJetDataset

from torch_geometric.nn.aggr import AttentionalAggregation
from basicTauBuilder import BasicTauBuilder
from glob import glob
import os.path as osp


class TauEndToEndSimple(nn.Module):
    def __init__(
        self,
    ):
        super(TauEndToEndSimple, self).__init__()

        self.act = nn.ELU

        width = 128
        embedding_dim = 32

        self.nn_pf_initialembedding = nn.Sequential(
            nn.Linear(6, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, embedding_dim),
        )

        self.nn_pf_mha = nn.ModuleList()
        self.nn_pf_mha_norm = nn.ModuleList()
        for i in range(3):
            self.nn_pf_mha.append(torch.nn.MultiheadAttention(embedding_dim, 4, batch_first=True))
            self.nn_pf_mha_norm.append(torch.nn.LayerNorm(embedding_dim))

        self.nn_pf_ga = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(embedding_dim, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, 1),
            )
        )

        self.nn_pred_istau = nn.Sequential(
            nn.Linear(4 + embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 1),
        )

        self.nn_pred_visenergy = nn.Sequential(
            nn.Linear(4 + embedding_dim, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, width),
            self.act(),
            nn.Linear(width, 1),
        )

    def forward(self, batch):
        pf_encoded = self.nn_pf_initialembedding(batch.jet_pf_features)

        # pad the PF tensors across jets
        pfs_list = list(torch_geometric.utils.unbatch(pf_encoded, batch.pf_to_jet))
        pfs_nested = torch.nested.nested_tensor(pfs_list)
        pfs_padded = torch.nested.to_padded_tensor(pfs_nested, 0.0)
        mask = pfs_padded[:, :, -1] == 0.0

        # run a simple self-attention over the PF candidates in each jet
        for mha_layer, norm in zip(self.nn_pf_mha, self.nn_pf_mha_norm):
            pf_encoded_corr, _ = mha_layer(pfs_padded, pfs_padded, pfs_padded, key_padding_mask=mask)
            pf_encoded_corr = pf_encoded_corr * (~mask.unsqueeze(-1))
            pfs_padded = norm(pfs_padded + pf_encoded_corr)

        # get the encoded PF candidates after attention, undo the padding
        pf_encoded = torch.cat([pfs_padded[i][~mask[i]] for i in range(pfs_padded.shape[0])])

        # now collapse the PF information in each jet with a global attention layer
        jet_encoded = self.nn_pf_ga(pf_encoded, batch.pf_to_jet)

        # get the list of per-jet features as a concat of
        # (original_features, multi-head attention features)
        jet_feats = torch.cat([batch.jet_features, jet_encoded], axis=-1)

        # run a binary classification whether or not this jet is from a tau
        pred_istau = torch.sigmoid(self.nn_pred_istau(jet_feats)).squeeze(-1)

        # run a per-jet NN for visible energy prediction
        pred_visenergy = self.nn_pred_visenergy(jet_feats).squeeze(-1)

        return pred_istau, pred_visenergy


def model_loop(model, ds_loader, optimizer, is_train, dev):
    loss_tot = 0.0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    for batch in tqdm.tqdm(ds_loader, total=len(ds_loader)):
        batch = batch.to(device=dev)
        pred_istau, pred_visenergy = model(batch)
        true_visenergy = batch.gen_tau_vis_energy
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        loss_energy = torch.nn.functional.mse_loss(pred_visenergy, true_visenergy)
        loss_cls = torch.nn.functional.binary_cross_entropy(pred_istau, true_istau)
        loss = loss_cls + loss_energy
        if is_train:
            loss.backward()
            optimizer.step()
        loss_tot += loss.detach().cpu().item()
        nsteps += 1
        njets += batch.jet_features.shape[0]
    return loss_tot / njets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleDNNTauBuilder(BasicTauBuilder):
    def __init__(
        self,
        model,
        config={},
    ):
        self.model = model
        model.eval()
        self._builderConfig = dict()
        for key in config:
            self._builderConfig[key] = config[key]

    def processJets(self, jets):
        ds = TauJetDataset()
        data_obj = ds.process_file_data(jets)
        pred_istau, pred_visenergy = self.model(data_obj)

        pred_istau = pred_istau.detach().numpy()
        pred_visenergy = pred_visenergy.detach().numpy()
        njets = len(jets["reco_jet_p4s"]["x"])
        assert njets == len(pred_istau)
        assert njets == len(pred_visenergy)

        tauP4 = vector.awk(
            ak.zip(
                {
                    "px": np.zeros(njets),
                    "py": np.zeros(njets),
                    "pz": np.zeros(njets),
                    "E": pred_visenergy,
                }
            )
        )
        tauCharges = np.zeros(njets)
        dmode = np.zeros(njets)

        # as a dummy placeholder, just return the first PFCand for each jet
        tau_cand_p4s = jets["reco_cand_p4s"][:, 0]

        return {
            "tauP4": tauP4,
            "tauSigCandP4s": tau_cand_p4s,
            "tauClassifier": pred_istau,
            "tauCharge": tauCharges,
            "tauDmode": dmode,
        }


@hydra.main(config_path="../config", config_name="endtoend_simple", version_base=None)
def main(cfg):
    qcd_files = list(glob(osp.join(cfg.input_dir_QCD, "*.parquet")))
    zh_files = list(glob(osp.join(cfg.input_dir_ZH_Htautau, "*.parquet")))
    print("qcd={} zh={}".format(len(qcd_files), len(zh_files)))
 
    qcd_files_train = qcd_files[:cfg.ntrain]
    qcd_files_val = qcd_files[cfg.ntrain:cfg.ntrain+cfg.nval]
    zh_files_train = qcd_files[:cfg.ntrain]
    zh_files_val = qcd_files[cfg.ntrain:cfg.ntrain+cfg.nval]

    ds_train = TauJetDataset(qcd_files_train + zh_files_train)
    ds_val = TauJetDataset(qcd_files_val + zh_files_val)

    print("Loaded TauJetDataset with {} train files".format(len(ds_train)))
    print("Loaded TauJetDataset with {} val files".format(len(ds_val)))
    # note, if batch_size>1, then the pf_to_jet indices will be incorrect and need to be recomputed
    ds_train_loader = DataLoader(ds_train, batch_size=1)
    ds_val_loader = DataLoader(ds_val, batch_size=1)

    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = TauEndToEndSimple().to(device=dev)
    print("params={}".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    for iepoch in range(cfg.epochs):
        loss_train = model_loop(model, ds_train_loader, optimizer, True, dev)
        loss_val = model_loop(model, ds_val_loader, optimizer, False, dev)
        print("epoch={} loss_train={:.4f} loss_val={:.4f}".format(iepoch, loss_train, loss_val))

    model = model.to(device="cpu")
    torch.save(model, "data/model.pt")


if __name__ == "__main__":
    main()
