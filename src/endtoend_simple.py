import os
import hydra
import vector
import awkward as ak
import numpy as np
import yaml
import torch
import torch_geometric
import torch.nn as nn
import sys
import random
from torch_geometric.loader import DataLoader
from taujetdataset import TauJetDataset

# from torch_geometric.nn.aggr import AttentionalAggregation
from basicTauBuilder import BasicTauBuilder

from torch.utils.tensorboard import SummaryWriter


# feedforward network that transformes input_dim->output_dim
def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        nn.Linear(input_dim, width),
        act(),
        nn.Dropout(dropout),
        torch.nn.LayerNorm(width),
        nn.Linear(width, width),
        act(),
        nn.Dropout(dropout),
        torch.nn.LayerNorm(width),
        nn.Linear(width, width),
        act(),
        nn.Dropout(dropout),
        torch.nn.LayerNorm(width),
        nn.Linear(width, width),
        act(),
        nn.Dropout(dropout),
        torch.nn.LayerNorm(width),
        nn.Linear(width, output_dim),
    )


# self-attention layer that transformes [B, N, x] -> [B, N, embedding_dim]
# by attending the N elements in each batch B
class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=32, num_heads=8, width=128, dropout=0.3):
        super(SelfAttentionLayer, self).__init__()
        self.act = nn.ELU
        self.mha = torch.nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.seq = ffn(embedding_dim, embedding_dim, width, self.act, dropout)

    def forward(self, x, mask):
        xa = self.mha(x, x, x, key_padding_mask=mask)[0]
        xa = self.dropout(xa)
        x = self.norm0(x + xa)

        xa = self.seq(x)
        x = self.norm1(x + xa)

        # make sure masked elements are 0
        x = x * (~mask.unsqueeze(-1))
        return x


class TauEndToEndSimple(nn.Module):
    def __init__(
        self,
    ):
        super(TauEndToEndSimple, self).__init__()

        self.act = nn.ELU
        self.dropout = 0.3
        self.width = 512
        self.embedding_dim = 512

        self.nn_pf_initialembedding = ffn(6, self.embedding_dim, self.width, self.act, self.dropout)

        self.nn_pf_mha = nn.ModuleList()
        for i in range(4):
            self.nn_pf_mha.append(
                SelfAttentionLayer(embedding_dim=self.embedding_dim, width=self.width, dropout=self.dropout)
            )

        # self.nn_pf_ga = AttentionalAggregation(ffn(self.embedding_dim, 1, self.width, self.act, self.dropout))
        self.agg = torch_geometric.nn.MeanAggregation()

        self.nn_pred_istau = ffn(4 + self.embedding_dim, 1, self.width, self.act, self.dropout)
        self.nn_pred_p4 = ffn(4 + self.embedding_dim, 4, self.width, self.act, self.dropout)

    def forward(self, batch):
        pf_encoded = self.nn_pf_initialembedding(batch.jet_pf_features)

        # pad the PF tensors across jets
        pfs_padded, mask = torch_geometric.utils.to_dense_batch(pf_encoded, batch.jet_pf_features_batch, 0.0)

        # run a simple self-attention over the PF candidates in each jet
        for mha_layer in self.nn_pf_mha:
            pfs_padded = mha_layer(pfs_padded, ~mask)

        # get the encoded PF candidates after attention, undo the padding
        pf_encoded = torch.cat([pfs_padded[i][mask[i]] for i in range(pfs_padded.shape[0])])
        assert pf_encoded.shape[0] == batch.jet_pf_features.shape[0]

        # now collapse the PF information in each jet with a global attention layer
        jet_encoded2 = self.agg(pf_encoded, batch.jet_pf_features_batch)

        # get the list of per-jet features as a concat of
        # (original_features, multi-head attention features)
        jet_feats = torch.cat([batch.jet_features, jet_encoded2], axis=-1)

        # run a binary classification whether or not this jet is from a tau
        pred_istau = torch.sigmoid(self.nn_pred_istau(jet_feats)).squeeze(-1)

        # run a per-jet NN for visible energy prediction
        pred_p4 = self.nn_pred_p4(jet_feats)

        return pred_istau, pred_p4


def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev):
    loss_tot = 0.0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    # loop over batches in data
    for batch in ds_loader:
        batch = batch.to(device=dev)
        pred_istau, pred_p4 = model(batch)
        true_p4 = batch.gen_tau_p4
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        loss_p4 = torch.nn.functional.mse_loss(pred_p4 * true_istau.unsqueeze(-1), true_p4 * true_istau.unsqueeze(-1))
        loss_cls = 10000.0 * torch.nn.functional.binary_cross_entropy(pred_istau, true_istau)

        loss = loss_cls + loss_p4
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        loss_tot += loss.detach().cpu().item()
        nsteps += 1
        njets += batch.jet_features.shape[0]
        print(
            batch.jet_features.shape[0],
            batch.jet_pf_features.shape[0],
            np.max(np.unique(batch.jet_pf_features_batch.cpu().numpy(), return_counts=True)[1]),
            true_istau.sum().cpu().item(),
            loss.detach().cpu().item(),
            scheduler.get_last_lr(),
        )
        sys.stdout.flush()
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
        pred_istau, pred_p4 = self.model(data_obj)

        pred_istau = pred_istau.detach().numpy()
        pred_p4 = pred_p4.detach().numpy()

        njets = len(jets["reco_jet_p4s"]["x"])
        assert njets == len(pred_istau)
        assert njets == len(pred_p4)

        tauP4 = vector.awk(
            ak.zip(
                {
                    "px": pred_p4[:, 0],
                    "py": pred_p4[:, 1],
                    "pz": pred_p4[:, 2],
                    "tau": pred_p4[:, 3],
                }
            )
        )

        # dummy placeholders for now
        tauCharges = np.zeros(njets)
        dmode = np.zeros(njets)

        # as a dummy placeholder, just return the first PFCand for each jet
        tau_cand_p4s = jets["reco_cand_p4s"][:, 0]

        return {
            "tau_p4s": tauP4,
            "tauSigCand_p4s": tau_cand_p4s,
            "tauClassifier": pred_istau,
            "tau_charge": tauCharges,
            "tau_decaymode": dmode,
        }


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]

        # FIXME: this is currently hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch-persistent for GPU training
        paths = [p.replace("/scratch-persistent/laurits/", "./data/") for p in paths]
        random.shuffle(paths)
        return paths


@hydra.main(config_path="../config", config_name="endtoend_simple", version_base=None)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]

    files_train = get_split_files(cfg.train_files, "train")
    files_val = get_split_files(cfg.validation_files, "validation")

    ds_train = TauJetDataset(files_train)
    ds_val = TauJetDataset(files_val)

    print("Loaded TauJetDataset with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDataset with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, follow_batch=["jet_pf_features"])
    ds_val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=True, follow_batch=["jet_pf_features"])

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.0001, steps_per_epoch=len(ds_train_loader), epochs=cfg.epochs
    )

    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")

    for iepoch in range(cfg.epochs):
        loss_train = model_loop(model, ds_train_loader, optimizer, scheduler, True, dev)
        tensorboard_writer.add_scalar("epoch/train_loss", loss_train, iepoch)
        loss_val = model_loop(model, ds_val_loader, optimizer, scheduler, False, dev)
        tensorboard_writer.add_scalar("epoch/val_loss", loss_val, iepoch)
        print("epoch={} loss_train={:.4f} loss_val={:.4f}".format(iepoch, loss_train, loss_val))
        torch.save(model, "{}/model_{}.pt".format(outpath, iepoch))

    model = model.to(device="cpu")
    torch.save(model, "data/model.pt")


if __name__ == "__main__":
    main()
