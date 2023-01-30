import os
import hydra
import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.loader import DataLoader
from taujetdataset import TauJetDataset

from torch.utils.data import Subset
from torch_geometric.nn import GlobalAttention


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

        self.nn_pf_ga = GlobalAttention(
            nn.Sequential(
                nn.Linear(embedding_dim, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, width),
                self.act(),
                nn.Linear(width, embedding_dim),
            )
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

        # run a per-jet NN for visible energy prediction
        pred_visenergy = self.nn_pred_visenergy(jet_feats)

        return pred_visenergy


def model_loop(model, ds_loader, optimizer, is_train, dev):
    loss_tot = 0.0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    for batch in ds_loader:
        batch = batch.to(device=dev)
        pred_visenergy = model(batch)[:, 0]
        true_visenergy = batch.gen_tau_vis_energy
        loss = torch.nn.functional.mse_loss(pred_visenergy, true_visenergy)
        if is_train:
            loss.backward()
            optimizer.step()
        loss_tot += loss.detach().cpu().item()
        nsteps += 1
        njets += batch.jet_features.shape[0]
    return loss_tot / njets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="../config", config_name="endtoend_simple", version_base=None)
def main(cfg):
    ds = TauJetDataset(cfg.input_dir)
    ds_train = Subset(ds, range(0, cfg.ntrain))
    ds_val = Subset(ds, range(cfg.ntrain, cfg.ntrain + cfg.ntest))

    print("Loaded TauJetDataset with {} files".format(len(ds)))
    ds_train_loader = DataLoader(ds_train, batch_size=1)
    ds_val_loader = DataLoader(ds_val, batch_size=1)

    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = TauEndToEndSimple().to(device=dev)
    print("params={}".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

    for iepoch in range(cfg.epochs):
        loss_train = model_loop(model, ds_train_loader, optimizer, True, dev)
        loss_val = model_loop(model, ds_val_loader, optimizer, False, dev)
        print("epoch={} loss_train={:.4f} loss_val={:.4f}".format(iepoch, loss_train, loss_val))


if __name__ == "__main__":
    main()
