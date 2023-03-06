from taujetdataset_twostep import TauJetDataset
import hydra
import vector
import awkward as ak
import numpy as np
import yaml
import torch
import torch_geometric
import torch.nn as nn
import sys
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch

from basicTauBuilder import BasicTauBuilder

# from omegaconf import DictConfig, OmegaConf
# from hydra import initialize, compose
import os

from torch.utils.tensorboard import SummaryWriter

from torch_geometric.nn.aggr import AttentionalAggregation


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


class TauTwoStepSimple(nn.Module):
    def __init__(
        self,
    ):
        super(TauTwoStepSimple, self).__init__()

        self.act = nn.ELU
        self.dropout = 0.2
        self.width = 128
        self.embedding_dim = 64

        self.nn_pf_initialembedding = ffn(11, self.embedding_dim, self.width, self.act, 0.0)

        self.nn_pf_mha = nn.ModuleList()
        for i in range(4):
            self.nn_pf_mha.append(
                SelfAttentionLayer(embedding_dim=self.embedding_dim, width=self.width, dropout=self.dropout)
            )

        self.nn_pf_ga = AttentionalAggregation(ffn(5 + self.embedding_dim, 1, self.width, self.act, self.dropout))
        self.agg = torch_geometric.nn.MeanAggregation()

        self.nn_pred_fromtau = ffn(4 + self.embedding_dim, 1, self.width, nn.ELU, self.droput)

        self.nn_pred_istau = ffn(2 * (4 + 1 + self.embedding_dim), 1, self.width, self.act, self.dropout)
        self.nn_pred_p4 = ffn(2 * (4 + 1 + self.embedding_dim), 4, self.width, self.act, self.dropout)

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

        # print(pf_encoded, batch.jet_features, pf_encoded.shape)
        ## now collapse the PF information in each jet with a global attention layer
        # jet_encoded1 = self.nn_pf_ga(pf_encoded, batch.jet_pf_features_batch)

        ## also compute a simple mean aggregation
        # jet_encoded2 = self.agg(pf_encoded, batch.jet_pf_features_batch)

        # print(batch.jet_features.shape, pf_encoded.shape)
        # print(np.unique(batch.jet_pf_features.numpy()))
        # ones_tensor = torch.ones((pf_encoded.shape[0],1))
        npfPerJet = np.sum(mask.cpu().numpy(), axis=1)
        perjet_feats = []
        for i in range(len(npfPerJet)):
            perjet_feats.append(torch.ones((npfPerJet[i], 1)).to(device=pf_encoded.device) * batch.jet_features[i])
        jet_feats = torch.cat([torch.cat(perjet_feats, axis=0), pf_encoded], axis=-1)
        # get the list of per-jet features as a concat of
        # (original_features, multi-head attention features)
        # print(ones_tensor.device, batch.device)
        # jet_feats = torch.cat([torch.cat(perjet_feats, axis=0), pf_encoded.cpu()], axis=-1).to(device=batch.device)
        # print(jet_feats.device)
        # run a binary classification whether or not these pf cands are from a tau
        pred_fromtau = self.nn_pred_fromtau(jet_feats)  # .squeeze(-1)

        # combine this with general jet features and original pfcand info
        jet_feats2 = torch.cat([jet_feats, pred_fromtau], axis=-1)

        # now collapse the PF information in each jet with a global attention layer
        jet_encoded1 = self.nn_pf_ga(jet_feats2, batch.jet_pf_features_batch)

        # also compute a simple mean aggregation
        jet_encoded2 = self.agg(jet_feats2, batch.jet_pf_features_batch)

        # run a binary classification whether or not this jet is from a tau
        pred_istau = self.nn_pred_istau(torch.cat([jet_encoded1, jet_encoded2], axis=-1)).squeeze(-1)

        # run a per-jet NN for visible energy prediction
        jet_p4 = batch.gen_tau_p4
        pred_p4 = jet_p4 * self.nn_pred_p4(torch.cat([jet_encoded1, jet_encoded2], axis=-1))
        # print(pred_istau.shape,pred_p4.shape, pred_fromtau.shape, batch.jet_pf_efrac.shape )
        ### run a per-jet NN for visible energy prediction
        # jet_p4 = batch.jet_features[:, :4]
        # pred_p4 = jet_p4 * self.nn_pred_p4(jet_feats)

        return pred_fromtau, pred_istau, pred_p4


def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev):
    # loss_cls_tot = 0.0
    # loss_p4_tot = 0.0
    loss_tot = 0
    loss_cls_tot = 0
    loss_p4_tot = 0
    loss_pf_tot = 0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    # loop over batches in data

    class_true = []
    class_pred = []
    for batch in ds_loader:
        batch = batch.to(device=dev)
        pred_fromtau, pred_istau, pred_p4 = model(batch)
        true_fromtau = batch.jet_pf_efrac
        true_p4 = batch.gen_tau_p4
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        perjetweight = batch.perjet_weight
        loss_p4_f = torch.nn.HuberLoss(reduction="none")
        loss_p4 = 0.5 * torch.sum(
            perjetweight[true_istau == 1] * torch.sum(loss_p4_f(pred_p4[true_istau == 1], true_p4[true_istau == 1]), axis=1)
        )
        loss_cls = 20000.0 * torch.nn.functional.binary_cross_entropy_with_logits(
            pred_istau, true_istau, weight=perjetweight
        )
        weights_for_pf = batch.pf_weights
        # torch.nn.MSELoss
        # print(pred_fromtau.shape,true_fromtau.shape)
        # print(pred_fromtau,true_fromtau)
        lossf_pf = torch.nn.MSELoss(reduction="none")
        loss_pf = lossf_pf(pred_fromtau, true_fromtau)
        weighted_loss_pf = 3 * torch.sum(weights_for_pf * loss_pf)
        # loss = loss_p4 + loss_cls + weighted_loss_pf
        # loss = weighted_loss_pf
        loss = loss_p4 + loss_cls + weighted_loss_pf
        # loss_p4 = torch.nn.functional.huber_loss(pred_p4[true_istau == 1], true_p4[true_istau == 1])
        # loss_cls = 10000.0 * torch.nn.functional.binary_cross_entropy_with_logits(pred_istau, true_istau)
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            class_true.append(true_istau.cpu().numpy())
            class_pred.append(torch.sigmoid(pred_istau).detach().cpu().numpy())
        # loss_cls_tot += loss_cls.detach().cpu().item()
        # loss_p4_tot += loss_p4.detach().cpu().item()
        loss_tot += loss.detach().cpu().item()
        loss_p4_tot += loss_p4.detach().cpu().item()
        loss_cls_tot += loss_cls.detach().cpu().item()
        loss_pf_tot += weighted_loss_pf.detach().cpu().item()
        nsteps += 1
        njets += batch.jet_features.shape[0]
        print(
            "jets={jets} pfs={pfs} max_pfs={max_pfs} loss={loss:.2f} loss_cls={loss_cls:.2f} loss_p4={loss_p4:.2f} loss_pf={loss_pf:.2f} lr={lr:.2E}".format(
                jets=batch.jet_features.shape[0],
                pfs=batch.jet_pf_features.shape[0],
                max_pfs=np.max(np.unique(batch.jet_pf_features_batch.cpu().numpy(), return_counts=True)[1]),
                loss=loss.detach().cpu().item() / batch.jet_features.shape[0],
                loss_cls=loss_cls / batch.jet_features.shape[0],
                loss_p4=loss_p4 / batch.jet_features.shape[0],
                loss_pf=weighted_loss_pf / batch.jet_features.shape[0],
                lr=scheduler.get_last_lr()[0],
            )
        )
        sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    # return loss_cls_tot / njets, loss_p4_tot / njets, (class_true, class_pred)
    return loss_tot / njets, loss_p4_tot / njets, loss_pf_tot / njets, loss_cls_tot / njets, (class_true, class_pred)


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        newpaths = []
        for p in paths:
            # if 'ZH' in p:
            newpaths.append(p)
        return newpaths


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="../config", config_name="twostep_simple", version_base=None)
def main(cfg):

    # hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = cfg.outpath

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

    model = TauTwoStepSimple().to(device=dev)
    print("params={}".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(ds_train_loader), epochs=cfg.epochs
    )

    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")

    for iepoch in range(cfg.epochs):
        loss_train, loss_train_p4, loss_train_pf, loss_train_cls, _ = model_loop(
            model, ds_train_loader, optimizer, scheduler, True, dev
        )
        tensorboard_writer.add_scalar("epoch/train_loss", loss_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_p4", loss_train_p4, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_pf", loss_train_pf, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_cls", loss_train_cls, iepoch)
        loss_val, loss_val_p4, loss_val_pf, loss_val_cls, retvals = model_loop(
            model, ds_val_loader, optimizer, scheduler, False, dev
        )

        tensorboard_writer.add_scalar("epoch/val_loss", loss_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_p4", loss_val_p4, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_pf", loss_val_pf, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_cls", loss_val_cls, iepoch)

        tensorboard_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], iepoch)
        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1], iepoch)

        print(
            "epoch={} loss={:.4f}/{:.4f} loss_cls={:.4f}/{:.4f} loss_p4={:.4f}/{:.4f} loss_pf={:.4f}/{:.4f}".format(
                iepoch,
                loss_train,
                loss_val,
                loss_train_cls,
                loss_val_cls,
                loss_train_p4,
                loss_val_p4,
                loss_train_pf,
                loss_val_pf,
            )
        )
        torch.save(model, "{}/model_{}.pt".format(outpath, iepoch))

    model = model.to(device="cpu")
    torch.save(model, "model.pt")
    return model


if __name__ == "__main__":
    main()


class TwoStepDNNTauBuilder(BasicTauBuilder):
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
        data_obj = Batch.from_data_list(ds.process_file_data(jets), follow_batch=["jet_pf_features"])
        pred_fromtau, pred_istau, pred_p4 = self.model(data_obj)

        pred_istau = torch.sigmoid(pred_istau)
        pred_istau = pred_istau.contiguous().detach().numpy()
        # to solve "ValueError: ndarray is not contiguous"
        pred_p4 = np.asfortranarray(pred_p4.detach().contiguous().numpy())

        njets = len(jets["reco_jet_p4s"]["x"])
        assert njets == len(pred_istau)
        assert njets == len(pred_p4)

        tauP4 = vector.awk(
            ak.zip(
                {
                    "px": pred_p4[:, 0],
                    "py": pred_p4[:, 1],
                    "pz": pred_p4[:, 2],
                    "mass": pred_p4[:, 3],
                }
            )
        )

        # dummy placeholders for now
        tauCharges = np.zeros(njets)
        dmode = np.zeros(njets)

        # as a dummy placeholder, just return the first PFCand for each jet
        tau_cand_p4s = jets["reco_cand_p4s"][:, 0:1]
        return {
            "tau_p4s": tauP4,
            "tauSigCand_p4s": tau_cand_p4s,
            "tauClassifier": pred_istau,
            "tau_charge": tauCharges,
            "tau_decaymode": dmode,
        }
