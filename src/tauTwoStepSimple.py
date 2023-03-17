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
import random
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
        act(),
        nn.Dropout(dropout),
        torch.nn.LayerNorm(width),
        nn.Linear(width, output_dim),
    )


class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]), 
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )

    def forward(self, x, edge_index, k):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, edge_index, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)

settings = {
    "conv_params": [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 128, 20)),
    ],
    "fc_params": [
        (0.1, 256)
    ],
    "input_features": 10,
    "output_classes": 2,
}

class TauTwoStepSimple(nn.Module):
    def __init__(
        self, extras,
    ):
        super(TauTwoStepSimple, self).__init__()

        ### gnn test
        previous_output_shape = settings['input_features']

        #self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])
        self.input_bn = torch.nn.BatchNorm1d(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K))
            previous_output_shape = channels[-1]
        self.nn_encode = ffn(20*(20+settings['input_features']), 64, 128,nn.ELU, 0.2)
        self.nn_pred_istau = ffn(64, 1, 64, nn.ELU, 0.2)
        self.nn_pred_p4 = ffn(64, 4, 64, nn.ELU, 0.2)
        ###
        self.extras = extras
    def forward(self, batch):
        ### gnn test
        #print(batch.gnnfeats.shape)
        #fts = self.input_bn(batch.gnnfeats)
        #pts = batch.gnnpos
        fts = self.input_bn(batch.gnnfeats[0,:])
        pts = batch.gnnpos[0,:]
        for idx, layer in enumerate(self.conv_process):
          #fts = layer(pts, fts, batch.gnnfeats_batch)
          fts = layer(pts, fts, batch.batch)
          pts = fts

        #pred_from_tau = torch.sigmoid(torch_geometric.nn.global_mean_pool(fts, batch.gnnfeats_batch))
        pred_from_tau = torch.sigmoid(torch_geometric.nn.global_mean_pool(fts, batch.batch))
        next_feats = torch.flatten(torch.cat([batch.gnnfeats[0,:], fts], axis=-1))
        nn_encode = self.nn_encode(next_feats)
        pred_istau = self.nn_pred_istau(nn_encode)
        pred_p4 = self.nn_pred_p4(nn_encode)
        return pred_from_tau, pred_istau, pred_p4

def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev, batch_size):
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
    count = batch_size
    njets_batch = 0
    pfs_batch = 0
    loss_batch = 0
    loss_p4_batch = 0
    loss_cls_batch = 0
    loss_pf_batch = 0
    maxpfs_batch = 0
    for batch in ds_loader:
        batch = batch.to(device=dev)
        pred_fromtau, pred_istau, pred_p4 = model(batch)
        pred_fromtau = torch.flatten(pred_fromtau)
        true_fromtau = torch.flatten(batch.gnnfracs)
        true_p4 = batch.gen_tau_p4
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        perjetweight = batch.perjet_weight
        loss_p4_f = torch.nn.HuberLoss(reduction="none")
        loss_p4 = 5 * torch.sum(
            torch.sum(loss_p4_f(pred_p4[true_istau[0] == 1], true_p4[true_istau == 1]), axis=1)
        )
        loss_cls = 200 *  torch.nn.functional.binary_cross_entropy_with_logits(
            pred_istau, true_istau, weight=perjetweight
        )
        weights_for_pf2 = torch.flatten(batch.gnnweights)
        lossf_pf = torch.nn.BCELoss(reduction="none")
        loss_pf = torch.flatten(lossf_pf(pred_fromtau, true_fromtau))
        weighted_loss_pf = torch.sum(weights_for_pf2 * loss_pf)
        loss = (loss_p4 + loss_cls + weighted_loss_pf) / batch.gnnfeats.shape[0]
        njets += batch.gnnfeats.shape[0]
        njets_batch += batch.gnnfeats.shape[0]
        pfs_batch += batch.gnnfeats.shape[1]
        loss_batch += loss.detach().cpu().item()
        loss_p4_batch += loss_p4 / batch.gnnfeats.shape[0]
        loss_cls_batch += loss_cls / batch.gnnfeats.shape[0]
        loss_pf_batch += weighted_loss_pf / batch.gnnfeats.shape[0]
        loss_tot += loss.detach().cpu().item()
        loss_p4_tot += loss_p4.detach().cpu().item() / batch.gnnfeats.shape[0]
        loss_cls_tot += loss_cls.detach().cpu().item() / batch.gnnfeats.shape[0]
        loss_pf_tot += weighted_loss_pf.detach().cpu().item() / batch.gnnfeats.shape[0]
        if is_train:
            loss.backward()
            if count == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                nsteps += 1
        else:
            class_true.append(true_istau.cpu().numpy())
            class_pred.append(torch.sigmoid(pred_istau).detach().cpu().numpy())
        if count == 0:
            print(
                "jets={jets} pfs={pfs} loss={loss:.2f} loss_cls={loss_cls:.2f} loss_p4={loss_p4:.2f} loss_pf={loss_pf:.2f} lr={lr:.2E}".format(
                    jets=njets_batch,
                    pfs= pfs_batch,
                    loss=loss_batch / njets_batch,
                    loss_cls=loss_cls_batch / njets_batch,
                    loss_p4=loss_p4_batch / njets_batch,
                    loss_pf=loss_pf_batch / njets_batch,
                    lr=scheduler.get_last_lr()[0],
                )
            )
            njets_batch = 0
            pfs_batch = 0
            loss_batch = 0
            loss_p4_batch = 0
            loss_cls_batch = 0
            loss_pf_batch = 0
            maxpfs_batch = 0
            count = batch_size
        count -= 1
        sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return loss_tot / njets, loss_p4_tot / njets, loss_pf_tot / njets, loss_cls_tot / njets, (class_true, class_pred)


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        newpaths = []
        for p in paths:
            # if 'ZH' in p:
            newpaths.append(p)
        #random.shuffle(newpaths)
        return newpaths


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="../config", config_name="twostep_simple", version_base=None)
def main(cfg):

    outpath = cfg.outpath
    files_train = get_split_files(cfg.train_files, "train")
    files_val = get_split_files(cfg.validation_files, "validation")
    ds_train = TauJetDataset(files_train)
    ds_val = TauJetDataset(files_val)

    print("Loaded TauJetDataset with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDataset with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, follow_batch=["gnnfeats"])
    ds_val_loader = DataLoader(ds_val, batch_size=1, shuffle=True, follow_batch=["gnnfeats"])

    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = TauTwoStepSimple(extras=cfg.extras).to(device=dev)
    print("params={}".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=len(ds_train_loader), epochs=cfg.epochs
    )

    tensorboard_writer = SummaryWriter(outpath + "/tensorboard/"+cfg.runName)

    for iepoch in range(cfg.epochs):
        loss_train, loss_train_p4, loss_train_pf, loss_train_cls, _ = model_loop(
            model, ds_train_loader, optimizer, scheduler, True, dev, cfg.batch_size
        )
        tensorboard_writer.add_scalar("epoch/train_loss", loss_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_p4", loss_train_p4, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_pf", loss_train_pf, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_cls", loss_train_cls, iepoch)
        loss_val, loss_val_p4, loss_val_pf, loss_val_cls, retvals = model_loop(
            model, ds_val_loader, optimizer, scheduler, False, dev, cfg.batch_size
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
        data_obj = Batch.from_data_list(ds.process_file_data(jets), follow_batch=["gnnfeats"])
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
