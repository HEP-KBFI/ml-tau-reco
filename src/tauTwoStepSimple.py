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
import mplhep as hep
hep.style.use(hep.styles.CMS)
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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


# settings = {
#     "conv_params": [
#         (16, (64, 64, 64)),
#         (16, (128, 128, 256)),
#         (16, (256, 256, 256)),
#         (16, (256, 128, 20)),
#     ],
#     "fc_params": [
#         (0.1, 256)
#     ],
#     "input_features": 14,
#     "output_classes": 2,
# }

class TauTwoStepSimple(nn.Module):
    def __init__(
        self, extras,
    ):
        super(TauTwoStepSimple, self).__init__()
        self.settings = {
            "conv_params": [
                (2, (64, 64, 64)),
                (4, (128, 128, 256)),
                (8, (256, 256, 256)),
                (16, (256, 256, 256)),
                (20, (256, 128, 20)),
            ],
            "fc_params": [
                (0.1, 256)
            ],
            "input_features": 14,
            "output_classes": 2,
        }
        if "TESTF" in extras: self.settings["input_features"]=17
        ### gnn test
        previous_output_shape = self.settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(self.settings['input_features'])
        #self.input_bn = torch.nn.LayerNorm(settings['input_features'])

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(self.settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K))
            previous_output_shape = channels[-1]
        self.nn_encode = ffn(20*(20+self.settings['input_features']), 128, 128,nn.ELU, 0.2)
        self.nn_pred_istau = ffn(128, 1, 128, nn.ELU, 0.2)
        self.nn_pred_p4 = ffn(128, 4, 128, nn.ELU, 0.2)
        if 'NOHOTMODE' not in extras:
            self.nn_pred_dmode = ffn(128, 6, 128, nn.ELU, 0.2)
            self.nn_pred_charge = ffn(128, 2, 128, nn.ELU, 0.2)
        else:
            self.nn_pred_dmode = ffn(128, 1, 128, nn.ELU, 0.2)
            self.nn_pred_charge = ffn(128, 1, 128, nn.ELU, 0.2)
        #self.nn_pred_fromTau2 = ffn(128, 20, 128, nn.ELU, 0.2)
        self.nn_pred_fromTau2 = ffn(128, 20, 256, nn.ELU, 0.2)
        self.nn_pred_fromTau = nn.Sequential(
            torch.nn.LayerNorm(20),
            nn.Linear(20, 128),
            nn.ELU(),
            torch.nn.LayerNorm(128),
            nn.Linear(128, 20),
            nn.Sigmoid()
        )
        ###
        self.extras = extras
    def forward(self, batch):
        ### gnn test
        #print(batch.gnnfeats.shape)
        fts = None
        if 'TESTF' in self.extras:
            fts = self.input_bn(batch.gnnfeats_test)
        else:
            fts = self.input_bn(batch.gnnfeats)
        pts = batch.gnnpos
        #fts = self.input_bn(batch.gnnfeats[0,:])
        #pts = batch.gnnpos[0,:]
        for idx, layer in enumerate(self.conv_process):
          #fts = layer(pts, fts)
          fts = layer(pts, fts, batch.gnnfeats_batch)
          pts = fts

        #pred_from_tau = torch.sigmoid(torch.mean(fts, axis=1))
        #print(pred_from_tau.shape)
        a = None
        if 'TESTF' in self.extras:
            a = torch_geometric.utils.to_dense_batch(batch.gnnfeats_test, batch.gnnfeats_batch)[0]
        else:
            a = torch_geometric.utils.to_dense_batch(batch.gnnfeats, batch.gnnfeats_batch)[0]
        b = torch_geometric.utils.to_dense_batch(fts, batch.gnnfeats_batch)[0]
        next_feats = torch.flatten(torch.cat([a,b], axis=2),start_dim = 1)
        nn_encode = self.nn_encode(next_feats)
        pred_istau = torch.sigmoid(self.nn_pred_istau(nn_encode))
        pred_p4 = 200 * self.nn_pred_p4(nn_encode)
        pred_charge = self.nn_pred_charge(nn_encode)
        pred_dmode = self.nn_pred_dmode(nn_encode)
        if 'NOHOTMODE' not in self.extras:
            pred_dmode = torch.softmax(pred_dmode,1)
            pred_charge = torch.softmax(pred_charge,1)
        pred_from_tau = torch.sigmoid(torch_geometric.nn.global_max_pool(fts, batch.gnnfeats_batch))
        if 'addNN' in self.extras:
            pred_from_tau = self.nn_pred_fromTau(pred_from_tau)
        elif 'PFASP4' in self.extras:
            pred_from_tau = torch.sigmoid(self.nn_pred_fromTau2(nn_encode))
        return pred_from_tau, pred_istau, pred_p4, pred_charge, pred_dmode
        #    batch_size x 20, batch_size x 1, batch_size x 4


def dmodeToHot(dmode):
    if dmode <0 : dmode = np.random.choice([0.,1.,2.,10.,11.,15.])
    if dmode == 0.: return [1.,0.,0.,0.,0.,0.]
    elif dmode == 1.: return [0.,1.,0.,0.,0.,0.]
    elif dmode == 2.: return [0.,0.,1.,0.,0.,0.]
    elif dmode == 10.: return [0.,0.,0.,1.,0.,0.]
    elif dmode == 11.: return [0.,0.,0.,0.,1.,0.]
    else: return [0.,0.,0.,0.,0.,1.]

def hotToDmode(dmode):
    dmode = np.argmax(dmode)
    if dmode == 0: return 0
    elif dmode == 1: return 1
    elif dmode == 2: return 2
    elif dmode == 3: return 10
    elif dmode == 4: return 11
    else: return 15

def hotToCharge(charge):
    charge = np.argmax(charge)
    if charge == 0: return -1
    else: return 1


def chargeToHot(charge):
    if charge == 0. :
        charge = np.random.choice([-1.,1.])
    if charge < 0. : return [0.,1.]
    else: return [1.,0.]



def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev, batch_size, extras):
    loss_tot = 0
    loss_cls_tot = 0
    loss_p4_tot = 0
    loss_pf_tot = 0
    loss_dmode_tot = 0
    loss_charge_tot = 0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    # loop over batches in data

    class_true = []
    class_pred = []
    pffrac_true = []
    pffrac_pred = []
    jetE_true = []
    jetE_pred = []
    count = 0 # batch_size
    njets_batch = 0
    pfs_batch = 0
    loss_batch = 0
    loss_p4_batch = 0
    loss_cls_batch = 0
    loss_pf_batch = 0
    loss_dmode_batch = 0
    loss_charge_batch = 0
    maxpfs_batch = 0
    pred_sig = []
    pred_bkg = []
    dmode_true = []
    dmode_pred = []
    charge_true = []
    charge_pred = []
    for batch in ds_loader:
        batch = batch.to(device=dev)
        if len(batch.gen_tau_decaymode)<1: continue
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        true_fromtau = torch.flatten(batch.gnnfracs)
        if 'PFCLS' in extras:
            true_fromtau = torch.ones_like(true_fromtau) * (true_fromtau >0.5).to(dtype=torch.float32)
        true_p4 = batch.gen_tau_p4 * true_istau.unsqueeze(-1)

        perjetweight = batch.perjet_weight
        weights_for_pf = torch.flatten(batch.gnnweights)
        if 'TESTW' in extras:
            weights_for_pf = torch.flatten(batch.gnnweights_test)
        pred_fromtau, pred_istau, pred_p4, pred_charge, pred_dmode = model(batch)
        pred_p4 = pred_p4 * true_istau.unsqueeze(-1)
        pred_fromtau = torch.flatten(pred_fromtau)
        loss_huber_f_p4 = torch.nn.HuberLoss(delta = 20)
        loss_p4 = 0.1 * loss_huber_f_p4 (input=pred_p4, target=true_p4)
        loss_cls_f = torch.nn.BCELoss( weight=perjetweight)
        loss_cls = 200 *  loss_cls_f(
            pred_istau.squeeze(), true_istau
        )
        lossf_pf = torch.nn.BCELoss(weight = weights_for_pf)
        if "NOPFWEIGHT" in extras:
            lossf_pf = torch.nn.BCELoss()
        weighted_loss_pf = 5000*lossf_pf(pred_fromtau, true_fromtau)
        loss_dmode = None
        loss_charge = None
        if 'NOHOTMODE' not in extras:
            CELoss= torch.nn.CrossEntropyLoss(reduction='none')
            true_dmode = torch.tensor([dmodeToHot(d) for d in batch.tau_decaymode]).to(device=dev)
            loss_dmode = 0.2 * torch.sum(CELoss(true_dmode, pred_dmode)*true_istau)
            true_charge = torch.tensor([chargeToHot(d) for d in batch.tau_charge]).to(device=dev)
            loss_charge = 0.4 * torch.sum(CELoss(true_charge, pred_charge)*true_istau)
        else:
            loss_huber_f_other = torch.nn.HuberLoss(delta = 0.1)
            true_dmode = batch.tau_decaymode * true_istau
            pred_dmode = pred_charge * true_istau.unsqueeze(-1)
            loss_dmode = 0.5 * loss_huber_f_other (input=pred_dmode.squeeze(-1), target=true_dmode)
            true_charge = batch.tau_charge * true_istau
            pred_charge = pred_charge * true_istau.unsqueeze(-1)
            loss_charge = 0.5 * loss_huber_f_other (input=pred_charge.squeeze(-1), target=true_charge)
        #weighted_loss_pf = torch.sum(weights_for_pf2 * loss_pf)
        loss = (loss_p4 + loss_cls + weighted_loss_pf + loss_charge + loss_dmode) #/ batch.gnnfeats.shape[0]
        njets += true_istau.shape[0]
        njets_batch += true_istau.shape[0]
        #pfs_batch += batch.gnnfeats.shape[1]
        loss_batch += loss.detach().cpu().item()
        loss_p4_batch += loss_p4 #/ batch.gnnfeats.shape[0]
        loss_charge_batch += loss_charge #/ batch.gnnfeats.shape[0]
        loss_dmode_batch += loss_dmode.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        loss_cls_batch += loss_cls #/ batch.gnnfeats.shape[0]
        loss_pf_batch += weighted_loss_pf #/ batch.gnnfeats.shape[0]
        loss_tot += loss.detach().cpu().item()
        loss_p4_tot += loss_p4.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        loss_charge_tot += loss_charge.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        loss_dmode_tot += loss_dmode.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        loss_cls_tot += loss_cls.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        loss_pf_tot += weighted_loss_pf.detach().cpu().item() #/ batch.gnnfeats.shape[0]
        if is_train:
            loss.backward()
            if count == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                nsteps += 1
        else:
            class_true.append(true_istau.cpu().numpy())
            class_pred.append(pred_istau.detach().cpu().numpy())
            pffrac_pred.append(torch.flatten(pred_fromtau).detach().cpu().numpy())
            pffrac_true.append(torch.flatten(true_fromtau).detach().cpu().numpy())
            pred_sig.append(pred_istau[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy())
            pred_bkg.append(pred_istau[(batch.gen_tau_decaymode == -1)].detach().cpu().numpy())
            if 'NOHOTMODE' not in extras:
                dmode_true.append(np.array([hotToDmode(d) for d in true_dmode[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy()]))
                dmode_pred.append(np.array([hotToDmode(d) for d in pred_dmode[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy()]))
                charge_true.append(np.array([hotToCharge(d) for d in true_charge[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy()]))
                charge_pred.append(np.array([hotToCharge(d) for d in pred_charge[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy()]))
            else:
                dmode_true.append(true_dmode[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy())
                dmode_pred.append(pred_dmode[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy())
                charge_true.append(true_charge[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy())
                charge_pred.append(pred_charge[(batch.gen_tau_decaymode != -1)].detach().cpu().numpy())
            pred_p4_ = ak.Array(pred_p4.detach().contiguous().cpu().numpy())
            true_p4_ = ak.Array(true_p4.detach().contiguous().cpu().numpy())
            tauP4_pred = vector.awk(
                ak.zip(
                    {
                        "px": pred_p4_[:, 0],
                        "py": pred_p4_[:, 1],
                        "pz": pred_p4_[:, 2],
                        "energy": pred_p4_[:, 3],
                    }
                )
            )
            tauP4_true = vector.awk(
                ak.zip(
                    {
                        "px": true_p4_[:, 0],
                        "py": true_p4_[:, 1],
                        "pz": true_p4_[:, 2],
                        "energy": true_p4_[:, 3],
                    }
                )
            )
            jetE_true.append(tauP4_true.energy.to_numpy())
            jetE_pred.append(tauP4_pred.energy.to_numpy())

        if count == 0:
            print(
                "jets={jets} pfs={pfs} loss={loss:.2f} loss_cls={loss_cls:.2f} loss_p4={loss_p4:.2f} loss_pf={loss_pf:.2f} loss_dmode={loss_dmode:.2f} loss_charge={loss_charge:.2f} lr={lr:.2E}".format(
                    jets=njets_batch,
                    pfs= pfs_batch,
                    loss=loss_batch / njets_batch,
                    loss_cls=loss_cls_batch / njets_batch,
                    loss_p4=loss_p4_batch / njets_batch,
                    loss_pf=loss_pf_batch / njets_batch,
                    loss_dmode = loss_dmode_batch / njets_batch,
                    loss_charge = loss_charge_batch / njets_batch,
                    lr=scheduler.get_last_lr()[0],
                )
            )
            njets_batch = 0
            pfs_batch = 0
            loss_batch = 0
            loss_p4_batch = 0
            loss_charge_batch = 0
            loss_dmode_batch = 0
            loss_cls_batch = 0
            loss_pf_batch = 0
            maxpfs_batch = 0
            count = 0 #batch_size
        #count -= 1
        sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
        pffrac_pred = np.concatenate(pffrac_pred)
        pffrac_true = np.concatenate(pffrac_true)
        jetE_pred = np.concatenate(jetE_pred)
        jetE_true = np.concatenate(jetE_true)
        pred_sig = np.concatenate(pred_sig)
        pred_bkg = np.concatenate(pred_bkg)
        dmode_true = np.concatenate(dmode_true)
        dmode_pred = np.concatenate(dmode_pred)
        charge_true = np.concatenate(charge_true)
        charge_pred = np.concatenate(charge_pred)
    return loss_tot / njets, loss_p4_tot / njets, loss_pf_tot / njets, loss_cls_tot / njets,  loss_dmode_tot / njets,  loss_charge_tot / njets, (pffrac_true, pffrac_pred), (jetE_true, jetE_pred), (class_true, class_pred), (pred_sig, pred_bkg), (dmode_true,dmode_pred), (charge_true, charge_pred)


def get_split_files(config_path, split, maxFiles=-1):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        newpaths = []
        for p in paths:
            # if 'ZH' in p:
            newpaths.append(p)
        random.shuffle(newpaths)
        return newpaths[:maxFiles]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@hydra.main(config_path="../config", config_name="twostep_simple", version_base=None)
def main(cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    outpath = cfg.outpath
    #files_train = get_split_files(cfg.train_files, "train", cfg.maxTrainFiles)
    #files_val = get_split_files(cfg.validation_files, "validation", cfg.maxValFiles)
    #ds_train = TauJetDataset(files_train)
    #ds_val = TauJetDataset(files_val)
    #ds_train = torch.utils.data.ConcatDataset([ds_train_1, ds_train_2])
    ds_train = TauJetDataset("data_test/dataset_train")
    ds_val = TauJetDataset("data_test/dataset_validation")
    #ds_val = TauJetDataset("data/dataset_train")
    #print("Loaded TauJetDataset with {} train steps".format(len(ds_train)))
    #print("Loaded TauJetDataset with {} val steps".format(len(ds_val)))
    maxTrainFiles = len(ds_train)
    if cfg.maxTrainFiles > 0:
        maxTrainFiles = cfg.maxTrainFiles
    maxValFiles = len(ds_val)
    if cfg.maxValFiles > 0:
        maxValFiles = cfg.maxValFiles
    train_data = [ds_train[i] for i in range(maxTrainFiles)]
    train_data = sum(train_data, [])
    val_data = [ds_val[i] for i in range(maxValFiles)]
    val_data = sum(val_data, [])
    ds_train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, follow_batch=["gnnfeats"], num_workers=8)
    ds_val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=True, follow_batch=["gnnfeats"], num_workers=8)

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
        loss_train, loss_train_p4, loss_train_pf, loss_train_cls, loss_train_charge, loss_train_dmode, _1, _2, _3, _4, _5, _6 = model_loop(
            model, ds_train_loader, optimizer, scheduler, True, dev, cfg.batch_size, cfg.extras
        )
        tensorboard_writer.add_scalar("epoch/train_loss", loss_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_p4", loss_train_p4, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_pf", loss_train_pf, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_cls", loss_train_cls, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_dmode", loss_train_dmode, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss_charge", loss_train_charge, iepoch)
        loss_val, loss_val_p4, loss_val_pf, loss_val_cls, loss_val_dmode, loss_val_charge, retvals_pf, retvals_jet, retvals, predvals, retvals_dmode, retvals_charge = model_loop(
            model, ds_val_loader, optimizer, scheduler, False, dev, cfg.batch_size, cfg.extras
        )

        tensorboard_writer.add_scalar("epoch/val_loss", loss_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_p4", loss_val_p4, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_pf", loss_val_pf, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_cls", loss_val_cls, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_dmode", loss_val_dmode, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss_charge", loss_val_charge, iepoch)

        tensorboard_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], iepoch)

        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1].squeeze(), iepoch)

        if iepoch % 5 == 0:
            fig_pffracscatter = plt.figure()
            ax_pffracscatter = fig_pffracscatter.add_subplot(1, 1, 1)
            ax_pffracscatter.scatter(retvals_pf[0], retvals_pf[1])
            ax_pffracscatter.set_xlabel('true frac')
            ax_pffracscatter.set_ylabel('pred frac')
            tensorboard_writer.add_figure("pffrac_"+str(iepoch), fig_pffracscatter, global_step=iepoch, close=True, walltime=None)

            fig_jetEscatter = plt.figure()
            ax_jetEscatter = fig_jetEscatter.add_subplot(1, 1, 1)
            ax_jetEscatter.scatter(retvals_jet[0],retvals_jet[1])
            ax_jetEscatter.set_xlabel('true tau E')
            ax_jetEscatter.set_ylabel('pred tau E')
            tensorboard_writer.add_figure("jetE_"+str(iepoch), fig_jetEscatter, global_step=iepoch, close=True, walltime=None)

            fig_pffrachist = plt.figure()
            ax_pffrachist = fig_pffrachist.add_subplot(1, 1, 1)
            vals_pffrachist = ax_pffrachist.hist2d(retvals_pf[0], retvals_pf[1], cmap="inferno", norm="log", bins = 50)
            fig_pffrachist.colorbar(vals_pffrachist[3], ax=ax_pffrachist)
            ax_pffrachist.set_xlabel('true frac')
            ax_pffrachist.set_ylabel('pred frac')
            tensorboard_writer.add_figure("pffrac_hist_"+str(iepoch), fig_pffrachist, global_step=iepoch, close=True, walltime=None)

            fig_jetEhist = plt.figure()
            ax_jetEhist = fig_jetEhist.add_subplot(1, 1, 1)
            vals_jetEhist = ax_jetEhist.hist2d(retvals_jet[0],retvals_jet[1], cmap="inferno", norm="log", bins = 50)
            fig_jetEhist.colorbar(vals_jetEhist[3], ax=ax_jetEhist)
            ax_jetEhist.set_xlabel('true tau E')
            ax_jetEhist.set_ylabel('pred tau E')
            tensorboard_writer.add_figure("jetE_hist_"+str(iepoch), fig_jetEhist, global_step=iepoch, close=True, walltime=None)


            fig_predSigB = plt.figure()
            ax_predSigB = fig_predSigB.add_subplot(1, 1, 1)
            n,bins,patches = ax_predSigB.hist(predvals[0], density=True, label='Sig', color='b', bins=100, alpha=0.3, log=True)
            ax_predSigB.hist(predvals[1], density=True, label='Bkg', color='r', bins=bins, alpha=0.3, log=True)
            ax_predSigB.legend()
            ax_predSigB.set_xlabel("CLS output")
            tensorboard_writer.add_figure("classOut_"+str(iepoch), fig_predSigB, global_step=iepoch, close=True, walltime=None)

            mapping_dmode = {
                0: r"$\pi^{\pm}$",
                1: r"$\pi^{\pm}\pi^{0}$",
                2: r"$\pi^{\pm}\pi^{0}\pi^{0}$",
                10: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}$",
                11: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}\pi^{0}$",
                15: "Other",
            }
            cats_dmode = [value for value in mapping_dmode.values()]
            fig_dmode = plt.figure()
            ax_dmode = fig_dmode.add_subplot(1, 1, 1)
            confm_dmode = confusion_matrix(retvals_dmode[0], retvals_dmode[1], normalize="true", labels=[0,1,2,10,11,15])
            ax_dmode.set_aspect("equal", adjustable="box")
            xbins_dmode = ybins_dmode = np.arange(len(cats_dmode) + 1)
            tick_values_dmode = np.arange(len(cats_dmode)) + 0.5
            ax_dmode.set_xticks(tick_values_dmode, cats_dmode, fontsize=14, rotation=0)
            ax_dmode.set_yticks(tick_values_dmode + 0.2, cats_dmode, fontsize=14, rotation=90)
            hep.hist2dplot(confm_dmode, xbins_dmode, ybins_dmode, cmap='Greys', cbar=True, ax = ax_dmode)
            ax_dmode.set_xlabel('true dmode')
            ax_dmode.set_ylabel('pred dmode')
            ax_dmode.tick_params(axis="both", which="both", length=0)
            for i in range(len(ybins_dmode) - 1):
                for j in range(len(xbins_dmode) - 1):
                    bin_value = confm_dmode.T[i, j]
                    ax_dmode.text(
                        xbins_dmode[j] + 0.5,
                        ybins_dmode[i] + 0.5,
                        f"{bin_value:.2f}",
                        color='r',
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
            tensorboard_writer.add_figure("dmode_"+str(iepoch), fig_dmode, global_step=iepoch, close=True, walltime=None)

            fig_dmode2 = plt.figure()
            ax_dmode2 = fig_dmode2.add_subplot(1, 1, 1)
            confm_dmode2 = confusion_matrix(retvals_dmode[0], retvals_dmode[1], normalize="pred", labels=[0,1,2,10,11,15])
            ax_dmode2.set_aspect("equal", adjustable="box")
            xbins_dmode2 = ybins_dmode2 = np.arange(len(cats_dmode) + 1)
            tick_values_dmode2 = np.arange(len(cats_dmode)) + 0.5
            ax_dmode2.set_xticks(tick_values_dmode2, cats_dmode, fontsize=14, rotation=0)
            ax_dmode2.set_yticks(tick_values_dmode2 + 0.2, cats_dmode, fontsize=14, rotation=90)
            hep.hist2dplot(confm_dmode2, xbins_dmode2, ybins_dmode2, cmap='Greys', cbar=True, ax = ax_dmode2)
            ax_dmode2.set_xlabel('true dmode')
            ax_dmode2.set_ylabel('pred dmode')
            ax_dmode2.tick_params(axis="both", which="both", length=0)
            for i in range(len(ybins_dmode2) - 1):
                for j in range(len(xbins_dmode2) - 1):
                    bin_value = confm_dmode2.T[i, j]
                    ax_dmode2.text(
                        xbins_dmode2[j] + 0.5,
                        ybins_dmode2[i] + 0.5,
                        f"{bin_value:.2f}",
                        color='r',
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
            tensorboard_writer.add_figure("dmode2_"+str(iepoch), fig_dmode2, global_step=iepoch, close=True, walltime=None)

            cats_charge = [str(-1), str(1)]
            fig_charge = plt.figure()
            ax_charge = fig_charge.add_subplot(1, 1, 1)
            confm_charge = confusion_matrix(retvals_charge[0], retvals_charge[1], normalize="true", labels=[-1, 1])
            ax_charge.set_aspect("equal", adjustable="box")
            xbins_charge = ybins_charge = np.arange(len(cats_charge) + 1)
            tick_values_charge = np.arange(len(cats_charge)) + 0.5
            ax_charge.set_xticks(tick_values_charge, cats_charge, fontsize=14, rotation=0)
            ax_charge.set_yticks(tick_values_charge + 0.2, cats_charge, fontsize=14, rotation=90)
            hep.hist2dplot(confm_charge, xbins_charge, ybins_charge, cmap='Greys', cbar=True, ax = ax_charge)
            ax_charge.set_xlabel('true charge')
            ax_charge.set_ylabel('pred charge')
            ax_charge.tick_params(axis="both", which="both", length=0)
            for i in range(len(ybins_charge) - 1):
                for j in range(len(xbins_charge) - 1):
                    bin_value = confm_charge.T[i, j]
                    ax_charge.text(
                        xbins_charge[j] + 0.5,
                        ybins_charge[i] + 0.5,
                        f"{bin_value:.2f}",
                        color='r',
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
            tensorboard_writer.add_figure("charge_"+str(iepoch), fig_charge, global_step=iepoch, close=True, walltime=None)

            fig_charge2 = plt.figure()
            ax_charge2 = fig_charge2.add_subplot(1, 1, 1)
            confm_charge2 = confusion_matrix(retvals_charge[0], retvals_charge[1], normalize="pred", labels=[-1,1])
            ax_charge2.set_aspect("equal", adjustable="box")
            xbins_charge2 = ybins_charge2 = np.arange(len(cats_charge) + 1)
            hep.hist2dplot(confm_charge2, xbins_charge2, ybins_charge2, cmap='Greys', cbar=True, ax = ax_charge2)
            tick_values_charge2 = np.arange(len(cats_charge)) + 0.5
            ax_charge2.set_xticks(tick_values_charge2, cats_charge, fontsize=14, rotation=0)
            ax_charge2.set_yticks(tick_values_charge2 + 0.2, cats_charge, fontsize=14, rotation=90)
            ax_charge2.set_xlabel('true charge')
            ax_charge2.set_ylabel('pred charge')
            ax_charge2.tick_params(axis="both", which="both", length=0)
            for i in range(len(ybins_charge2) - 1):
                for j in range(len(xbins_charge2) - 1):
                    bin_value = confm_charge2.T[i, j]
                    ax_charge2.text(
                        xbins_charge2[j] + 0.5,
                        ybins_charge2[i] + 0.5,
                        f"{bin_value:.2f}",
                        color='r',
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
            tensorboard_writer.add_figure("charge2_"+str(iepoch), fig_charge2, global_step=iepoch, close=True, walltime=None)


        print(
            "epoch={} loss={:.4f}/{:.4f} loss_cls={:.4f}/{:.4f} loss_p4={:.4f}/{:.4f} loss_pf={:.4f}/{:.4f} loss_charge={:.4f}/{:.4f} loss_dmode={:.4f}/{:.4f}".format(
                iepoch,
                loss_train,
                loss_val,
                loss_train_cls,
                loss_val_cls,
                loss_train_p4,
                loss_val_p4,
                loss_train_pf,
                loss_val_pf,
                loss_train_charge,
                loss_val_charge,
                loss_train_dmode,
                loss_val_dmode
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
        pred_fromtau, pred_istau, pred_p4, pred_charge, pred_dmode = self.model(data_obj)

        pred_istau = torch.flatten(pred_istau).contiguous().detach().numpy()
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
                    "energy": pred_p4[:, 3],
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
            "tau_charge": np.array([hotToCharge(d) for d in pred_charge.detach().numpy()]),
            "tau_decaymode": np.array([hotToDmode(d) for d in pred_dmode.detach().numpy()]),
        }
