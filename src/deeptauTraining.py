import os
import json
import sys
import hydra
import yaml
import awkward as ak
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os.path as osp
from glob import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import vector
import math
import collections
from taujetdataset_withgrid import TauJetDatasetWithGrid
from torch.utils.tensorboard import SummaryWriter

from part_var import Var

def ffn(input_dim, output_dim, width, act, drop_out=0.2):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, width),
        act(),
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        act(),
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        act(),
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, output_dim),
        nn.Sigmoid()
    )
    
def conv(input_dim, outputdim, width1, width2, dim, act, kernel=1, dropout=0.2):
    return nn.Sequential(
        nn.LayerNorm([input_dim, dim, dim]),
        nn.Conv2d(input_dim, width1, kernel),
        act(),
        nn.Dropout(dropout),
        nn.LayerNorm([width1, dim, dim]),
        nn.Conv2d(width1, width2, kernel),
        act(),
        nn.Dropout(dropout),
        nn.LayerNorm([width2, dim, dim]),
        nn.Conv2d(width2, outputdim, kernel),
        act(),
        nn.Dropout(dropout)
    )

class DeepTau(nn.Module):
    def __init__(self, grid_cfg):
        super(DeepTau, self).__init__()
        self.act = nn.PReLU
        self.grid_config = grid_cfg["GridAlgo"]
        self.grid_blocks = collections.OrderedDict()
        self.output_from_grid = 64
        self.num_particles_in_grid = grid_cfg["num_particles_in_grid"]
        self.grid_blocks['inner_grid'] = conv(self.num_particles_in_grid*Var.max_value(), self.output_from_grid, 104, 88, self.grid_config['inner_grid']["n_cells"], self.act)
        self.grid_blocks['outer_grid'] = conv(self.num_particles_in_grid*Var.max_value(), self.output_from_grid, 104, 88, self.grid_config['outer_grid']["n_cells"], self.act)
        self.ffn = ffn(2*self.output_from_grid+18, 1, 100, self.act)
        #self.ffn = ffn(18, 1, 200, self.act)
        #self.ffn = ffn(2*self.output_from_grid, 1, 100, self.act)

    # x represents our data
    def forward(self, batch):
        # Pass data through conv1
        tau_ftrs_plus_part_ftrs = [batch.tau_features]
        for (grid,conv) in self.grid_blocks.items():
            layer = conv(batch[f'{grid}'])
            current_dim = layer.shape[2]
            shape_1 = layer.shape[1]
            while current_dim !=1:
                layer = nn.LayerNorm([shape_1, current_dim, current_dim])(layer)
                layer = nn.Conv2d(shape_1, shape_1, 3)(layer)
                layer = self.act()(layer)
                layer = nn.Dropout(0.2)(layer)
                current_dim -= 2
            flatten_features = torch.flatten(layer, start_dim=1)
            tau_ftrs_plus_part_ftrs.append(flatten_features)
        tau_all_block_features = torch.concatenate( tau_ftrs_plus_part_ftrs, axis=-1)
        output = self.ffn(tau_all_block_features).squeeze(-1)
        return output    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_loop(model, ds_loader, optimizer, loss_fn, scheduler, is_train, dev):
    loss_cls_tot = 0.0
    loss_p4_tot = 0.0

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
        pred_istau = model(batch)
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        #print('pred_istau: ', pred_istau)
        #print('true_istau: ', true_istau)
        acc = (pred_istau.round() == true_istau).float().mean()
        loss_cls = loss_fn(pred_istau, true_istau)
        loss = loss_cls
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            class_true.append(true_istau.cpu().numpy())
            class_pred.append(torch.sigmoid(pred_istau).detach().cpu().numpy())
        loss_cls_tot += loss_cls.detach().cpu().item()
        nsteps += 1
        njets += true_istau.shape[0]
        print('njets: ', true_istau.shape[0], " ntau: ", true_istau.sum().cpu().item())
    sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return loss_cls_tot*100. / njets, 1, (class_true, class_pred), acc

def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths

@hydra.main(config_path="../config", config_name="deeptauTraining", version_base=None)
def main(cfg):
    sig_input_dir = cfg.algorithms.HPS.sig_ntuples_dir
    bkg_input_dir = cfg.algorithms.HPS.bkg_ntuples_dir
    gridFileName = cfg.DeepTau_training.grid_config
    cfgFile = open(gridFileName, "r")
    grid_cfg = json.load(cfgFile)

    sig_train_paths = [
        os.path.join(sig_input_dir, os.path.basename(path)) for path in cfg.datasets.train.paths if "ZH_Htautau" in path
    ]
    bkg_train_paths = [
        os.path.join(bkg_input_dir, os.path.basename(path)) for path in cfg.datasets.train.paths if "QCD" in path
    ]
    sig_val_paths = [
        os.path.join(sig_input_dir, os.path.basename(path)) for path in cfg.datasets.validation.paths if "ZH_Htautau" in path
    ]
    bkg_val_paths = [
        os.path.join(bkg_input_dir, os.path.basename(path)) for path in cfg.datasets.validation.paths if "QCD" in path
    ]
    
    files_train = ['/local/laurits/CLIC_taus_20230215_v1/HPS/ZH_Htautau/reco_p8_ee_ZH_Htautau_ecm380_360.parquet']#'grid/Grid/ZH_Htautau/reco_p8_ee_ZH_Htautau_ecm380_1.parquet']# + ['hps/HPS/QCD/reco_p8_ee_QCD_ecm380_1.parquet']#+ sig_train_paths + bkg_train_paths
    files_val = ['/local/laurits/CLIC_taus_20230215_v1/HPS/QCD/reco_p8_ee_qcd_ecm365_1.parquet']#grid/Grid/QCD/reco_p8_ee_QCD_ecm380_1.parquet']#files_train#sig_val_paths + bkg_val_paths
    #files_train = sig_train_paths[:100] + bkg_train_paths[:100]
    #files_val = sig_val_paths[:100] + bkg_val_paths[:100]
    
    ds_train = TauJetDatasetWithGrid(files_train)
    ds_val = TauJetDatasetWithGrid(files_val)

    print("Loaded TauJetDatasetWithGrid with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDatasetWithGrid with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(ds_train, batch_size=cfg.DeepTau_training.batch_size, shuffle=True)
    ds_val_loader = DataLoader(ds_val, batch_size=cfg.DeepTau_training.batch_size, shuffle=True)

    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = DeepTau(grid_cfg).to(device=dev)
    print("params={}".format(count_parameters(model)))
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.000001, steps_per_epoch=len(ds_train_loader), epochs=cfg.DeepTau_training.epochs
    )
    
    loss_true = []
    loss_val = []
    acc_true = []
    acc_val = []
    for iepoch in range(cfg.DeepTau_training.epochs):
        loss_cls_train, loss_p4_train, _, acc_cls_train = model_loop(model, ds_train_loader, optimizer, loss_fn, scheduler, True, dev)
        loss_true.append(loss_cls_train)
        acc_true.append(acc_cls_train)
        loss_cls_val, loss_p4_val, retvals, acc_cls_val = model_loop(model, ds_val_loader, optimizer, loss_fn, scheduler, False, dev)
        loss_val.append(loss_cls_val)
        acc_val.append(acc_cls_val)
        print(
            "epoch={} loss={:.4f}/{:.4f} p4={:.4f}/{:.4f} acc={:.3f}/{:.3f}".format(
                iepoch, loss_cls_train, loss_cls_val, loss_p4_train, loss_p4_val, acc_cls_train, acc_cls_val
            )
        )
        torch.save(model, "data/model_deeptau_v2_{}.pt".format(iepoch))
    epochs = np.arange(0,cfg.DeepTau_training.epochs,1)
    plt.plot(epochs, np.array(loss_true), 'b', epochs, np.array(loss_val), 'r--', epochs, np.array(acc_true), 'b', epochs, np.array(acc_val), 'r--')
    plt.savefig('loss.png')
    plt.close('all')
    plt.plot(epochs, np.array(acc_true), 'b', epochs, np.array(acc_val), 'r--')
    plt.savefig('accu.png')
    plt.close('all')
    model = model.to(device="cpu")
    torch.save(model, "data/model_deeptau_v2.pt")

if __name__ == "__main__":
    main()
    
