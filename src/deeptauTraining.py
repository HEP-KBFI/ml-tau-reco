import os
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
        act(),
        nn.Dropout(drop_out)
    )
    
def reduce_2d_conv(input_dim, kernel1, kernel2, kernel3, kernel4, kernel5, act, dropout=0.2):
    return nn.Sequential(
        #nn.LayerNorm(input_dim),
        nn.Conv2d(input_dim, input_dim, kernel1),
        act(),
        nn.Dropout(dropout),
        #nn.LayerNorm(input_dim),
        nn.Conv2d(input_dim, input_dim, kernel2),
        act(),
        nn.Dropout(dropout),
        #nn.LayerNorm(input_dim),
        nn.Conv2d(input_dim, input_dim, kernel3),
        act(),
        nn.Dropout(dropout),
        #nn.LayerNorm(input_dim),
        nn.Conv2d(input_dim, input_dim, kernel4),
        act(),
        nn.Dropout(dropout),
        #nn.LayerNorm(input_dim),
        nn.Conv2d(input_dim, input_dim, kernel5),
        act(),
        nn.Dropout(dropout)
    )

def conv_wdim(input_dim, outputdim, width1, width2, kernel, act, dim, dropout=0.2):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Conv2d(input_dim, width1, kernel),
        act(),
        nn.Dropout(dropout),
        nn.Conv2d(width1, width2, kernel),
        act(),
        nn.Dropout(dropout),
        nn.Conv2d(width2, outputdim, kernel),
        act(),
        nn.Dropout(dropout)
    )

def conv(input_dim, outputdim, width1, width2, kernel, act, inner=True, dropout=0.2):
    if inner:
        dim = [input_dim, 11, 11]
    else:
        dim = [input_dim, 21, 21]
    return conv_wdim(input_dim, outputdim, width1, width2, kernel, act, dim, dropout=0.2)

class DeepTau(nn.Module):
    def __init__(self):
        super(DeepTau, self).__init__()
        self.act = nn.PReLU
        self.part_blocks = collections.OrderedDict()
        self.part_blocks['inner_grid'] = collections.OrderedDict([
            ('ele_gamma_block',      conv(8, 104, 207, 129, 1, self.act)),
            ('mu_block',             conv(4, 77,  154, 96,  1, self.act)),
            ('charged_neutral_block', conv(8, 46,  92,  57,  1, self.act))
        ])
        
        self.part_blocks['outer_grid'] = collections.OrderedDict([
            ('ele_gamma_block',      conv(8, 104, 207, 129, 1, self.act, False)),
            ('mu_block',             conv(4, 77,  154, 96,  1, self.act, False)),
            ('charged_neutral_block', conv(8, 46,  92,  57,  1, self.act, False))
        ])
        self.all_part_conv_inner_block = conv(227, 64, 141, 88, 1, self.act)
        self.all_part_conv_outer_block = conv(227, 64, 141, 88, 1, self.act, False)
        self.reduce_2d_conv_block = reduce_2d_conv(64, 3, 3, 3, 3, 3, self.act)
        self.ffn = ffn(146, 1, 200, self.act)

    # x represents our data
    def forward(self, batch):
        # Pass data through conv1
        tau_ftrs_plus_part_ftrs = [batch.tau_features]
        for grid in self.part_blocks.keys():
            part_block_conv_ftrs = []
            for (part_block, conv) in self.part_blocks[grid].items():
                part_block_conv_ftrs.append(conv(batch[f'{grid}_{part_block}']))
            all_part_block_conv_ftrs = torch.concatenate([part_ftrs for part_ftrs in part_block_conv_ftrs], axis=1)
            all_features = self.all_part_conv_inner_block(all_part_block_conv_ftrs) if grid == 'inner_grid'\
                           else self.all_part_conv_outer_block(all_part_block_conv_ftrs)
            reduce_2d_features = self.reduce_2d_conv_block(all_features)
            if grid == 'outer_grid':
                reduce_2d_features = self.reduce_2d_conv_block(reduce_2d_features)
            flatten_features = torch.flatten(reduce_2d_features, start_dim=1)
            tau_ftrs_plus_part_ftrs.append(flatten_features)
        tau_all_block_features = torch.concatenate( tau_ftrs_plus_part_ftrs, axis=-1)
        output = self.ffn(tau_all_block_features).squeeze(-1)
        # Use the rectified-linear activation function over x
        
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev):
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
        loss_cls = 10000.0 * torch.nn.functional.binary_cross_entropy_with_logits(pred_istau, true_istau)

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
        print(
            "jets={jets} ntau={ntau} loss={loss:.2f} lr={lr:.2E}".format(
                jets=batch.tau_features.shape[0],
                ntau=true_istau.sum().cpu().item(),
                loss=loss.detach().cpu().item(),
                lr=scheduler.get_last_lr()[0],
            )
        )
        sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return loss_cls_tot / njets, 1, (class_true, class_pred)

def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths

@hydra.main(config_path="../config", config_name="deeptauTraining", version_base=None)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]

    files_train = get_split_files(cfg.train_files, "train")
    files_val = get_split_files(cfg.validation_files, "validation")

    ds_train = TauJetDatasetWithGrid(files_train)
    ds_val = TauJetDatasetWithGrid(files_val)

    print("Loaded TauJetDatasetWithGrid with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDatasetWithGrid with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True)
    ds_val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=True)

    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = DeepTau().to(device=dev)
    print("params={}".format(count_parameters(model)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.000001, steps_per_epoch=len(ds_train_loader), epochs=cfg.epochs
    )
    
    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")
    
    for iepoch in range(cfg.epochs):
        loss_cls_train, loss_p4_train, _ = model_loop(model, ds_train_loader, optimizer, scheduler, True, dev)
        tensorboard_writer.add_scalar("epoch/train_cls_loss", loss_cls_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_p4_loss", loss_p4_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss", loss_cls_train + loss_p4_train, iepoch)

        loss_cls_val, loss_p4_val, retvals = model_loop(model, ds_val_loader, optimizer, scheduler, False, dev)

        tensorboard_writer.add_scalar("epoch/val_cls_loss", loss_cls_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_p4_loss", loss_p4_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss", loss_cls_val + loss_p4_val, iepoch)

        tensorboard_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], iepoch)
        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1], iepoch)

        print(
            "epoch={} cls={:.4f}/{:.4f} p4={:.4f}/{:.4f}".format(
                iepoch, loss_cls_train, loss_cls_val, loss_p4_train, loss_p4_val
            )
        )
        torch.save(model, "{}/model_{}.pt".format(outpath, iepoch))

    model = model.to(device="cpu")
    torch.save(model, "data/model_ddeptau.pt")

if __name__ == "__main__":
    main()
