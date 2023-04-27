import os
import json
import sys
import hydra
import yaml
import torch.nn as nn
import torch
import math
import random

torch.cuda.empty_cache()
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FocalLoss import FocalLoss

focal_loss = FocalLoss(gamma=1)
import numpy as np
import collections
from taujetdataset_withgrid import TauJetDatasetWithGrid

from part_var import Var

# https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds):
        super(MyIterableDataset).__init__()
        self.ds = ds
        # this will be overridden later
        self.num_jets = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            range_per_worker = range(len(self.ds))
        else:  # in a worker process
            per_worker = int(math.ceil(len(self.ds)) / worker_info.num_workers)
            tw = worker_info.id
            range_per_worker = range(tw * per_worker, min((tw + 1) * per_worker, len(self.ds)))
        # loop over files in the .pt dataset
        for idx in range_per_worker:
            # get jets from this file
            jets = self.ds.get(idx)
            # shuffle jets from the file
            random.shuffle(jets)
            # loop over jets in file
            for jet in jets:
                yield jet

    def __len__(self):
        return self.num_jets


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.patience >= 0 and self.counter >= self.patience:
                print(f"val_los has not decreased in {self.patience} epochs, stopping")
                return True
        return False


def weighted_huber_loss(pred_tau_p4, true_tau_p4, weights):
    loss_p4 = torch.nn.functional.huber_loss(input=pred_tau_p4, target=true_tau_p4, reduction="none")
    weighted_losses = loss_p4 * weights.unsqueeze(-1)
    return weighted_losses.mean()


def weighted_bce_with_logits(pred_istau, true_istau, weights):
    loss_cls = 10000.0 * focal_loss(pred_istau, true_istau.long())
    weighted_loss_cls = (loss_cls * weights) / torch.sum(weights)
    return torch.sum(weighted_loss_cls)


def ffn(input_dim, output_dim, width, act, drop_out=0.0):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, width),
        act,
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        act,
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, width),
        act,
        nn.Dropout(drop_out),
        nn.LayerNorm(width),
        nn.Linear(width, output_dim),
    )


def conv(input_dim, outputdim, width1, width2, dim, act, kernel=1, dropout=0.0):
    return nn.Sequential(
        nn.LayerNorm([input_dim, dim, dim]),
        nn.Conv2d(input_dim, width1, kernel),
        act,
        nn.Dropout(dropout),
        nn.LayerNorm([width1, dim, dim]),
        nn.Conv2d(width1, width2, kernel),
        act,
        nn.Dropout(dropout),
        nn.LayerNorm([width2, dim, dim]),
        nn.Conv2d(width2, outputdim, kernel),
        act,
        nn.Dropout(dropout),
    )


def reduce_2d(current_dim, shape_1, dev, act):
    reduce_2d_layer = nn.Sequential()
    while current_dim != 1:
        reduce_2d_layer += nn.Sequential(nn.LayerNorm([shape_1, current_dim, current_dim], device=dev))
        reduce_2d_layer += nn.Sequential(nn.Conv2d(shape_1, shape_1, 3, device=dev))
        reduce_2d_layer += nn.Sequential(act)
        reduce_2d_layer += nn.Sequential(nn.Dropout(0.0))
        current_dim -= 2
    return reduce_2d_layer


class DeepTau(nn.Module):
    def __init__(self, grid_cfg, dev):
        super(DeepTau, self).__init__()
        self.device = dev
        self.act = nn.PReLU(device=self.device)
        self.grid_config = grid_cfg["GridAlgo"]
        self.grid_blocks = collections.OrderedDict()
        self.output_from_grid = 64
        self.num_particles_in_grid = grid_cfg["num_particles_in_grid"]
        self.inner_grid = conv(
            self.num_particles_in_grid * Var.max_value(),
            self.output_from_grid,
            104,
            88,
            self.grid_config["inner_grid"]["n_cells"],
            self.act,
        )
        self.outer_grid = conv(
            self.num_particles_in_grid * Var.max_value(),
            self.output_from_grid,
            104,
            88,
            self.grid_config["outer_grid"]["n_cells"],
            self.act,
        )
        self.inner_grid.to(device=self.device)
        self.outer_grid.to(device=self.device)
        self.tau_ftrs = ffn(21, 57, 100, self.act)
        self.pred_istau = ffn(2 * self.output_from_grid + 21, 2, 100, self.act)
        # self.pred_istau = ffn(21, 2, 100, self.act)
        # self.pred_p4 = ffn(21, 4, 100, self.act)
        self.reduce_2d_inner_grid = reduce_2d(
            self.grid_config["inner_grid"]["n_cells"], self.output_from_grid, self.device, self.act
        )
        self.reduce_2d_outer_grid = reduce_2d(
            self.grid_config["outer_grid"]["n_cells"], self.output_from_grid, self.device, self.act
        )

    # x represents our data
    def forward(self, batch):
        # Pass data through conv1
        tau_ftrs_plus_part_ftrs = []
        tau_ftrs_plus_part_ftrs.append(self.tau_ftrs(batch.tau_features))
        # tau_ftrs_plus_part_ftrs = []
        for grid in ["inner_grid", "outer_grid"]:
            layer = self.inner_grid(batch[f"{grid}"]) if "inner" in grid else self.outer_grid(batch[f"{grid}"])
            layer = self.reduce_2d_inner_grid(layer) if "inner" in grid else self.reduce_2d_outer_grid(layer)
            flatten_features = torch.flatten(layer, start_dim=1)
            tau_ftrs_plus_part_ftrs.append(flatten_features)
        tau_all_block_features = torch.cat(tau_ftrs_plus_part_ftrs, axis=-1)
        pred_istau = self.pred_istau(tau_all_block_features)
        return pred_istau


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev):
    loss_cls_tot = 0.0
    if is_train:
        model.train()
    else:
        model.eval()
    nsteps = 0
    njets = 0
    # loop over batches in data
    class_true = []
    class_pred = []
    tot_acc = []
    for ibatch, batch in enumerate(ds_loader):
        optimizer.zero_grad()
        batch = batch.to(device=dev)
        pred_istau = model(batch)
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        acc = (torch.softmax(pred_istau, axis=-1)[:, 1].round() == true_istau).float().mean()
        tot_acc.append(acc.detach().cpu())
        weights = batch.weight

        # loss_p4 = weighted_huber_loss(pred_p4, true_p4, weights)
        loss_cls = weighted_bce_with_logits(pred_istau, true_istau, weights)

        loss = loss_cls
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            class_true.append(true_istau.detach().cpu().numpy())
            class_pred.append(torch.softmax(pred_istau, axis=-1)[:, 1].detach().cpu().numpy())
        loss_cls_tot += loss_cls.detach().cpu().item()
        # loss_p4_tot += loss_p4.detach().cpu().item()
        nsteps += 1
        njets += true_istau.shape[0]
    sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return (loss_cls_tot / njets, (class_true, class_pred), sum(tot_acc) / len(tot_acc))


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths


@hydra.main(config_path="../config", config_name="deeptauTraining", version_base=None)
def main(cfg):
    gridFileName = cfg.DeepTau_training.grid_config
    cfgFile = open(gridFileName, "r")
    grid_cfg = json.load(cfgFile)

    ds_train = TauJetDatasetWithGrid("/local/snandan/CLIC_data_withcorrectpartmul/dataset_train/")
    ds_val = TauJetDatasetWithGrid("/local/snandan/CLIC_data_withcorrectpartmul/dataset_validation/")

    ds_train_iter = MyIterableDataset(ds_train)
    ds_val_iter = MyIterableDataset(ds_val)

    ds_train_loader = DataLoader(ds_train_iter, batch_size=cfg.DeepTau_training.batch_size, num_workers=4, prefetch_factor=4)
    ds_val_loader = DataLoader(ds_val_iter, batch_size=cfg.DeepTau_training.batch_size, num_workers=4, prefetch_factor=4)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = DeepTau(grid_cfg, dev)
    print(model)
    model.to(device=dev)
    print("params={}".format(count_parameters(model)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    # optimizer, max_lr=0.1, steps_per_epoch=len(ds_train_loader), epochs=cfg.DeepTau_training.epochs
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.01
    # , threshold=0.01, verbose=True, patience=5)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]
    # outpath = hydra_cfg["runtime"]["output_dir"]('home/snandan')
    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")
    early_stopper = EarlyStopper(patience=50, min_delta=10)
    best_loss = np.inf

    for iepoch in range(cfg.DeepTau_training.epochs):
        loss_cls_train, _, acc_cls_train = model_loop(model, ds_train_loader, optimizer, scheduler, True, dev)
        tensorboard_writer.add_scalar("epoch/train_cls_loss", loss_cls_train, iepoch)
        loss_cls_val, retvals, acc_cls_val = model_loop(model, ds_val_loader, optimizer, scheduler, False, dev)
        tensorboard_writer.add_scalar("epoch/val_cls_loss", loss_cls_val, iepoch)
        tensorboard_writer.add_scalar("epoch/lr", optimizer.param_groups[0]["lr"], iepoch)
        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1], iepoch)
        print(
            "epoch={} loss_cls={:.4f}/{:.4f} acc={:.3f}/{:.3f}".format(
                iepoch, loss_cls_train, loss_cls_val, acc_cls_train, acc_cls_val
            )
        )
        # scheduler.step(loss_cls_val+loss_p4_val)
        if loss_cls_val < best_loss:
            best_loss = loss_cls_val
            print("best model is saved in {}/model_best_epoch_{}.pt".format(outpath, iepoch))
            torch.save(model, "{}/model_best_epoch_{}.pt".format(outpath, iepoch))
        if early_stopper.early_stop(loss_cls_val):
            break
    model = model.to(device="cpu")
    torch.save(model, "data/model_deeptau_v2.pt")


if __name__ == "__main__":
    main()
