import os
import json
import sys
import hydra
import yaml
import torch.nn as nn
import torch

torch.cuda.empty_cache()
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from FocalLoss import FocalLoss

focal_loss = FocalLoss(gamma=1)
import numpy as np
import collections
from taujetdataset_withgrid import TauJetDatasetWithGrid

from part_var import Var


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
    weighted_loss_cls = loss_cls * weights
    return weighted_loss_cls.mean()


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
        # nn.Sigmoid(),
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
        nn.Dropout(dropout),
    )


class DeepTau(nn.Module):
    def __init__(self, grid_cfg, dev):
        super(DeepTau, self).__init__()
        self.device = dev
        self.act = nn.PReLU
        self.grid_config = grid_cfg["GridAlgo"]
        self.grid_blocks = collections.OrderedDict()
        self.output_from_grid = 64
        self.num_particles_in_grid = grid_cfg["num_particles_in_grid"]
        self.grid_blocks["inner_grid"] = conv(
            self.num_particles_in_grid * Var.max_value(),
            self.output_from_grid,
            104,
            88,
            self.grid_config["inner_grid"]["n_cells"],
            self.act,
        )
        self.grid_blocks["outer_grid"] = conv(
            self.num_particles_in_grid * Var.max_value(),
            self.output_from_grid,
            104,
            88,
            self.grid_config["outer_grid"]["n_cells"],
            self.act,
        )
        self.grid_blocks["inner_grid"].to(device=self.device)
        self.grid_blocks["outer_grid"].to(device=self.device)
        self.pred_istau = ffn(2 * self.output_from_grid + 18, 2, 100, self.act)
        self.pred_p4 = ffn(2 * self.output_from_grid + 18, 4, 100, self.act)
        # self.pred_istau = ffn(18, 2, 100, self.act)
        # self.pred_p4 = ffn(18, 4, 100, self.act)
        self.reduce_2d = collections.OrderedDict()
        for grid in self.grid_blocks.keys():
            reduce_2d_layer = nn.Sequential()
            current_dim = self.grid_config[grid]["n_cells"]
            shape_1 = self.output_from_grid
            while current_dim != 1:
                reduce_2d_layer += nn.Sequential(nn.LayerNorm([shape_1, current_dim, current_dim], device=self.device))
                reduce_2d_layer += nn.Sequential(nn.Conv2d(shape_1, shape_1, 3, device=self.device))
                reduce_2d_layer += nn.Sequential(self.act(device=self.device))
                reduce_2d_layer += nn.Sequential(nn.Dropout(0.2))
                current_dim -= 2
            self.reduce_2d[grid] = reduce_2d_layer
        # self.ffn = ffn(18, 1, 200, self.act)
        # self.ffn = ffn(2*self.output_from_grid, 1, 100, self.act)

    # x represents our data
    def forward(self, batch):
        # Pass data through conv1
        tau_ftrs_plus_part_ftrs = [batch.tau_features]
        # tau_ftrs_plus_part_ftrs = []
        for (grid, conv) in self.grid_blocks.items():
            layer = conv(batch[f"{grid}"])
            layer = self.reduce_2d[grid](layer)
            flatten_features = torch.flatten(layer, start_dim=1)
            tau_ftrs_plus_part_ftrs.append(flatten_features)
        tau_all_block_features = torch.concatenate(tau_ftrs_plus_part_ftrs, axis=-1)
        pred_istau = self.pred_istau(tau_all_block_features)  # .squeeze(-1)
        pred_p4 = batch.tau_features[0 : batch.tau_features.shape[0], 0:4] * self.pred_p4(tau_all_block_features)
        return pred_istau, pred_p4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_loop(model, ds_loader, optimizer, loss_cls_fn, loss_p4_fn, scheduler, is_train, dev):
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
    tot_acc = []
    tot_acc_p4 = []
    for ibatch, batch in enumerate(ds_loader):
        optimizer.zero_grad()
        batch = batch.to(device=dev)
        pred_istau, pred_p4 = model(batch)
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        true_p4 = batch.gen_tau_p4
        acc = (torch.softmax(pred_istau, axis=-1)[:, 1].round() == true_istau).float().mean()
        acc_p4 = ((true_p4 - pred_p4) ** 2).mean()
        tot_acc.append(acc.detach().cpu())
        tot_acc_p4.append(acc_p4.detach().cpu())
        weights = batch.weight

        loss_p4 = weighted_huber_loss(pred_p4, true_p4, weights)
        loss_cls = weighted_bce_with_logits(pred_istau, true_istau, weights)

        loss = loss_cls + loss_p4
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            class_true.append(true_istau.detach().cpu().numpy())
            class_pred.append(torch.softmax(pred_istau, axis=-1)[:, 1].detach().cpu().numpy())
        loss_cls_tot += loss_cls.detach().cpu().item()
        loss_p4_tot += loss_p4.detach().cpu().item()
        nsteps += 1
        njets += true_istau.shape[0]
    sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return (
        loss_cls_tot / njets,
        loss_p4_tot / njets,
        (class_true, class_pred),
        sum(tot_acc) / len(tot_acc),
        sum(tot_acc_p4) / len(tot_acc_p4),
    )


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

    ds_train = TauJetDatasetWithGrid("/local/snandan/CLIC_data/dataloader")
    ds_val = TauJetDatasetWithGrid("/local/snandan/CLIC_data/dataloader_validation")

    train_data = [ds_train[i] for i in range(5)]  # len(ds_train))]
    train_data = sum(train_data, [])
    val_data = [ds_val[i] for i in range(5)]  # len(ds_val))]
    val_data = sum(val_data, [])

    print("Loaded TauJetDatasetWithGrid with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDatasetWithGrid with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(train_data, batch_size=cfg.DeepTau_training.batch_size, shuffle=True)
    ds_val_loader = DataLoader(val_data, batch_size=cfg.DeepTau_training.batch_size, shuffle=True)
    """for x in ds_train_loader:
    print(len(x))"""
    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = DeepTau(grid_cfg, dev)
    model.to(device=dev)
    # for name, conv in model.grid_blocks.items():
    # name = conv.to(device=dev)
    print("params={}".format(count_parameters(model)))
    loss_cls_fn = torch.nn.BCELoss()
    loss_p4_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    # optimizer, max_lr=0.000001, steps_per_epoch=len(ds_train_loader), epochs=cfg.DeepTau_training.epochs
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]
    # outpath = hydra_cfg["runtime"]["output_dir"]('home/snandan')
    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")
    early_stopper = EarlyStopper(patience=50, min_delta=10)
    best_loss = np.inf

    for iepoch in range(cfg.DeepTau_training.epochs):
        loss_cls_train, loss_p4_train, _, acc_cls_train, acc_p4_train = model_loop(
            model, ds_train_loader, optimizer, loss_cls_fn, loss_p4_fn, scheduler, True, dev
        )
        tensorboard_writer.add_scalar("epoch/train_cls_loss", loss_cls_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_p4_loss", loss_p4_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss", loss_cls_train + loss_p4_train, iepoch)

        loss_cls_val, loss_p4_val, retvals, acc_cls_val, acc_p4_val = model_loop(
            model, ds_val_loader, optimizer, loss_cls_fn, loss_p4_fn, scheduler, False, dev
        )
        tensorboard_writer.add_scalar("epoch/val_cls_loss", loss_cls_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_p4_loss", loss_p4_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss", loss_cls_val + loss_p4_val, iepoch)
        tensorboard_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], iepoch)
        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1], iepoch)
        print(
            "epoch={} loss_cls={:.4f}/{:.4f} loss_p4={:.4f}/{:.4f} acc={:.3f}/{:.3f} acc_p4={:.3f}/{:.3f}".format(
                iepoch,
                loss_cls_train,
                loss_cls_val,
                loss_p4_train,
                loss_p4_val,
                acc_cls_train,
                acc_cls_val,
                acc_p4_train,
                acc_p4_val,
            )
        )
        if loss_cls_val + loss_p4_val < best_loss:
            best_loss = loss_cls_val + loss_p4_val
            print("best model is saved in {}/model_best_epoch_{}.pt".format(outpath, iepoch))
            torch.save(model, "{}/model_best_epoch_{}.pt".format(outpath, iepoch))
        if early_stopper.early_stop(loss_cls_val):
            break
    model = model.to(device="cpu")
    torch.save(model, "data/model_deeptau_v2.pt")


if __name__ == "__main__":
    main()
