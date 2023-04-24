#!/usr/bin/python3

import datetime
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
import psutil
import subprocess
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from LorentzNetDataset import LorentzNetDataset
from LorentzNet import LorentzNet
from FeatureStandardization import FeatureStandardization
from FocalLoss import FocalLoss
from logTrainingProgress import logTrainingProgress


def get_split_files(cfg_filename, split):
    with open(cfg_filename, "r") as cfg_file:
        data = yaml.safe_load(cfg_file)
        paths = data[split]["paths"]

        # FIXME: this is hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch/persistent for GPU training
        # paths = [p.replace("/local/laurits", "./data") for p in paths]
        return paths


def train_loop(
    idx_epoch,
    dataloader_train,
    transform,
    model,
    dev,
    loss_fn,
    use_per_jet_weights,
    optimizer,
    lr_scheduler,
    tensorboard,
):
    num_jets_train = len(dataloader_train.dataset)
    loss_train = 0.0
    loss_normalization_train = 0.0
    accuracy_train = 0.0
    accuracy_normalization_train = 0.0
    class_true_train = []
    class_pred_train = []
    model.train()
    for idx_batch, (X, y, weight) in enumerate(dataloader_train):
        # Compute prediction and loss
        if transform:
            X = transform(X)
        x = X["x"].to(device=dev)
        scalars = X["scalars"].to(device=dev)
        mask = X["mask"].to(device=dev)
        y = y.squeeze(dim=1).to(device=dev)
        weight = weight.squeeze(dim=1).to(device=dev)
        pred = model(x, scalars, mask)

        loss = None
        if use_per_jet_weights:
            loss = loss_fn(pred, y)
            loss = loss * weight
        else:
            loss = loss_fn(pred, y)
        loss_train += loss.sum().item()
        loss_normalization_train += torch.flatten(loss).size(dim=0)
        accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
        accuracy_train += accuracy.sum().item()
        accuracy_normalization_train += torch.flatten(accuracy).size(dim=0)

        class_true_train.extend(y.detach().cpu().numpy())
        class_pred_train.extend(pred.argmax(dim=1).detach().cpu().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        lr_scheduler.step()

        batchsize = pred.size(dim=0)
        num_jets_processed = min((idx_batch + 1) * batchsize, num_jets_train)
        if (idx_batch % 100) == 0 or num_jets_processed >= (num_jets_train - batchsize):
            print(" Running loss: %1.6f  [%i/%s]" % (loss.mean().item(), num_jets_processed, num_jets_train))

    loss_train /= loss_normalization_train
    accuracy_train /= accuracy_normalization_train

    logTrainingProgress(
        tensorboard, idx_epoch, "train", loss_train, accuracy_train, np.array(class_true_train), np.array(class_pred_train)
    )

    return loss_train


def validation_loop(
    idx_epoch,
    dataloader_validation,
    transform,
    model,
    dev,
    loss_fn,
    use_per_jet_weights,
    tensorboard,
):
    loss_validation = 0.0
    loss_normalization_validation = 0.0
    accuracy_validation = 0.0
    accuracy_normalization_validation = 0.0
    class_true_validation = []
    class_pred_validation = []
    model.eval()
    with torch.no_grad():
        for idx_batch, (X, y, weight) in enumerate(dataloader_validation):
            if transform:
                X = transform(X)
            x = X["x"].to(device=dev)
            scalars = X["scalars"].to(device=dev)
            mask = X["mask"].to(device=dev)
            y = y.squeeze(dim=1).to(device=dev)
            weight = weight.squeeze(dim=1).to(device=dev)
            pred = model(x, scalars, mask)

            if use_per_jet_weights:
                loss = loss_fn(pred, y)
                loss = loss * weight
            else:
                loss = loss_fn(pred, y).item()
            loss_validation += loss.sum().item()
            loss_normalization_validation += torch.flatten(loss).size(dim=0)
            accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
            accuracy_validation += accuracy.sum().item()
            accuracy_normalization_validation += torch.flatten(accuracy).size(dim=0)

            class_true_validation.extend(y.detach().cpu().numpy())
            class_pred_validation.extend(pred.argmax(dim=1).detach().cpu().numpy())

    loss_validation /= loss_normalization_validation
    accuracy_validation /= accuracy_normalization_validation

    logTrainingProgress(
        tensorboard,
        idx_epoch,
        "validation",
        loss_validation,
        accuracy_validation,
        np.array(class_true_validation),
        np.array(class_pred_validation),
    )

    return loss_validation


def run_command(cmd):
    result = subprocess.run(cmd.split(" "), stdout=subprocess.PIPE, universal_newlines=True)
    print(result.stdout)


@hydra.main(config_path="../config", config_name="trainLorentzNet", version_base=None)
def trainLorentzNet(train_cfg: DictConfig) -> None:
    print("<trainLorentzNet>:")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    config_path = "./config"
    for cfg in hydra_cfg["runtime"]["config_sources"]:
        if cfg["schema"] == "file":
            config_path = cfg["path"]
    config_name = hydra_cfg["job"]["config_name"]
    print("Loading training configuration from file: %s/%s.yaml" % (config_path, config_name))
    outpath = hydra_cfg["runtime"]["output_dir"]
    print(" outpath = %s" % outpath)

    filelist_train = get_split_files("config/datasets/train.yaml", "train")
    filelist_validation = get_split_files("config/datasets/validation.yaml", "validation")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: %s" % dev)

    jsonFileName = "%s/%s" % (config_path, train_cfg.model_config_file)
    print("Loading model configuration from file: %s" % jsonFileName)
    LorentzNet_cfg = None
    if os.path.isfile(jsonFileName):
        jsonFile = open(jsonFileName, "r")
        LorentzNet_cfg = json.load(jsonFile)
        if "LorentzNet" not in LorentzNet_cfg.keys():
            raise RuntimeError("Failed to parse config file %s !!")
        LorentzNet_cfg = LorentzNet_cfg["LorentzNet"]
        for key, value in LorentzNet_cfg.items():
            print(" %s = " % key, value)
        jsonFile.close()
    else:
        raise RuntimeError("Failed to read config file %s !!")

    n_hidden = LorentzNet_cfg["n_hidden"]
    n_class = LorentzNet_cfg["n_class"]
    dropout = LorentzNet_cfg["dropout"]
    n_layers = LorentzNet_cfg["n_layers"]
    c_weight = LorentzNet_cfg["c_weight"]
    max_cands = LorentzNet_cfg["max_cands"]
    add_beams = LorentzNet_cfg["add_beams"]
    use_pdgId = LorentzNet_cfg["use_pdgId"]
    n_scalar = 8 if use_pdgId else 2
    standardize_inputs = LorentzNet_cfg["standardize_inputs"]
    preselection = {
        "min_jet_theta": LorentzNet_cfg["min_jet_theta"],
        "max_jet_theta": LorentzNet_cfg["max_jet_theta"],
        "min_jet_pt": LorentzNet_cfg["min_jet_pt"],
        "max_jet_pt": LorentzNet_cfg["max_jet_pt"],
    }

    print("Building model...")
    model = LorentzNet(
        n_scalar=n_scalar,
        n_hidden=n_hidden,
        n_class=n_class,
        dropout=dropout,
        n_layers=n_layers,
        c_weight=c_weight,
        verbosity=train_cfg.verbosity,
    ).to(device=dev)
    print("Finished building model:")
    print(model)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_weights = sum([np.prod(p.size()) for p in model_params])
    print("#trainable parameters = %i" % num_trainable_weights)

    print("Starting to build training dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_train = LorentzNetDataset(
        filelist_train,
        max_num_files=train_cfg.max_num_files,
        max_cands=max_cands,
        add_beams=add_beams,
        use_pdgId=use_pdgId,
        preselection=preselection,
    )
    print("Finished building training dataset.")
    print(" current time:", datetime.datetime.now())

    print("Starting to build validation dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_validation = LorentzNetDataset(
        filelist_validation,
        max_num_files=train_cfg.max_num_files,
        max_cands=max_cands,
        add_beams=add_beams,
        use_pdgId=use_pdgId,
        preselection=preselection,
    )
    print("Finished building validation dataset.")
    print(" current time:", datetime.datetime.now())

    dataloader_train = DataLoader(
        dataset_train, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )

    transform = None
    if standardize_inputs:
        transform = FeatureStandardization(features=["x", "scalars"], dim=2, verbosity=train_cfg.verbosity)
        transform.compute_params(dataloader_train)
        transform.save_params(LorentzNet_cfg["json_file_FeatureStandardization"])

    classweight_bgr = 1.0
    classweight_sig = 1.0
    if train_cfg.use_class_weights:
        classweight_bgr = train_cfg.classweight_bgr
        classweight_sig = train_cfg.classweight_sig
    classweight_tensor = torch.tensor([classweight_bgr, classweight_sig], dtype=torch.float32).to(device=dev)
    loss_fn = None
    if train_cfg.use_focal_loss:
        loss_fn = FocalLoss(gamma=train_cfg.focal_loss_gamma, alpha=classweight_tensor, reduction="none")
    else:
        loss_fn = nn.CrossEntropyLoss(weight=classweight_tensor, reduction="none")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-2)
    num_batches_train = len(dataloader_train)
    print("Training for %i epochs." % train_cfg.num_epochs)
    print("#batches(train) = %i" % num_batches_train)
    lr_scheduler = OneCycleLR(
        optimizer, max_lr=1.0e-3, epochs=train_cfg.num_epochs, steps_per_epoch=num_batches_train, anneal_strategy="cos"
    )

    print("Starting training...")
    print(" current time:", datetime.datetime.now())
    tensorboard = SummaryWriter(outpath + "/tensorboard")
    min_loss_validation = -1.0
    for idx_epoch in range(train_cfg.num_epochs):
        print("Processing epoch #%i" % idx_epoch)
        print(" current time:", datetime.datetime.now())

        train_loop(
            idx_epoch,
            dataloader_train,
            transform,
            model,
            dev,
            loss_fn,
            train_cfg.use_per_jet_weights,
            optimizer,
            lr_scheduler,
            tensorboard,
        )
        print(" lr = %1.3e" % lr_scheduler.get_last_lr()[0])
        # print(" lr = %1.3e" % get_lr(optimizer))
        tensorboard.add_scalar("lr", lr_scheduler.get_last_lr()[0], idx_epoch)

        loss_validation = validation_loop(
            idx_epoch,
            dataloader_validation,
            transform,
            model,
            dev,
            loss_fn,
            train_cfg.use_per_jet_weights,
            tensorboard,
        )
        if min_loss_validation == -1.0 or loss_validation < min_loss_validation:
            print("Found new best model :)")
            best_model_file = train_cfg.model_file.replace(".pt", "_best.pt")
            print("Saving best model to file %s" % best_model_file)
            torch.save(model.state_dict(), best_model_file)
            print("Done.")
            min_loss_validation = loss_validation

        print("System utilization:")
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent(interval=1)
        print(" CPU-Util = %1.2f%%" % cpu_percent)
        print(" Memory-Usage = %i Mb" % (process.memory_info().rss / 1048576))
        if dev == "cuda":
            print("GPU:")
            run_command("nvidia-smi --id=%i" % torch.cuda.current_device())
        else:
            print("GPU: N/A")
    print("Finished training.")
    print(" current time:", datetime.datetime.now())

    print("Saving model to file %s" % train_cfg.model_file)
    torch.save(model.state_dict(), train_cfg.model_file)
    print("Done.")

    tensorboard.close()


if __name__ == "__main__":
    trainLorentzNet()
