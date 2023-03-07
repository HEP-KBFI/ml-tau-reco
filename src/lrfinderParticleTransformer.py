#!/usr/bin/python3

import datetime
import hydra
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import os
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

from ParticleTransformerDataset import ParticleTransformerDataset
from ParticleTransformer import ParticleTransformer
from FocalLoss import FocalLoss


def get_split_files(cfg_filename, split):
    with open(cfg_filename, "r") as cfg_file:
        data = yaml.safe_load(cfg_file)
        paths = data[split]["paths"]

        # FIXME: this is hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch-persistent for GPU training
        # paths = [p.replace("/local/laurits", "./data") for p in paths]
        return paths


def train_one_epoch(idx_epoch, dataloader_train, dataloader_test, model, dev, loss_fn, use_per_jet_weights, optimizer):
    num_jets_train = len(dataloader_train.dataset)
    loss_train = 0.0
    accuracy_train = 0.0
    model.train()
    for idx_batch, (X, y, weight) in enumerate(dataloader_train):
        # Compute prediction and loss
        x = X["x"].to(device=dev)
        v = X["v"].to(device=dev)
        mask = X["mask"].to(device=dev)
        y = y.squeeze(dim=1).to(device=dev)
        weight = weight.squeeze(dim=1).to(device=dev)
        pred = model(x, v, mask).to(device=dev)

        loss = None
        if use_per_jet_weights:
            loss = loss_fn(pred, y)
            loss = loss * weight
        else:
            loss = loss_fn(pred, y)
        loss_train += loss.sum().item()
        accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
        accuracy_train += accuracy.sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        batchsize = pred.size(dim=0)
        if (idx_batch % 100) == 0 or idx_batch >= (num_jets_train - batchsize):
            running_loss = loss.mean().item()
            num_jets_processed = min((idx_batch + 1) * batchsize, num_jets_train)
            print(" Running loss: %1.6f  [%i/%s]" % (running_loss, num_jets_processed, num_jets_train))

    loss_train /= num_jets_train
    accuracy_train /= num_jets_train
    print("Train: Avg loss = %1.6f, accuracy = %1.2f" % (loss_train, 100 * accuracy_train))

    num_jets_test = len(dataloader_test.dataset)
    loss_test = 0.0
    accuracy_test = 0.0
    model.eval()
    with torch.no_grad():
        for idx_batch, (X, y, weight) in enumerate(dataloader_test):
            x = X["x"].to(device=dev)
            v = X["v"].to(device=dev)
            mask = X["mask"].to(device=dev)
            y = y.squeeze(dim=1).to(device=dev)
            weight = weight.squeeze(dim=1).to(device=dev)
            pred = model(x, v, mask).to(device=dev)

            if use_per_jet_weights:
                loss = loss_fn(pred, y)
                loss = loss * weight
            else:
                loss = loss_fn(pred, y).item()
            loss_test += loss.sum().item()
            accuracy = (pred.argmax(dim=1) == y).type(torch.float32)
            accuracy_test += accuracy.sum().item()

    loss_test /= num_jets_test
    accuracy_test /= num_jets_test
    print("Test: Avg loss = %1.6f, accuracy = %1.2f" % (loss_test, 100 * accuracy_test))

    return loss_train, loss_test


@hydra.main(config_path="../config", config_name="lrfinderParticleTransformer", version_base=None)
def lrfinderParticleTransformer(train_cfg: DictConfig) -> None:
    print("<lrfinderParticleTransformer>:")

    filelist_train = get_split_files("config/datasets/train.yaml", "train")
    filelist_test = get_split_files("config/datasets/validation.yaml", "validation")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: %s" % dev)

    jsonFileName = "./config/ParticleTransformer_cfg.json"
    print("Loading model configuration from file: %s" % jsonFileName)
    ParticleTransformer_cfg = None
    if os.path.isfile(jsonFileName):
        jsonFile = open(jsonFileName, "r")
        ParticleTransformer_cfg = json.load(jsonFile)
        if "ParticleTransformer" not in ParticleTransformer_cfg.keys():
            raise RuntimeError("Failed to parse config file %s !!")
        ParticleTransformer_cfg = ParticleTransformer_cfg["ParticleTransformer"]
        for key, value in ParticleTransformer_cfg.items():
            print(" %s = " % key, value)
        jsonFile.close()
    else:
        raise RuntimeError("Failed to read config file %s !!")

    max_cands = ParticleTransformer_cfg["max_cands"]
    metric = ParticleTransformer_cfg["metric"]
    min_lr = train_cfg.min_lr
    max_lr = train_cfg.max_lr

    print("Starting to build training dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_train = ParticleTransformerDataset(filelist_train, max_num_files=train_cfg.max_num_files, max_cands=max_cands)
    print("Finished building training dataset.")
    print(" current time:", datetime.datetime.now())

    print("Starting to build validation dataset...")
    print(" current time:", datetime.datetime.now())
    dataset_test = ParticleTransformerDataset(
        filelist_test, max_num_files=train_cfg.max_num_files, metric=metric, max_cands=max_cands
    )
    print("Finished building validation dataset.")
    print(" current time:", datetime.datetime.now())

    dataloader_train = DataLoader(
        dataset_train, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=train_cfg.batch_size, num_workers=train_cfg.num_dataloader_workers, shuffle=True
    )

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

    points_lr = []
    points_loss_train = []
    points_loss_test = []
    lr = min_lr
    while lr < max_lr:
        print("Processing lr=%1.3e" % lr)
        print(" current time:", datetime.datetime.now())

        model = ParticleTransformer(
            input_dim=17,
            num_classes=2,
            use_pre_activation_pair=False,
            for_inference=False,
            use_amp=False,
            metric=metric,
            verbosity=train_cfg.verbosity,
        ).to(device=dev)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-2)

        loss_train, loss_test = train_one_epoch(
            0, dataloader_train, dataloader_test, model, dev, loss_fn, train_cfg.use_per_jet_weights, optimizer
        )

        points_lr.append(lr)
        points_loss_train.append(loss_train)
        points_loss_test.append(loss_test)

        lr *= 1.2

    (graph_train,) = plt.plot(points_lr, points_loss_train, "b", label="train")
    (graph_test,) = plt.plot(points_lr, points_loss_test, "r", label="test")
    plt.xscale("log")
    plt.legend(handles=[graph_train, graph_test])
    plt.xlabel("lr")
    plt.ylabel("loss")
    plt.savefig("lrfinderParticleTransformer.png")


if __name__ == "__main__":
    lrfinderParticleTransformer()
