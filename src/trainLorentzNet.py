#!/usr/bin/python3

import datetime
import hydra
import json
from omegaconf import DictConfig
import os
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader

from LorentzNetDataset import LorentzNetDataset
from LorentzNet import LorentzNet


def get_split_files(cfg_filename, split):
    with open(cfg_filename, "r") as cfg_file:
        data = yaml.safe_load(cfg_file)
        paths = data[split]["paths"]

        # FIXME: this is hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch-persistent for GPU training
        # paths = [p.replace("/local/laurits", "./data") for p in paths]
        return paths


def train_loop(dataloader, model, dev, loss_fn, optimizer):
    print("<train_loop>:")
    print("current time:", datetime.datetime.now())
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        scalars = X["scalars"].to(device=dev)
        x = X["x"].to(device=dev)
        y = y.squeeze(dim=1).to(device=dev)
        # print("shape(y) = ", y.shape)
        # print("y = ", y)
        pred = model(x, scalars)
        # print("shape(pred) = ", pred.shape)
        # print("pred = ", pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 100) == 0 or batch == (size - 1):
            loss, current = loss.item(), (batch + 1) * len(pred)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, dev, loss_fn):
    print("<test_loop>:")
    print("current time:", datetime.datetime.now())
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            scalars = X["scalars"].to(device=dev)
            x = X["x"].to(device=dev)
            y = y.squeeze(dim=1).to(device=dev)
            pred = model(x, scalars)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


@hydra.main(config_path="../config", config_name="trainLorentzNet", version_base=None)
def trainLorentzNet(train_cfg: DictConfig) -> None:
    print("<trainLorentzNet>:")

    filelist_train = get_split_files("config/datasets/train.yaml", "train")
    filelist_test = get_split_files("config/datasets/validation.yaml", "validation")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: %s" % dev)

    jsonFileName = "./config/LorentzNet_cfg.json"
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

    n_scalar = LorentzNet_cfg["n_scalar"]
    n_hidden = LorentzNet_cfg["n_hidden"]
    n_class = LorentzNet_cfg["n_class"]
    dropout = LorentzNet_cfg["dropout"]
    n_layers = LorentzNet_cfg["n_layers"]
    c_weight = LorentzNet_cfg["c_weight"]
    max_cands = LorentzNet_cfg["max_cands"]
    add_beams = LorentzNet_cfg["add_beams"]

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

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    print("Starting to build training dataset...")
    dataset_train = LorentzNetDataset(
        filelist_train, max_num_files=train_cfg.max_num_files, max_cands=max_cands, add_beams=add_beams
    )
    print("Finished building training dataset.")

    print("Starting to build validation dataset...")
    dataset_test = LorentzNetDataset(
        filelist_test, max_num_files=train_cfg.max_num_files, max_cands=max_cands, add_beams=add_beams
    )
    print("Finished building validation dataset.")

    dataloader_train = DataLoader(dataset_train, batch_size=train_cfg.batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=train_cfg.batch_size, shuffle=True)

    print("Starting training (%i epochs)..." % train_cfg.num_epochs)
    for t in range(train_cfg.num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader_train, model, dev, loss_fn, optimizer)
        test_loop(dataloader_test, model, dev, loss_fn)
    print("Finished training.")

    print("Saving model to file %s." % train_cfg.model_file)
    torch.save(model.state_dict(), train_cfg.model_file)
    print("Done.")


if __name__ == "__main__":
    trainLorentzNet()
