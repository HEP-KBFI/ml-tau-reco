import os
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from glob import glob

from LorentzNetDataset import LorentzNetDataset
from LorentzNet import LorentzNet

print("<trainLorentzNet>:")

model_file = "LorentzNet_model.pt"

def get_split_files(cfg_filename, split):
    with open(cfg_filename, "r") as cfg_file:
        data = yaml.safe_load(cfg_file)
        paths = data[split]["paths"]

        # FIXME: this is hardcoded, /local is too slow for GPU training
        # datasets should be kept in /home or /scratch-persistent for GPU training
        #paths = [p.replace("/local/laurits", "./data") for p in paths]
        return paths

filelist_train = get_split_files("config/datasets/train.yaml", "train")
filelist_test = get_split_files("config/datasets/validation.yaml", "validation")

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: %s" % dev)

print("Building model...")
model = LorentzNet(n_scalar=2, n_hidden=72, n_class=2, dropout=0.2, n_layers=6, c_weight=0.005).to(device=dev)
print("Finished building model:")
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

print("Starting to build training dataset...")
dataset_train = LorentzNetDataset(filelist_train, max_num_files=1, add_beams=True)
print("Finished building training dataset.")

print("Starting to build validation dataset...")
dataset_test = LorentzNetDataset(filelist_test, max_num_files=1, add_beams=True)
print("Finished building validation dataset.")

dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        scalars = X['scalars'].to(device=dev)
        x = X['x'].to(device=dev)
        y = y.squeeze(dim=1)
        #print("shape(y) = ", y.shape)
        #print("y = ", y)
        pred = model(scalars, x)
        #print("shape(pred) = ", pred.shape)
        #print("pred = ", pred)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(scalars=X['scalars'], x=X['x'])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

print("Starting training...")
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader_train, model, loss_fn, optimizer)
    test_loop(dataloader_test, model, loss_fn)
print("Finished training.")

print("Saving model to file %s." % model_file)
torch.save(model.state_dict(), model_file)
print("Done.")


