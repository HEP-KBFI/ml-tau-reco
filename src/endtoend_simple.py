import os
import hydra
import vector
import awkward as ak
import numpy as np
import yaml
import torch
import torch_geometric
from torch import nn
import sys
import sklearn
import sklearn.metrics
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
from taujetdataset import TauJetDataset

from basicTauBuilder import BasicTauBuilder

from torch.utils.tensorboard import SummaryWriter

from FocalLoss import FocalLoss

focal_loss = FocalLoss(gamma=2.0)
ISTEP_GLOBAL = 0


# feedforward network that transformes input_dim->output_dim
def ffn(input_dim, output_dim, width, act, dropout):
    return nn.Sequential(
        nn.Linear(input_dim, width),
        # nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        # nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        # nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        # nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        # nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, output_dim),
    )


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


class TauEndToEndSimple(nn.Module):
    def __init__(self, sparse_mode=False):
        super(TauEndToEndSimple, self).__init__()

        self.act = nn.ReLU
        self.act_obj = self.act()
        self.dropout = 0.1
        self.width = 512
        self.embedding_dim = 512
        self.sparse_mode = sparse_mode

        self.num_jet_features = 8
        self.num_pf_features = 36

        self.nn_pf_initialembedding = ffn(self.num_pf_features, self.embedding_dim, self.width, self.act, self.dropout)

        self.A_mean = torch.nn.Parameter(data=torch.Tensor(self.num_jet_features), requires_grad=False)
        self.A_std = torch.nn.Parameter(data=torch.Tensor(self.num_jet_features), requires_grad=False)
        self.B_mean = torch.nn.Parameter(data=torch.Tensor(self.num_pf_features), requires_grad=False)
        self.B_std = torch.nn.Parameter(data=torch.Tensor(self.num_pf_features), requires_grad=False)

        if self.sparse_mode:
            self.agg1 = torch_geometric.nn.MeanAggregation()
            self.agg2 = torch_geometric.nn.MaxAggregation()
            self.agg3 = torch_geometric.nn.StdAggregation()
            # self.agg4 = torch_geometric.nn.AttentionalAggregation(
            # ffn(self.embedding_dim, 1, self.width, self.act, self.dropout))

        self.nn_pred_istau = ffn(self.num_jet_features + 3 * self.embedding_dim, 2, self.width, self.act, self.dropout)
        self.nn_pred_p4 = ffn(self.num_jet_features + 3 * self.embedding_dim, 4, self.width, self.act, self.dropout)

    # forward function for training with pytorch geometric
    def forward_sparse(self, inputs):
        jet_features, jet_pf_features, jet_pf_features_batch = inputs

        jet_features_normed = jet_features - self.A_mean
        jet_features_normed = jet_features_normed / self.A_std
        jet_pf_features_normed = jet_pf_features - self.B_mean
        jet_pf_features_normed = jet_pf_features_normed / self.B_std

        pf_encoded = self.act_obj(self.nn_pf_initialembedding(jet_pf_features_normed))

        # # now collapse the PF information in each jet with a global attention layer
        jet_encoded1 = self.act_obj(self.agg1(pf_encoded, jet_pf_features_batch))
        jet_encoded2 = self.act_obj(self.agg2(pf_encoded, jet_pf_features_batch))
        jet_encoded3 = self.act_obj(self.agg3(pf_encoded, jet_pf_features_batch))
        # jet_encoded4 = self.act_obj(self.agg4(pf_encoded, jet_pf_features_batch))

        # get the list of per-jet features as a concat of
        jet_feats = torch.cat([jet_features_normed, jet_encoded1, jet_encoded2, jet_encoded3], axis=-1)

        # run a binary classification whether or not this jet is from a tau
        pred_istau = self.nn_pred_istau(jet_feats)

        # run a per-jet NN for visible energy prediction
        jet_p4 = jet_features[:, :4]
        pred_p4 = jet_p4 * self.nn_pred_p4(jet_feats)

        return pred_istau, pred_p4

    #    # custom forward function for HLS4ML export, assuming a single 3D input
    #    def forward_3d(self, inputs):
    #
    #        assert len(inputs.shape) == 3
    #        # njet = inputs.shape[0]  # number of jets in batch
    #        # npf_per_jet = inputs.shape[1]  # max PF candidates across jets + 1 (the jet itself)
    #        # nfeat = inputs.shape[2]  # features of the jets / PF candidates
    #
    #        # get the jet properties
    #        jet_feats_orig = inputs[:, 0, :8]
    #
    #        # get the PF properties of each jet
    #        pf_feats_orig = inputs[:, 1:, :]
    #
    #        jet_features_normed = jet_feats_orig - self.A_mean
    #        jet_features_normed = jet_features_normed / self.A_std
    #        jet_pf_features_normed = pf_feats_orig - self.B_mean
    #        jet_pf_features_normed = jet_pf_features_normed / self.B_std
    #
    #        # encode the PF elements with the FFN
    #        pf_encoded = self.act_obj(self.nn_pf_initialembedding(jet_pf_features_normed))
    #
    #        # aggregate PFs across jets (need to add masking here for a fully correct implementation)
    #        jet_encoded1 = self.act_obj(torch.mean(pf_encoded, axis=1))
    #        jet_encoded2 = self.act_obj(torch.max(pf_encoded, axis=1).values)
    #        jet_encoded3 = self.act_obj(torch.std(pf_encoded, axis=1))
    #
    #        # get the list of per-jet features as a concat of
    #        jet_features = torch.cat([jet_features_normed, jet_encoded1, jet_encoded2, jet_encoded3], axis=-1)
    #
    #        # run a binary classification whether or not this jet is from a tau
    #        pred_istau = self.nn_pred_istau(jet_features)
    #
    #        # run a per-jet NN for visible energy prediction
    #        jet_p4 = jet_feats_orig[:, :4]
    #        pred_p4 = jet_p4 * self.nn_pred_p4(jet_features)
    #
    #        ret = torch.concat([pred_istau, pred_p4], axis=-1)
    #
    #        return ret

    def forward(self, inputs):
        if self.sparse_mode:
            return self.forward_sparse(inputs)
        else:
            return self.forward_3d(inputs)


def weighted_huber_loss(pred_tau_p4, true_tau_p4, weights):
    loss_p4 = torch.nn.functional.huber_loss(input=pred_tau_p4, target=true_tau_p4, reduction="none")
    weighted_losses = loss_p4 * weights.unsqueeze(-1)
    return weighted_losses.mean()


def weighted_bce_with_logits(pred_istau, true_istau, weights):
    loss_cls = focal_loss(pred_istau, true_istau.long())
    weighted_loss_cls = loss_cls * weights
    return weighted_loss_cls.mean()


def model_loop(model, ds_loader, optimizer, scheduler, is_train, dev, tensorboard_writer):
    global ISTEP_GLOBAL
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
    for ibatch, batch in enumerate(ds_loader):
        optimizer.zero_grad()
        batch = batch.to(device=dev)
        pred_istau, pred_p4 = model((batch.jet_features, batch.jet_pf_features, batch.jet_pf_features_batch))
        true_p4 = batch.gen_tau_p4
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        pred_p4 = pred_p4 * true_istau.unsqueeze(-1)
        weights = batch.weight

        loss_p4 = 1e5 * weighted_huber_loss(pred_p4, true_p4, weights)
        loss_cls = 1e7 * weighted_bce_with_logits(pred_istau, true_istau, weights)

        loss = loss_cls + loss_p4
        if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()
            ISTEP_GLOBAL += 1
            tensorboard_writer.add_scalar("step/train_loss", loss.detach().cpu().item(), ISTEP_GLOBAL)
            tensorboard_writer.add_scalar("step/num_signal", torch.sum(true_istau).cpu().item(), ISTEP_GLOBAL)
        else:
            class_true.append(true_istau.cpu().numpy())
            class_pred.append(torch.softmax(pred_istau, axis=-1)[:, 1].detach().cpu().numpy())
        loss_cls_tot += loss_cls.detach().cpu().item()
        loss_p4_tot += loss_p4.detach().cpu().item()
        nsteps += 1
        njets += batch.jet_features.shape[0]
        print(loss.detach().cpu().item())
        sys.stdout.flush()
    if not is_train:
        class_true = np.concatenate(class_true)
        class_pred = np.concatenate(class_pred)
    return loss_cls_tot / njets, loss_p4_tot / njets, (class_true, class_pred)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SimpleDNNTauBuilder(BasicTauBuilder):
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
        data_obj = Batch.from_data_list(ds.process_file_data(jets), follow_batch=["jet_pf_features"])
        pred_istau, pred_p4 = self.model((data_obj.jet_features, data_obj.jet_pf_features, data_obj.jet_pf_features_batch))

        pred_istau = torch.softmax(pred_istau, axis=-1)[:, 1]
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


def get_split_files(config_path, split):
    with open(config_path, "r") as fi:
        data = yaml.safe_load(fi)
        paths = data[split]["paths"]
        return paths


# Loops over files in the base dataset, yields jets from each file in order
class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ds):
        super(MyIterableDataset).__init__()
        self.ds = ds

    def __iter__(self):

        # loop over files in the .pt dataset
        for ifile in range(len(self.ds)):

            # loop over jets in file
            for jet in self.ds.get(ifile):
                yield jet


@hydra.main(config_path="../config", config_name="endtoend_simple", version_base=None)
def main(cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]

    ds_train = TauJetDataset("data/dataset_train")
    ds_val = TauJetDataset("data/dataset_validation")

    # load a part of the training set to memory to get feature standardization coefficients
    train_data = [ds_train.get(i) for i in range(10)]
    train_data = sum(train_data, [])

    # extract mean and std of the jet and PF features in the training set
    A = torch.concat([x.jet_features for x in train_data])
    B = torch.concat([x.jet_pf_features for x in train_data])
    A_mean = torch.mean(A, axis=0)
    A_std = torch.std(A, axis=0)
    B_mean = torch.mean(B, axis=0)
    B_std = torch.std(B, axis=0)

    # define a dataset that yields jets from each file
    ds_train_iter = MyIterableDataset(ds_train)
    ds_val_iter = MyIterableDataset(ds_val)

    # batch the jets from the iterated dataset
    ds_train_loader = DataLoader(ds_train_iter, batch_size=cfg.batch_size, follow_batch=["jet_pf_features"])
    ds_val_loader = DataLoader(ds_val_iter, batch_size=cfg.batch_size, follow_batch=["jet_pf_features"])

    # just loop over the training dataset and print each data object
    for x in ds_train_loader:
        print(x)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = TauEndToEndSimple(sparse_mode=True).to(device=dev)
    # set the model mean and stddev parameters for data normalization
    model.A_mean[:] = A_mean[:]
    model.A_std[:] = A_std[:]
    model.B_mean[:] = B_mean[:]
    model.B_std[:] = B_std[:]
    print("params={}".format(count_parameters(model)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=cfg.lr, steps_per_epoch=len(ds_train_loader), epochs=cfg.epochs
    # )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    tensorboard_writer = SummaryWriter(outpath + "/tensorboard")
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    best_loss = np.inf
    for iepoch in range(cfg.epochs):
        loss_cls_train, loss_p4_train, _ = model_loop(
            model, ds_train_loader, optimizer, scheduler, True, dev, tensorboard_writer
        )
        tensorboard_writer.add_scalar("epoch/train_cls_loss", loss_cls_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_p4_loss", loss_p4_train, iepoch)
        tensorboard_writer.add_scalar("epoch/train_loss", loss_cls_train + loss_p4_train, iepoch)

        loss_cls_val, loss_p4_val, retvals = model_loop(
            model, ds_val_loader, optimizer, scheduler, False, dev, tensorboard_writer
        )
        loss_val = loss_cls_val + loss_p4_val

        tensorboard_writer.add_scalar("epoch/val_cls_loss", loss_cls_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_p4_loss", loss_p4_val, iepoch)
        tensorboard_writer.add_scalar("epoch/val_loss", loss_val, iepoch)

        tensorboard_writer.add_scalar("epoch/lr", scheduler.get_last_lr()[0], iepoch)
        tensorboard_writer.add_pr_curve("epoch/roc_curve", retvals[0], retvals[1], iepoch)

        fpr, tpr, thresh = sklearn.metrics.roc_curve(retvals[0], retvals[1])
        tensorboard_writer.add_scalar("epoch/fpr_at_tpr0p6", fpr[np.searchsorted(tpr, 0.6)], iepoch)

        print(
            "epoch={} cls={:.4f}/{:.4f} p4={:.4f}/{:.4f}".format(
                iepoch, loss_cls_train, loss_cls_val, loss_p4_train, loss_p4_val
            )
        )

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model, "{}/model_best.pt".format(outpath))

        if early_stopper.early_stop(loss_cls_val):
            break

    model = model.to(device="cpu")
    torch.save(model, "data/model.pt")


if __name__ == "__main__":
    main()
