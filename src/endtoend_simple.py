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
import tqdm
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
        torch.nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        torch.nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        torch.nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        torch.nn.LayerNorm(width),
        act(),
        nn.Dropout(dropout),
        nn.Linear(width, width),
        torch.nn.LayerNorm(width),
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


# self-attention layer that transformes [B, N, x] -> [B, N, embedding_dim]
# by attending the N elements in each batch B
class SelfAttentionLayer(nn.Module):
    def __init__(self, embedding_dim=32, num_heads=32, width=128, dropout=0.3):
        super(SelfAttentionLayer, self).__init__()
        self.act = nn.ELU
        self.act_obj = self.act()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm0 = torch.nn.LayerNorm(embedding_dim)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.seq = ffn(embedding_dim, embedding_dim, width, self.act, dropout)

    def forward(self, x, mask):
        xa = self.mha(x, x, x, key_padding_mask=mask)[0]
        x = self.norm0(x + xa)
        x = self.act_obj(x)

        xa = self.seq(x)
        x = self.norm1(x + xa)
        x = self.act_obj(x)
        x = self.dropout(x)

        # make sure masked elements are 0
        x = x * (~mask.unsqueeze(-1))
        return x


class TauEndToEndSimple(nn.Module):
    def __init__(
        self,
    ):
        super(TauEndToEndSimple, self).__init__()

        self.act = nn.ELU
        self.act_obj = self.act()
        self.dropout = 0.1
        self.width = 512
        self.embedding_dim = 512

        # if set to True, disables aggregation across the batch
        # and replaces it with a fake version, just to test that ONNX export
        # otherwise works. Need to replace this with something meaningful for
        # an actually viable ONNX export!
        self.onnx_workaround_agg = False

        self.nn_pf_initialembedding = ffn(14 + 22, self.embedding_dim, self.width, self.act, self.dropout)

        # self.nn_pf_mha = nn.ModuleList()
        # for i in range(1):
        #     self.nn_pf_mha.append(
        #         SelfAttentionLayer(embedding_dim=self.embedding_dim, width=self.width, dropout=self.dropout)
        #     )

        self.agg1 = torch_geometric.nn.MeanAggregation()
        self.agg2 = torch_geometric.nn.MaxAggregation()
        self.agg3 = torch_geometric.nn.StdAggregation()
        self.agg4 = torch_geometric.nn.SoftmaxAggregation(learn=True)

        self.nn_pred_istau = ffn(8 + 4 * self.embedding_dim, 2, self.width, self.act, self.dropout)
        self.nn_pred_p4 = ffn(8 + 4 * self.embedding_dim, 4, self.width, self.act, self.dropout)

    def forward(self, jet_features, jet_pf_features, jet_pf_features_batch):
        pf_encoded = self.act_obj(self.nn_pf_initialembedding(jet_pf_features))

        # # pad the PF tensors across jets
        # pfs_padded, mask = torch_geometric.utils.to_dense_batch(pf_encoded, batch.jet_pf_features_batch, 0.0)

        # # run a simple self-attention over the PF candidates in each jet
        # for mha_layer in self.nn_pf_mha:
        #     pfs_padded = self.act_obj(mha_layer(pfs_padded, ~mask))

        # # get the encoded PF candidates after attention, undo the padding
        # pf_encoded = torch.cat([pfs_padded[i][mask[i]] for i in range(pfs_padded.shape[0])])

        # assert pf_encoded.shape[0] == batch.jet_pf_features.shape[0]

        # # now collapse the PF information in each jet with a global attention layer
        if self.onnx_workaround_agg:
            jet_encoded1 = self.act_obj(torch.mean(pf_encoded, axis=0).unsqueeze(axis=0).repeat(jet_features.shape[0], 1))
            jet_encoded2 = self.act_obj(torch.mean(pf_encoded, axis=0).unsqueeze(axis=0).repeat(jet_features.shape[0], 1))
            jet_encoded3 = self.act_obj(torch.mean(pf_encoded, axis=0).unsqueeze(axis=0).repeat(jet_features.shape[0], 1))
            jet_encoded4 = self.act_obj(torch.mean(pf_encoded, axis=0).unsqueeze(axis=0).repeat(jet_features.shape[0], 1))
        else:
            jet_encoded1 = self.act_obj(self.agg1(pf_encoded, jet_pf_features_batch))
            jet_encoded2 = self.act_obj(self.agg2(pf_encoded, jet_pf_features_batch))
            jet_encoded3 = self.act_obj(self.agg3(pf_encoded, jet_pf_features_batch))
            jet_encoded4 = self.act_obj(self.agg4(pf_encoded, jet_pf_features_batch))

        # get the list of per-jet features as a concat of
        jet_feats = torch.cat([jet_features, jet_encoded1, jet_encoded2, jet_encoded3, jet_encoded4], axis=-1)

        # run a binary classification whether or not this jet is from a tau
        pred_istau = self.nn_pred_istau(jet_feats)

        # run a per-jet NN for visible energy prediction
        jet_p4 = jet_features[:, :4]
        pred_p4 = jet_p4 * self.nn_pred_p4(jet_feats)

        return pred_istau, pred_p4


def weighted_huber_loss(pred_tau_p4, true_tau_p4, weights):
    loss_p4 = torch.nn.functional.huber_loss(input=pred_tau_p4, target=true_tau_p4, reduction="none")
    weighted_losses = loss_p4 * weights.unsqueeze(-1)
    return weighted_losses.mean()


def weighted_bce_with_logits(pred_istau, true_istau, weights):
    loss_cls = 10000.0 * focal_loss(pred_istau, true_istau.long())
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
    for ibatch, batch in enumerate(tqdm.tqdm(ds_loader, total=len(ds_loader))):
        optimizer.zero_grad()
        batch = batch.to(device=dev)
        pred_istau, pred_p4 = model(batch.jet_features, batch.jet_pf_features, batch.jet_pf_features_batch)
        true_p4 = batch.gen_tau_p4
        true_istau = (batch.gen_tau_decaymode != -1).to(dtype=torch.float32)
        pred_p4 = pred_p4 * true_istau.unsqueeze(-1)
        weights = batch.weight

        loss_p4 = weighted_huber_loss(pred_p4, true_p4, weights)
        loss_cls = weighted_bce_with_logits(pred_istau, true_istau, weights)

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
        # print(
        #    "jets={jets} pfs={pfs} max_pfs={max_pfs} ntau={ntau} loss={loss:.2f} lr={lr:.2E}".format(
        #        jets=batch.jet_features.shape[0],
        #        pfs=batch.jet_pf_features.shape[0],
        #        max_pfs=np.max(np.unique(batch.jet_pf_features_batch.cpu().numpy(), return_counts=True)[1]),
        #        ntau=true_istau.sum().cpu().item(),
        #        loss=loss.detach().cpu().item(),
        #        lr=scheduler.get_last_lr()[0],
        #    )
        # )
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
        pred_istau, pred_p4 = self.model(data_obj.jet_features, data_obj.jet_pf_features, data_obj.jet_pf_features_batch)

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


@hydra.main(config_path="../config", config_name="endtoend_simple", version_base=None)
def main(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    outpath = hydra_cfg["runtime"]["output_dir"]

    files_train = get_split_files(cfg.train_files, "train")
    files_val = get_split_files(cfg.validation_files, "validation")

    if cfg.n_files == -1:
        n_files = None
    else:
        n_files = cfg.n_files

    ds_train = TauJetDataset(files_train[:n_files])
    ds_val = TauJetDataset(files_val[:n_files])

    print("Loaded TauJetDataset with {} train steps".format(len(ds_train)))
    print("Loaded TauJetDataset with {} val steps".format(len(ds_val)))
    ds_train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, follow_batch=["jet_pf_features"])
    ds_val_loader = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=True, follow_batch=["jet_pf_features"])

    assert len(ds_train_loader) > 0
    assert len(ds_val_loader) > 0
    print("train={} val={}".format(len(ds_train_loader), len(ds_val_loader)))

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print("device={}".format(dev))

    model = TauEndToEndSimple().to(device=dev)
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
