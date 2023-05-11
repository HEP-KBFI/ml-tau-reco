import awkward as ak
import json
import numpy as np
import os
import vector

import torch

from basicTauBuilder import BasicTauBuilder
from LorentzNet import LorentzNet
from LorentzNetDataset import buildLorentzNetTensors
from FeatureStandardization import FeatureStandardization
from sklearn.preprocessing import OneHotEncoder


class LorentzNetTauBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/LorentzNet_cfg.json", verbosity=0):
        print("<LorentzNetTauBuilder::LorentzNetTauBuilder>:")
        super(BasicTauBuilder, self).__init__()

        self.filename_model = "data/LorentzNet_model_wReweighting_2023Mar24_wPdgId.pt"
        self.filename_transform = ""

        if os.path.isfile(cfgFileName):
            cfgFile = open(cfgFileName, "r")
            cfg = json.load(cfgFile)
            if "LorentzNet" not in cfg.keys():
                raise RuntimeError("Failed to parse config file %s !!")
            self._builderConfig = cfg["LorentzNet"]
            for key, value in self._builderConfig.items():
                print(" %s = " % key, value)
            self.verbosity = verbosity
            cfgFile.close()
        else:
            raise RuntimeError("Failed to read config file %s !!")

        self.n_hidden = self._builderConfig["n_hidden"]
        self.n_class = self._builderConfig["n_class"]
        self.dropout = self._builderConfig["dropout"]
        self.n_layers = self._builderConfig["n_layers"]
        self.c_weight = self._builderConfig["c_weight"]
        self.max_cands = self._builderConfig["max_cands"]
        self.add_beams = self._builderConfig["add_beams"]
        self.use_pdgId = self._builderConfig["use_pdgId"]
        self.pdgId_embedding = None
        if self.use_pdgId:
            # CV: pdgId=111 added to work around the bug fixed in this commit:
            #       https://github.com/HEP-KBFI/ml-tau-reco/pull/135/files#diff-9b848ad8e5903b4346d4030ebe41a391612220637cdd302d30d34b3fa07c96ea
            #    (this work-around allows us to keep using old files)
            self.pdgId_embedding = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
                [[11], [13], [22], [111], [130], [211], [2212]]
            )
        self.n_scalar = 8 if self.use_pdgId else 2
        standardize_inputs = self._builderConfig["standardize_inputs"]
        self.min_jet_theta = self._builderConfig["min_jet_theta"]
        self.max_jet_theta = self._builderConfig["max_jet_theta"]
        self.min_jet_pt = self._builderConfig["min_jet_pt"]
        self.max_jet_pt = self._builderConfig["max_jet_pt"]

        self.transform = None
        if standardize_inputs:
            self.transform = FeatureStandardization(method=self._builderConfig["method_FeatureStandardization"], features=["x", "scalars"], feature_dim=2, verbosity=self.verbosity)
            self.transform.load_params(self.filename_transform)

        self.model = LorentzNet(
            n_scalar=self.n_scalar,
            n_hidden=self.n_hidden,
            n_class=self.n_class,
            dropout=self.dropout,
            n_layers=self.n_layers,
            c_weight=self.c_weight,
            verbosity=verbosity,
        )
        self.model.load_state_dict(torch.load(self.filename_model, map_location=torch.device("cpu")))
        self.model.eval()

        self.verbosity = verbosity

    def processJets(self, data):
        print("<LorentzNetTauBuilder::processJets>:")

        data_jet_p4s = data["reco_jet_p4s"]
        jet_p4s = vector.awk(
            ak.zip({"px": data_jet_p4s.x, "py": data_jet_p4s.y, "pz": data_jet_p4s.z, "mass": data_jet_p4s.tau})
        )
        num_jets = len(data_jet_p4s)

        data_cand_p4s = data["reco_cand_p4s"]
        cand_p4s = vector.awk(
            ak.zip({"px": data_cand_p4s.x, "py": data_cand_p4s.y, "pz": data_cand_p4s.z, "mass": data_cand_p4s.tau})
        )
        data_cand_pdgIds = data["reco_cand_pdg"]
        data_cand_qs = data["reco_cand_charge"]

        x_tensors = []
        scalars_tensors = []
        node_mask_tensors = []
        pred_mask_tensors = []
        for idx in range(num_jets):
            if self.verbosity >= 2 and (idx % 100) == 0:
                print("Processing entry %i" % idx)

            jet_p4 = jet_p4s[idx]
            # print("jet: pT = %1.2f, theta = %1.3f, phi = %1.3f, mass = %1.2f" % \
            #  (jet_p4.pt, jet_p4.theta, jet_p4.phi, jet_p4.mass))

            jet_constituent_p4s = cand_p4s[idx]
            jet_constituent_pdgIds = data_cand_pdgIds[idx]
            jet_constituent_qs = data_cand_qs[idx]
            x_tensor, _, scalars_tensor, _, node_mask_tensor = buildLorentzNetTensors(
                jet_constituent_p4s,
                jet_constituent_pdgIds,
                jet_constituent_qs,
                self.max_cands,
                self.add_beams,
                self.use_pdgId,
                self.pdgId_embedding,
            )
            x_tensors.append(x_tensor)
            scalars_tensors.append(scalars_tensor)
            node_mask_tensors.append(node_mask_tensor)

            pred_mask = None
            if (
                (self.min_jet_theta < 0.0 or jet_p4.theta >= self.min_jet_theta)
                and (self.max_jet_theta < 0.0 or jet_p4.theta <= self.max_jet_theta)
                and (self.min_jet_pt < 0.0 or jet_p4.pt >= self.min_jet_pt)
                and (self.max_jet_pt < 0.0 or jet_p4.pt <= self.max_jet_pt)
            ):
                pred_mask = 1.0
            else:
                pred_mask = 0.0
            pred_mask_tensors.append(torch.tensor(pred_mask, dtype=torch.float32))

        x_tensor = torch.stack(x_tensors, dim=0)
        scalars_tensor = torch.stack(scalars_tensors, dim=0)
        node_mask_tensor = torch.stack(node_mask_tensors, dim=0)
        pred_mask_tensor = torch.stack(pred_mask_tensors, dim=0)

        if self.transform:
            X = {
                "x": x_tensor,
                "scalars": scalars_tensor,
                "mask": node_mask_tensor,
            }
            X_transformed = self.transform(X)
            x_tensor = X_transformed["x"]
            scalars_tensor = X_transformed["scalars"]
            node_mask_tensor = X_transformed["mask"]

        if self.verbosity >= 4:
            print("shape(x_tensor) = ", x_tensor.shape)
            print("shape(scalars_tensor) = ", scalars_tensor.shape)
            print("shape(node_mask) = ", node_mask_tensor.shape)

        pred = self.model(x_tensor, scalars_tensor, node_mask_tensor)
        pred = torch.softmax(pred, dim=1)
        if self.verbosity >= 4:
            print("shape(pred) = ", pred.shape)
            print("pred = ", pred)
            print("shape(pred_mask) = ", pred_mask_tensor.shape)
            print("pred_mask = ", pred_mask_tensor)
        tauClassifier = pred[:, 1] * pred_mask_tensor
        if self.verbosity >= 4:
            print("shape(tauClassifier) = ", tauClassifier.shape)
            print("tauClassifier = ", tauClassifier)
        tauClassifier = list(tauClassifier.detach().numpy())
        assert num_jets == len(tauClassifier)

        tau_p4s = vector.awk(
            ak.zip(
                {
                    "px": data["reco_jet_p4s"].x,
                    "py": data["reco_jet_p4s"].y,
                    "pz": data["reco_jet_p4s"].z,
                    "mass": data["reco_jet_p4s"].tau,
                }
            )
        )
        tauSigCand_p4s = data["reco_cand_p4s"]
        tauCharges = np.zeros(num_jets)
        tau_decaymode = np.zeros(num_jets)

        return {
            "tau_p4s": tau_p4s,
            "tauSigCand_p4s": tauSigCand_p4s,
            "tauClassifier": tauClassifier,
            "tau_charge": tauCharges,
            "tau_decaymode": tau_decaymode,
        }
