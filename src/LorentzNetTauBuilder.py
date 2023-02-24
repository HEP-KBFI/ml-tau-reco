import awkward as ak
import json
import numpy as np
import os
import vector

import torch

from basicTauBuilder import BasicTauBuilder
from LorentzNet import LorentzNet
from LorentzNetDataset import buildLorentzNetTensors


class LorentzNetTauBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/LorentzNet_cfg.json", verbosity=0):
        print("<LorentzNetTauBuilder::LorentzNetTauBuilder>:")
        super(BasicTauBuilder, self).__init__()
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

        self.n_scalar = self._builderConfig["n_scalar"]
        self.n_hidden = self._builderConfig["n_hidden"]
        self.n_class = self._builderConfig["n_class"]
        self.dropout = self._builderConfig["dropout"]
        self.n_layers = self._builderConfig["n_layers"]
        self.c_weight = self._builderConfig["c_weight"]
        self.max_cands = self._builderConfig["max_cands"]
        self.add_beams = self._builderConfig["add_beams"]

        self.model = LorentzNet(
            n_scalar=self.n_scalar,
            n_hidden=self.n_hidden,
            n_class=self.n_class,
            dropout=self.dropout,
            n_layers=self.n_layers,
            c_weight=self.c_weight,
            verbosity=verbosity,
        )
        self.model.load_state_dict(torch.load("data/LorentzNet_model_2023Feb22.pt", map_location=torch.device("cpu")))
        self.model.eval()

        self.verbosity = verbosity

    def processJets(self, data):
        print("<LorentzNetTauBuilder::processJets>:")

        num_jets = len(data["reco_jet_p4s"])

        data_cand_p4s = data["reco_cand_p4s"]
        cand_p4s = vector.awk(
            ak.zip({"px": data_cand_p4s.x, "py": data_cand_p4s.y, "pz": data_cand_p4s.z, "mass": data_cand_p4s.tau})
        )

        x_tensors = []
        scalars_tensors = []
        node_mask_tensors = []
        for idx in range(num_jets):
            if self.verbosity >= 2 and (idx % 100) == 0:
                print("Processing entry %i" % idx)

            jet_constituent_p4s = cand_p4s[idx]
            x_tensor, scalars_tensor, node_mask_tensor = buildLorentzNetTensors(
                jet_constituent_p4s, self.max_cands, self.add_beams
            )
            x_tensors.append(x_tensor)
            scalars_tensors.append(scalars_tensor)
            node_mask_tensors.append(node_mask_tensor)
        x_tensor = torch.stack(x_tensors, dim=0)
        scalars_tensor = torch.stack(scalars_tensors, dim=0)
        node_mask_tensor = torch.stack(node_mask_tensors, dim=0)
        if self.verbosity >= 4:
            print("shape(x_tensor) = ", x_tensor.shape)
            print("shape(scalars_tensor) = ", scalars_tensor.shape)
            print("shape(node_mask) = ", node_mask_tensor.shape)

        pred = self.model(x_tensor, scalars_tensor, node_mask_tensor)
        pred = torch.softmax(pred, dim=1)
        if self.verbosity >= 4:
            print("shape(pred) = ", pred.shape)
            print("pred = ", pred)
        tauClassifier = list(pred[:, 1].detach().numpy())
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
