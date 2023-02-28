import awkward as ak
import json
import numpy as np
import os
import vector

import torch

from basicTauBuilder import BasicTauBuilder
from ParticleTransformer import ParticleTransformer
from ParticleTransformerDataset import buildParticleTransformerTensors
from hpsAlgoTools import comp_angle, comp_deltaEta, comp_deltaTheta, comp_deltaR


class ParticleTransformerTauBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/ParticleTransformer_cfg.json", verbosity=0):
        print("<ParticleTransformerTauBuilder::ParticleTransformerTauBuilder>:")
        super(BasicTauBuilder, self).__init__()
        if os.path.isfile(cfgFileName):
            cfgFile = open(cfgFileName, "r")
            cfg = json.load(cfgFile)
            if "ParticleTransformer" not in cfg.keys():
                raise RuntimeError("Failed to parse config file %s !!")
            self._builderConfig = cfg["ParticleTransformer"]
            for key, value in self._builderConfig.items():
                print(" %s = " % key, value)
            self.verbosity = verbosity
            cfgFile.close()
        else:
            raise RuntimeError("Failed to read config file %s !!")

        self.max_cands = self._builderConfig["max_cands"]
        metric = self._builderConfig["metric"]
        self.metric_dR = None
        self.metric_dEta = None
        if metric == "eta-phi":
            self.metric_dR = comp_deltaR
            self.metric_dEta = comp_deltaEta
        elif metric == "theta-phi":
            self.metric_dR = comp_angle
            self.metric_dEta = comp_deltaTheta
        else:
            raise RuntimeError("Invalid configuration parameter 'metric' = '%s' !!" % metric)

        self.model = ParticleTransformer(
            input_dim=17,
            num_classes=2,
            use_pre_activation_pair=False,
            for_inference=False,  # CV: keep same as for training and apply softmax function on NN output manually
            use_amp=False,
            metric=metric,
            verbosity=verbosity,
        )
        self.model.load_state_dict(
            torch.load("data/ParticleTransformer_model_2023MarXX.pt", map_location=torch.device("cpu"))
        )
        self.model.eval()

        self.verbosity = verbosity

    def processJets(self, data):
        print("<ParticleTransformerTauBuilder::processJets>:")

        data_jet_p4s = data["reco_jet_p4s"]
        jet_p4s = vector.awk(
            ak.zip({"px": data_jet_p4s.x, "py": data_jet_p4s.y, "pz": data_jet_p4s.z, "mass": data_jet_p4s.tau})
        )
        num_jets = len(data_jet_p4s)

        data_cand_p4s = data["reco_cand_p4s"]
        cand_p4s = vector.awk(
            ak.zip({"px": data_cand_p4s.x, "py": data_cand_p4s.y, "pz": data_cand_p4s.z, "mass": data_cand_p4s.tau})
        )
        data_cand_pdgIds = data["reco_cand_pdgIds"]
        data_cand_qs = data["reco_cand_charge"]
        data_cand_d0s = data["reco_cand_d0s"]
        data_cand_d0errs = data["reco_cand_d0errs"]
        data_cand_dzs = data["reco_cand_dzs"]
        data_cand_dzerrs = data["reco_cand_dzerrs"]

        v_tensors = []
        x_tensors = []
        node_mask_tensors = []
        for idx in range(num_jets):
            if self.verbosity >= 2 and (idx % 100) == 0:
                print("Processing entry %i" % idx)

            jet_p4 = jet_p4s[idx]

            jet_constituent_p4s = cand_p4s[idx]
            jet_constituent_pdgIds = data_cand_pdgIds[idx]
            jet_constituent_qs = data_cand_qs[idx]
            jet_constituent_d0s = data_cand_d0s[idx]
            jet_constituent_d0errs = data_cand_d0errs[idx]
            jet_constituent_dzs = data_cand_dzs[idx]
            jet_constituent_dzerrs = data_cand_dzerrs[idx]

            v_tensor, x_tensor, node_mask_tensor = buildParticleTransformerTensors(
                jet_p4,
                jet_constituent_p4s,
                jet_constituent_pdgIds,
                jet_constituent_qs,
                jet_constituent_d0s,
                jet_constituent_d0errs,
                jet_constituent_dzs,
                jet_constituent_dzerrs,
                self.metric_dR,
                self.metric_Eta,
                self.max_cands,
            )
            x_tensors.append(x_tensor)
            v_tensors.append(v_tensor)
            node_mask_tensors.append(node_mask_tensor)
        x_tensor = torch.stack(x_tensors, dim=0)
        v_tensor = torch.stack(v_tensors, dim=0)
        node_mask_tensor = torch.stack(node_mask_tensors, dim=0)
        if self.verbosity >= 4:
            print("shape(x_tensor) = ", x_tensor.shape)
            print("shape(v_tensor) = ", v_tensor.shape)
            print("shape(node_mask) = ", node_mask_tensor.shape)

        pred = self.model(x_tensor, v_tensor, node_mask_tensor)
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
