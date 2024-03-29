import awkward as ak
import json
import numpy as np
import os
import vector

import torch

from basicTauBuilder import BasicTauBuilder
from ParticleTransformer import ParticleTransformer
from ParticleTransformerDataset import buildParticleTransformerTensors
from FeatureStandardization import FeatureStandardization
from hpsAlgoTools import comp_angle3d, comp_deltaEta, comp_deltaTheta, comp_deltaR_etaphi, comp_deltaR_thetaphi


class ParticleTransformerTauBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/ParticleTransformer_cfg.json", verbosity=0):
        print("<ParticleTransformerTauBuilder::ParticleTransformerTauBuilder>:")
        super(BasicTauBuilder, self).__init__()

        filepath = "/home/veelken/ml-tau-reco/data/"
        # self.filename_model = os.path.join(filepath, "ParticleTransformer_model_wLifetime_2023May30.pt")
        self.filename_model = os.path.join(filepath, "ParticleTransformer_model_wLifetime_2023Jun22.pt")
        print(" filename_model = %s" % self.filename_model)
        self.filename_transform = os.path.join(
            # filepath, "ParticleTransformer_FeatureStandardization_wLifetime_2023May30.json"
            filepath,
            "ParticleTransformer_FeatureStandardization_wLifetime_2023Jun22.json",
        )
        print(" filename_transform = %s" % self.filename_transform)

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
        self.use_pdgId = self._builderConfig["use_pdgId"]
        self.use_lifetime = self._builderConfig["use_lifetime"]
        self.input_dim = 7
        if self.use_pdgId:
            self.input_dim += 6
        if self.use_lifetime:
            self.input_dim += 4
        metric = self._builderConfig["metric"]
        self.metric_dR_or_angle = None
        self.metric_dEta_or_dTheta = None
        if metric == "eta-phi":
            self.metric_dR_or_angle = comp_deltaR_etaphi
            self.metric_dEta_or_dTheta = comp_deltaEta
        elif metric == "theta-phi":
            self.metric_dR_or_angle = comp_deltaR_thetaphi
            self.metric_dEta_or_dTheta = comp_deltaTheta
        elif metric == "angle3d":
            self.metric_dR_or_angle = comp_angle3d
            self.metric_dEta_or_dTheta = comp_deltaTheta
        else:
            raise RuntimeError("Invalid configuration parameter 'metric' = '%s' !!" % metric)
        standardize_inputs = self._builderConfig["standardize_inputs"]
        self.min_jet_theta = self._builderConfig["min_jet_theta"]
        self.max_jet_theta = self._builderConfig["max_jet_theta"]
        self.min_jet_pt = self._builderConfig["min_jet_pt"]
        self.max_jet_pt = self._builderConfig["max_jet_pt"]

        self.transform = None
        if standardize_inputs:
            self.transform = FeatureStandardization(
                method=self._builderConfig["method_FeatureStandardization"],
                features=["x", "v"],
                feature_dim=1,
                verbosity=self.verbosity,
            )
            self.transform.load_params(self.filename_transform)

        self.model = ParticleTransformer(
            input_dim=self.input_dim,
            num_classes=2,
            use_pre_activation_pair=False,
            for_inference=False,  # CV: keep same as for training and apply softmax function on NN output manually
            use_amp=False,
            metric=metric,
            verbosity=verbosity,
        )
        self.model.load_state_dict(torch.load(self.filename_model, map_location=torch.device("cpu")))
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
        data_cand_pdgIds = data["reco_cand_pdg"]
        data_cand_qs = data["reco_cand_charge"]
        data_cand_d0s = data["reco_cand_dxy"]
        data_cand_d0errs = data["reco_cand_dxy_err"]
        data_cand_dzs = data["reco_cand_dz"]
        data_cand_dzerrs = data["reco_cand_dz_err"]

        x_tensors = []
        v_tensors = []
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
            jet_constituent_d0s = data_cand_d0s[idx]
            jet_constituent_d0errs = data_cand_d0errs[idx]
            jet_constituent_dzs = data_cand_dzs[idx]
            jet_constituent_dzerrs = data_cand_dzerrs[idx]

            x_tensor, _, v_tensor, _, node_mask_tensor = buildParticleTransformerTensors(
                jet_p4,
                jet_constituent_p4s,
                jet_constituent_pdgIds,
                jet_constituent_qs,
                jet_constituent_d0s,
                jet_constituent_d0errs,
                jet_constituent_dzs,
                jet_constituent_dzerrs,
                self.metric_dR_or_angle,
                self.metric_dEta_or_dTheta,
                self.max_cands,
                self.use_pdgId,
                self.use_lifetime,
            )
            x_tensors.append(x_tensor)
            v_tensors.append(v_tensor)
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
        v_tensor = torch.stack(v_tensors, dim=0)
        node_mask_tensor = torch.stack(node_mask_tensors, dim=0)
        pred_mask_tensor = torch.stack(pred_mask_tensors, dim=0)

        if self.transform:
            X = {
                "v": v_tensor,
                "x": x_tensor,
                "mask": node_mask_tensor,
            }
            X_transformed = self.transform(X)
            x_tensor = X_transformed["x"]
            v_tensor = X_transformed["v"]
            node_mask_tensor = X_transformed["mask"]

        if self.verbosity >= 4:
            print("shape(x_tensor) = ", x_tensor.shape)
            print("shape(v_tensor) = ", v_tensor.shape)
            print("shape(node_mask) = ", node_mask_tensor.shape)

        pred = self.model(x_tensor, v_tensor, node_mask_tensor)
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
