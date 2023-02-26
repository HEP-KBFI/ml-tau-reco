import awkward as ak
import math
import vector

import torch
from torch.utils.data import Dataset


from hpsAlgoTools import comp_angle, comp_deltaEta, comp_deltaPhi, comp_deltaTheta, comp_deltaR


def buildParticleTransformerTensors(
    jet_p4,
    jet_constituent_p4s,
    jet_constituent_pdgIds,
    jet_constituent_qs,
    jet_constituent_d0s,
    jet_constituent_d0errs,
    jet_constituent_dzs,
    jet_constituent_dzerrs,
    metric_dR_or_angle,
    metric_dEta_or_dTheta,
    max_cands,
):
    jet_constituent_p4s = jet_constituent_p4s[:max_cands]
    jet_constituent_p4s_zipped = list(
        zip(jet_constituent_p4s.px, jet_constituent_p4s.py, jet_constituent_p4s.pz, jet_constituent_p4s.energy)
    )
    num_jet_constituents = int(len(jet_constituent_p4s_zipped))
    v_tensor = torch.tensor(jet_constituent_p4s_zipped, dtype=torch.float32)
    v_tensor = torch.nn.functional.pad(v_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0)

    jet_constituent_features = []
    for idx in range(num_jet_constituents):
        jet_constituent_p4 = jet_constituent_p4s[idx]
        jet_constituent_pdgId = jet_constituent_p4s[idx]
        jet_constituent_abs_pdgId = abs(jet_constituent_pdgId)
        jet_constituent_q = jet_constituent_p4s[idx]
        jet_constituent_d0 = jet_constituent_p4s[idx]
        jet_constituent_d0err = jet_constituent_p4s[idx]
        jet_constituent_dz = jet_constituent_p4s[idx]
        jet_constituent_dzerr = jet_constituent_p4s[idx]

        part_pt_log = math.log(jet_constituent_p4.pt)
        part_e_log = math.log(jet_constituent_p4.energy)
        part_logptrel = math.log(jet_constituent_p4.pt / jet_p4.pt)
        part_logerel = math.log(jet_constituent_p4.energy / jet_p4.energy)
        part_deltaR = metric_dR_or_angle(jet_constituent_p4, jet_p4)
        part_charge = jet_constituent_q
        part_isChargedHadron = 1.0 if jet_constituent_abs_pdgId == 211 else 0.0
        part_isNeutralHadron = 1.0 if jet_constituent_abs_pdgId in [130, 2112] else 0.0
        part_isPhoton = 1.0 if jet_constituent_abs_pdgId == 22 else 0.0
        part_isElectron = 1.0 if jet_constituent_abs_pdgId == 11 else 0.0
        part_isMuon = 1.0 if jet_constituent_abs_pdgId == 13 else 0.0
        part_d0 = 0.0
        part_d0err = 0.0
        part_dz = 0.0
        part_dzerr = 0.0
        if abs(part_charge) > 0.5 and part_d0 > -99.0 and part_dz > -99.0:
            part_d0 = math.tanh(jet_constituent_d0)
            part_d0err = jet_constituent_d0err
            part_dz = math.tanh(jet_constituent_dz)
            part_dzerr = jet_constituent_dzerr
        part_deta = metric_dEta_or_dTheta(jet_constituent_p4, jet_p4)
        part_dphi = comp_deltaPhi(jet_constituent_p4, jet_p4)
        jet_constituent_features.append(
            [
                part_pt_log,
                part_e_log,
                part_logptrel,
                part_logerel,
                part_deltaR,
                part_charge,
                part_isChargedHadron,
                part_isNeutralHadron,
                part_isPhoton,
                part_isElectron,
                part_isMuon,
                part_d0,
                part_d0err,
                part_dz,
                part_dzerr,
                part_deta,
                part_dphi,
            ]
        )
    x_tensor = torch.tensor(jet_constituent_p4s_zipped, dtype=torch.float32)
    x_tensor = torch.nn.functional.pad(v_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0)

    node_mask_tensor = torch.ones(num_jet_constituents, dtype=torch.float32)
    node_mask_tensor = torch.nn.functional.pad(node_mask_tensor, (0, max_cands - num_jet_constituents), "constant", 0.0)
    node_mask_tensor = torch.unsqueeze(node_mask_tensor, dim=-1)

    return v_tensor, x_tensor, node_mask_tensor


class ParticleTransformerDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, max_cands=50, metric="eta-phi"):
        print("<ParticleTransformerDataset::ParticleTransformerDataset>:")
        print(" #files = %i" % len(filelist))
        print(" max_cands = %i" % max_cands)
        print(" metric = '%s'" % metric)

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

        if max_num_files != -1:
            num_sig_files = 0
            num_bgr_files = 0
            restricted_filelist = []
            for file in filelist:
                if "ZH_Htautau" in file:
                    if num_sig_files < max_num_files:
                        restricted_filelist.append(file)
                        num_sig_files += 1
                elif "QCD" in file:
                    if num_bgr_files < max_num_files:
                        restricted_filelist.append(file)
                        num_bgr_files += 1
                else:
                    raise RuntimeError("Failed to parse filename = '%s' !!" % file)
            filelist = restricted_filelist
            print("Restricting the size of the dataset to %i files." % len(filelist))

        self.filelist = filelist
        self.max_cands = max_cands

        self.v_tensors = []
        self.x_tensors = []
        self.node_mask_tensors = []
        self.y_tensors = []

        self.num_jets = 0
        for file in filelist:
            print("Opening file %s." % file)
            data = ak.from_parquet(file)

            data_jet_p4s = data["reco_jet_p4s"]
            jet_p4s = vector.awk(
                ak.zip({"px": data_jet_p4s.x, "py": data_jet_p4s.y, "pz": data_jet_p4s.z, "mass": data_jet_p4s.tau})
            )
            num_jets_in_file = len(data_jet_p4s)
            print("File %s contains %i entries." % (file, num_jets_in_file))

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

            data_gen_tau_decaymodes = data["gen_jet_tau_decaymode"]

            for idx in range(num_jets_in_file):
                if idx > 0 and (idx % 10000) == 0:
                    print(" Processing entry %i" % idx)

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
                y_tensor = torch.tensor([1 if data_gen_tau_decaymodes[idx] != -1 else 0], dtype=torch.long)

                self.v_tensors.append(v_tensor)
                self.x_tensors.append(x_tensor)
                self.node_mask_tensors.append(node_mask_tensor)
                self.y_tensors.append(y_tensor)

            print("Closing file %s." % file)

            self.num_jets += num_jets_in_file

        print("Dataset contains %i entries." % self.num_jets)

        assert len(self.v_tensors) == self.num_jets
        assert len(self.x_tensors) == self.num_jets
        assert len(self.node_mask_tensors) == self.num_jets
        assert len(self.y_tensors) == self.num_jets

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return {"v": self.v_tensors[idx], "x": self.x_tensors[idx], "mask": self.node_mask_tensors[idx]}, self.y_tensors[
                idx
            ]
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))