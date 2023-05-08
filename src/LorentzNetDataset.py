import awkward as ak
import math
import vector

import torch
from torch.utils.data import Dataset
from LGEB import psi
from sklearn.preprocessing import OneHotEncoder


def buildLorentzNetTensors(
    jet_constituent_p4s, jet_constituent_pdgIds, jet_constituent_qs, max_cands, add_beams, use_pdgId, pdgId_embedding
):
    # print("<buildLorentzNetTensors>:")

    jet_constituent_p4s = jet_constituent_p4s[:max_cands]
    jet_constituent_p4s_zipped = list(
        zip(jet_constituent_p4s.energy, jet_constituent_p4s.px, jet_constituent_p4s.py, jet_constituent_p4s.pz)
    )
    num_jet_constituents = int(len(jet_constituent_p4s_zipped))
    x_tensor = torch.tensor(jet_constituent_p4s_zipped, dtype=torch.float32)
    x_tensor = torch.nn.functional.pad(x_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0)

    scalars_tensor = None
    if use_pdgId:
        jet_constituent_abs_pdgIds = abs(jet_constituent_pdgIds)
        jet_constituent_abs_pdgIds = [[jet_constituent_pdgId] for jet_constituent_pdgId in jet_constituent_pdgIds]
        one_hot = pdgId_embedding.transform(jet_constituent_abs_pdgIds[:max_cands])
        one_hot_tensor = torch.tensor(one_hot, dtype=torch.float32)
        charge_tensor = torch.tensor(jet_constituent_qs[:max_cands], dtype=torch.float32).unsqueeze(-1)
        scalars_tensor = torch.cat((one_hot_tensor, charge_tensor), dim=1)
        scalars_tensor = torch.nn.functional.pad(
            scalars_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0
        )
    else:
        scalars_tensor = psi(torch.tensor(jet_constituent_p4s.mass, dtype=torch.float32)).unsqueeze(-1)
        scalars_tensor = torch.nn.functional.pad(
            scalars_tensor, (0, 1, 0, max_cands - num_jet_constituents), "constant", 0.0
        )

    node_mask_tensor = torch.ones(num_jet_constituents, dtype=torch.float32)
    node_mask_tensor = torch.nn.functional.pad(node_mask_tensor, (0, max_cands - num_jet_constituents), "constant", 0.0)

    if add_beams:
        beam_mass = 1.0
        beam1_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, +1.0]
        beam2_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, -1.0]
        x_beams = torch.tensor([beam1_p4, beam2_p4], dtype=torch.float32)
        x_tensor = torch.cat([x_beams, x_tensor], dim=0)

        if use_pdgId:
            one_hot = pdgId_embedding.transform([[2212], [2212]])
            one_hot_beams = torch.tensor(one_hot, dtype=torch.float32)
            charge_beams = torch.tensor([+1.0, -1.0], dtype=torch.float32).unsqueeze(-1)
            scalars_beams = torch.cat((one_hot_beams, charge_beams), dim=1)
            scalars_tensor = torch.cat([scalars_beams, scalars_tensor], dim=0)
        else:
            scalars_beams = psi(torch.tensor([beam_mass, beam_mass], dtype=torch.float32)).unsqueeze(-1)
            scalars_beams = torch.nn.functional.pad(scalars_beams, (1, 0), "constant", 0.0)
            scalars_tensor = torch.cat([scalars_beams, scalars_tensor], dim=0)

        node_mask_beams = torch.ones(2, dtype=torch.float32)
        node_mask_tensor = torch.cat([node_mask_beams, node_mask_tensor], dim=0)

    node_mask_tensor = torch.unsqueeze(node_mask_tensor, dim=-1)

    return x_tensor, scalars_tensor, node_mask_tensor


def read_cut(cuts, key):
    if key in cuts.keys():
        return cuts[key]
    else:
        return -1.0


class LorentzNetDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, max_cands=50, add_beams=True, use_pdgId=False, preselection={}):

        print("<LorentzNetDataset::LorentzNetDataset>:")
        print(" #files = %i" % len(filelist))
        print(" max_cands = %i" % max_cands)
        print(" add_beams = %s" % add_beams)
        print(" use_pdgId = %s" % use_pdgId)

        self.min_jet_theta = read_cut(preselection, "min_jet_theta")
        self.max_jet_theta = read_cut(preselection, "max_jet_theta")
        self.min_jet_pt = read_cut(preselection, "min_jet_pt")
        self.max_jet_pt = read_cut(preselection, "max_jet_pt")
        print(" min_jet_theta = %1.3f" % self.min_jet_theta)
        print(" max_jet_theta = %1.3f" % self.max_jet_theta)
        print(" min_jet_pt = %1.3f" % self.min_jet_pt)
        print(" max_jet_pt = %1.3f" % self.max_jet_pt)

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
        self.add_beams = add_beams
        self.use_pdgId = use_pdgId
        self.pdgId_embedding = None
        if self.use_pdgId:
            # CV: pdgId=111 added to work around the bug fixed in this commit:
            #       https://github.com/HEP-KBFI/ml-tau-reco/pull/135/files#diff-9b848ad8e5903b4346d4030ebe41a391612220637cdd302d30d34b3fa07c96ea
            #    (this work-around allows us to keep using old files)
            self.pdgId_embedding = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(               
                [[11], [13], [22], [111], [130], [211], [2212]]
            )

        self.x_tensors = []
        self.scalars_tensors = []
        self.node_mask_tensors = []
        self.y_tensors = []
        self.weight_tensors = []

        self.num_jets = 0
        for file in filelist:
            print("Opening file %s" % file)
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
            data_cand_pdgIds = data["reco_cand_pdg"]
            data_cand_qs = data["reco_cand_charge"]

            data_gen_tau_decaymodes = data["gen_jet_tau_decaymode"]

            data_weights = data["weight"]

            for idx in range(num_jets_in_file):
                if idx > 0 and (idx % 10000) == 0:
                    print(" Processing entry %i" % idx)

                jet_p4 = jet_p4s[idx]
                if not (
                    (self.min_jet_theta < 0.0 or jet_p4.theta >= self.min_jet_theta)
                    and (self.max_jet_theta < 0.0 or jet_p4.theta <= self.max_jet_theta)
                    and (self.min_jet_pt < 0.0 or jet_p4.pt >= self.min_jet_pt)
                    and (self.max_jet_pt < 0.0 or jet_p4.pt <= self.max_jet_pt)
                ):
                    continue

                jet_constituent_p4s = cand_p4s[idx]
                jet_constituent_pdgIds = data_cand_pdgIds[idx]
                jet_constituent_qs = data_cand_qs[idx]
                x_tensor, scalars_tensor, node_mask_tensor = buildLorentzNetTensors(
                    jet_constituent_p4s,
                    jet_constituent_pdgIds,
                    jet_constituent_qs,
                    self.max_cands,
                    self.add_beams,
                    self.use_pdgId,
                    self.pdgId_embedding,
                )

                y_tensor = torch.tensor([1 if data_gen_tau_decaymodes[idx] != -1 else 0], dtype=torch.long)
                weight_tensor = torch.tensor([data_weights[idx]], dtype=torch.float32)

                self.x_tensors.append(x_tensor)
                self.scalars_tensors.append(scalars_tensor)
                self.node_mask_tensors.append(node_mask_tensor)
                self.y_tensors.append(y_tensor)
                self.weight_tensors.append(weight_tensor)

                self.num_jets += 1

            print("Closing file %s" % file)

        print("Dataset contains %i entries." % self.num_jets)

        assert len(self.x_tensors) == self.num_jets
        assert len(self.scalars_tensors) == self.num_jets
        assert len(self.node_mask_tensors) == self.num_jets
        assert len(self.y_tensors) == self.num_jets
        assert len(self.weight_tensors) == self.num_jets

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return (
                {
                    "x": self.x_tensors[idx],
                    "scalars": self.scalars_tensors[idx],
                    "mask": self.node_mask_tensors[idx],
                },
                self.y_tensors[idx],
                self.weight_tensors[idx],
            )
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))
