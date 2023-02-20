import awkward as ak
import math
import vector

import torch
from torch.utils.data import Dataset
from LGEB import psi


def buildLorentzNetTensors(jet_constituent_p4s, max_cands, add_beams):
    jet_constituent_p4s = jet_constituent_p4s[: max_cands]
    jet_constituent_p4s_zipped = list(
        zip(jet_constituent_p4s.energy, jet_constituent_p4s.px, jet_constituent_p4s.py, jet_constituent_p4s.pz)
    )
    num_jet_constituents = int(len(jet_constituent_p4s_zipped))
    x_tensor = torch.tensor(jet_constituent_p4s_zipped, dtype=torch.float32)
    x_tensor = torch.nn.functional.pad(
        x_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0
    )

    scalars_tensor = psi(torch.tensor(jet_constituent_p4s.mass, dtype=torch.float32)).unsqueeze(-1)
    scalars_tensor = torch.nn.functional.pad(
        scalars_tensor, (0, 1, 0, max_cands - num_jet_constituents), "constant", 0.0
    )

    if add_beams:
        beam_mass = 1.0
        beam1_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, +1.0]
        beam2_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, -1.0]
        x_beams = torch.tensor([beam1_p4, beam2_p4], dtype=torch.float32)
        x_tensor = torch.cat([x_beams, x_tensor], dim=0)

        scalars_beams = psi(torch.tensor([beam_mass, beam_mass], dtype=torch.float32)).unsqueeze(-1)
        scalars_beams = torch.nn.functional.pad(scalars_beams, (1, 0), "constant", 0.0)
        scalars_tensor = torch.cat([scalars_beams, scalars_tensor], dim=0)
    
    return x_tensor, scalars_tensor


class LorentzNetDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, max_cands=50, add_beams=True):

        print("<LorentzNetDataset::LorentzNetDataset>:")
        print(" #files = %i" % len(filelist))
        print(" add_beams = %s" % add_beams)

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

        self.x_tensors = []
        self.scalars_tensors = []
        self.y_tensors = []

        self.num_jets = 0
        for file in filelist:
            print("Opening file %s." % file)
            data = ak.from_parquet(file)

            num_jets_in_file = len(data["reco_jet_p4s"])
            print("File %s contains %i entries." % (file, num_jets_in_file))

            data_cand_p4s = data["reco_cand_p4s"]
            cand_p4s = vector.awk(
                ak.zip({"px": data_cand_p4s.x, "py": data_cand_p4s.y, "pz": data_cand_p4s.z, "mass": data_cand_p4s.tau})
            )

            data_gen_tau_decaymodes = data["gen_jet_tau_decaymode"]

            for idx in range(num_jets_in_file):
                if idx > 0 and (idx % 10000) == 0:
                    print(" Processing entry %i" % idx)

                jet_constituent_p4s = cand_p4s[idx]
                x_tensor, scalars_tensor = buildLorentzNetTensors(jet_constituent_p4s, self.max_cands, self.add_beams)
                y_tensor = torch.tensor([1 if data_gen_tau_decaymodes[idx] != -1 else 0], dtype=torch.long)

                self.scalars_tensors.append(scalars_tensor)
                self.x_tensors.append(x_tensor)
                self.y_tensors.append(y_tensor)

            print("Closing file %s." % file)

            self.num_jets += num_jets_in_file

        print("Dataset contains %i entries." % self.num_jets)

        assert len(self.x_tensors) == self.num_jets
        assert len(self.scalars_tensors) == self.num_jets
        assert len(self.y_tensors) == self.num_jets

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        if idx < self.num_jets:
            return {"x": self.x_tensors[idx], "scalars": self.scalars_tensors[idx]}, self.y_tensors[idx]
        else:
            raise RuntimeError("Invalid idx = %i (num_jets = %i) !!" % (idx, self.num_jets))
