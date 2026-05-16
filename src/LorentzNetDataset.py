import torch
from torch.utils.data import Dataset
import glob
import os
import bisect
import gc
import math
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
    x_is_one_hot_encoded = [False, False, False, False]

    scalars_tensor = None
    scalars_is_one_hot_encoded = None
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
        scalars_is_one_hot_encoded = []
        for idx in range(len(pdgId_embedding.categories_[0])):
            scalars_is_one_hot_encoded.append(True)
        scalars_is_one_hot_encoded.append(False)
    else:
        scalars_tensor = psi(torch.tensor(jet_constituent_p4s.mass, dtype=torch.float32)).unsqueeze(-1)
        scalars_tensor = torch.nn.functional.pad(
            scalars_tensor, (0, 1, 0, max_cands - num_jet_constituents), "constant", 0.0
        )
        scalars_is_one_hot_encoded = [False, False]

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

    assert x_tensor.size(dim=1) == len(x_is_one_hot_encoded)
    assert scalars_tensor.size(dim=1) == len(scalars_is_one_hot_encoded)

    return x_tensor, x_is_one_hot_encoded, scalars_tensor, scalars_is_one_hot_encoded, node_mask_tensor


def read_cut(cuts, key):
    if key in cuts.keys():
        return cuts[key]
    else:
        return -1.0

class LorentzNetDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, max_jets=-1, max_cands=50, add_beams=True, use_pdgId=False, preselection={}, cache_size=1, **kwargs):
        """
        Args:
            filelist: List of .pt files or a single directory containing .pt files.
            max_num_files: Maximum number of files to load.
            max_jets: Maximum number of jets (entries) to load.
            max_cands: Maximum number of constituents per jet.
            add_beams: Whether to add beam particles.
            use_pdgId: Whether to use PDG ID features.
            preselection: Dict containing cuts (min_jet_pt, etc.)
            cache_size: Number of shards to keep in memory.
        """
        print("<LorentzNetDataset::LorentzNetDataset>:")
        
        self.min_jet_theta = read_cut(preselection, "min_jet_theta")
        self.max_jet_theta = read_cut(preselection, "max_jet_theta")
        self.min_jet_pt = read_cut(preselection, "min_jet_pt")
        self.max_jet_pt = read_cut(preselection, "max_jet_pt")
        print(" min_jet_theta = %1.3f" % self.min_jet_theta)
        print(" max_jet_theta = %1.3f" % self.max_jet_theta)
        print(" min_jet_pt = %1.3f" % self.min_jet_pt)
        print(" max_jet_pt = %1.3f" % self.max_jet_pt)

        self.max_cands = max_cands
        self.add_beams = add_beams
        self.use_pdgId = use_pdgId

        if isinstance(filelist, str):
            if os.path.isdir(filelist):
                filelist = sorted(glob.glob(os.path.join(filelist, "*.pt")))
            else:
                filelist = [filelist]
        elif isinstance(filelist, (list, tuple)):
            expanded_list = []
            for f in filelist:
                if os.path.isdir(f):
                    expanded_list.extend(sorted(glob.glob(os.path.join(f, "*.pt"))))
                elif "*" in f:
                    expanded_list.extend(sorted(glob.glob(f)))
                else:
                    expanded_list.append(f)
            filelist = expanded_list

        if max_num_files != -1 and len(filelist) > max_num_files:
            filelist = filelist[:max_num_files]
            print(f"Restricting to {len(filelist)} files.")

        self.filelist = []
        self.shard_lengths = []
        self.cumulative_lengths = [0]
        self.valid_indices = [] 
        self.total_jets = 0
        
        self.x_is_one_hot_encoded = None
        self.scalars_is_one_hot_encoded = None

        print("Indexing shards...")
        for i, file in enumerate(filelist):
            if max_jets != -1 and self.total_jets >= max_jets:
                break
                
            print(f"Indexing shard {i}/{len(filelist)}: {file}")
            data = torch.load(file, map_location="cpu", mmap=True)
            
            if "jet_pt" in data and "jet_theta" in data:
                jet_pt = data["jet_pt"]
                jet_theta = data["jet_theta"]
                
                mask = (
                    ((self.min_jet_theta < 0.0) | (jet_theta >= self.min_jet_theta))
                    & ((self.max_jet_theta < 0.0) | (jet_theta <= self.max_jet_theta))
                    & ((self.min_jet_pt < 0.0) | (jet_pt >= self.min_jet_pt))
                    & ((self.max_jet_pt < 0.0) | (jet_pt <= self.max_jet_pt))
                )
                shard_valid_idx = torch.where(mask)[0]
            else:
                print(f" Warning: shard {file} does not contain jet kinematics. Skipping cuts.")
                shard_valid_idx = torch.arange(data["x"].size(0))

            n = shard_valid_idx.size(0)
            
            if max_jets != -1 and (self.total_jets + n) > max_jets:
                n = max_jets - self.total_jets
                shard_valid_idx = shard_valid_idx[:n]
                
            self.filelist.append(file)
            self.shard_lengths.append(n)
            self.valid_indices.append(shard_valid_idx)
            self.total_jets += n
            self.cumulative_lengths.append(self.total_jets)
            
            if self.x_is_one_hot_encoded is None:
                # LorentzNet x is (E, px, py, pz)
                self.x_is_one_hot_encoded = torch.tensor([False, False, False, False], dtype=torch.bool)
                if self.use_pdgId:
                    # 6 one-hots + 1 charge
                    self.scalars_is_one_hot_encoded = torch.tensor([True]*6 + [False], dtype=torch.bool)
                else:
                    # mass features
                    self.scalars_is_one_hot_encoded = torch.tensor([False, False], dtype=torch.bool)
            
            del data
            gc.collect()

        self.cache_size = cache_size
        self.cache = {} 
        self.cache_order = [] 
        
        print(f"Dataset contains {self.total_jets} entries in {len(self.filelist)} shards.")

    def __len__(self):
        return self.total_jets

    def _load_shard(self, shard_idx):
        file_path = self.filelist[shard_idx]
        if file_path in self.cache:
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            return self.cache[file_path]
        
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()
            
        data = torch.load(file_path, map_location="cpu", mmap=True)
        if data["y"].dim() == 1:
            data["y"] = data["y"].unsqueeze(-1)
        if data["weight"].dim() == 1:
            data["weight"] = data["weight"].unsqueeze(-1)
            
        self.cache[file_path] = data
        self.cache_order.append(file_path)
        return data

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_jets:
            raise IndexError(f"Index {idx} out of range")
            
        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        shard_offset = idx - self.cumulative_lengths[shard_idx]
        local_idx = self.valid_indices[shard_idx][shard_offset].item()
        
        data = self._load_shard(shard_idx)
        
        # LorentzNet features reconstruction
        # data["v"] is (P, 4) with (px, py, pz, E) - Wait, in shards it is (4, P)
        v = data["v"][local_idx]
        v = v.permute(1, 0) # (P, 4)
        x_tensor = v[:, [3, 0, 1, 2]] # (P, 4) rearranged to (E, px, py, pz)
        
        node_mask = data["mask"][local_idx] # (1, P)
        # LorentzNet wants node_mask as (P, 1)
        node_mask = node_mask.permute(1, 0)

        scalars_tensor = None
        if self.use_pdgId:
            # data["x"] has: index 7=charge, 8-13=one-hots
            feat_x = data["x"][local_idx] # (C, P)
            # LorentzNet wants (P, C)
            feat_x = feat_x.permute(1, 0)
            one_hots = feat_x[:, 8:14] # (P, 6)
            charge = feat_x[:, 7:8] # (P, 1)
            scalars_tensor = torch.cat([one_hots, charge], dim=1)
        else:
            # Recompute mass-based scalars if needed
            # In old code it was psi(mass)
            from LGEB import psi
            # mass^2 = E^2 - px^2 - py^2 - pz^2
            mass_sq = x_tensor[:, 0]**2 - x_tensor[:, 1]**2 - x_tensor[:, 2]**2 - x_tensor[:, 3]**2
            mass = torch.sqrt(torch.clamp(mass_sq, min=0.0))
            scalars_tensor = psi(mass).unsqueeze(-1)
            # Pad to 2 scalars as in old code
            scalars_tensor = torch.nn.functional.pad(scalars_tensor, (0, 1), "constant", 0.0)

        if self.add_beams:
            beam_mass = 1.0
            beam1_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, +1.0]
            beam2_p4 = [math.sqrt(1 + beam_mass**2), 0.0, 0.0, -1.0]
            x_beams = torch.tensor([beam1_p4, beam2_p4], dtype=torch.float32)
            x_tensor = torch.cat([x_beams, x_tensor], dim=0)

            if self.use_pdgId:
                # Beams are protons (2212)
                # Protons one-hot in our scheme is index 13 (6th one-hot)
                # indices: 0:11, 1:13, 2:22, 3:211, 4:130, 5:2212
                proton_oh = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                scalars_beams = torch.tensor([
                    proton_oh + [+1.0], # beam 1
                    proton_oh + [-1.0]  # beam 2
                ], dtype=torch.float32)
                scalars_tensor = torch.cat([scalars_beams, scalars_tensor], dim=0)
            else:
                from LGEB import psi
                scalars_beams = psi(torch.tensor([beam_mass, beam_mass], dtype=torch.float32)).unsqueeze(-1)
                scalars_beams = torch.nn.functional.pad(scalars_beams, (1, 0), "constant", 0.0)
                scalars_tensor = torch.cat([scalars_beams, scalars_tensor], dim=0)

            node_mask_beams = torch.ones(2, 1, dtype=torch.float32)
            node_mask = torch.cat([node_mask_beams, node_mask], dim=0)

        return (
            {
                "x": x_tensor,
                "x_is_one_hot_encoded": self.x_is_one_hot_encoded,
                "scalars": scalars_tensor,
                "scalars_is_one_hot_encoded": self.scalars_is_one_hot_encoded,
                "mask": node_mask,
            },
            data["y"][local_idx].clone(),
            data["weight"][local_idx].clone(),
        )
