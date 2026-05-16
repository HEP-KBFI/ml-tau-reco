import torch
from torch.utils.data import Dataset
import glob
import os
import bisect
import gc
import math

from hpsAlgoTools import (
    comp_angle3d,
    comp_deltaEta,
    comp_deltaPhi,
    comp_deltaTheta,
    comp_deltaR_etaphi,
    comp_deltaR_thetaphi,
)


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
    use_pdgId,
    use_lifetime,
):
    # print("<buildParticleTransformerTensors>:")

    jet_constituent_p4s = jet_constituent_p4s[:max_cands]
    jet_constituent_p4s_zipped = list(
        zip(jet_constituent_p4s.px, jet_constituent_p4s.py, jet_constituent_p4s.pz, jet_constituent_p4s.energy)
    )
    num_jet_constituents = int(len(jet_constituent_p4s_zipped))

    jet_constituent_features = []
    is_one_hot_encoded = []
    for idx in range(num_jet_constituents):
        jet_constituent_p4 = jet_constituent_p4s[idx]
        jet_constituent_pdgId = jet_constituent_pdgIds[idx]
        jet_constituent_abs_pdgId = abs(jet_constituent_pdgId)
        jet_constituent_q = jet_constituent_qs[idx]
        jet_constituent_d0 = jet_constituent_d0s[idx]
        jet_constituent_d0err = jet_constituent_d0errs[idx]
        jet_constituent_dz = math.fabs(jet_constituent_dzs[idx])
        jet_constituent_dzerr = jet_constituent_dzerrs[idx]

        part_deta = metric_dEta_or_dTheta(jet_constituent_p4, jet_p4)
        part_dphi = comp_deltaPhi(jet_constituent_p4, jet_p4)
        part_logpt = math.log(jet_constituent_p4.pt)
        part_loge = math.log(jet_constituent_p4.energy)
        part_logptrel = math.log(jet_constituent_p4.pt / jet_p4.pt)
        part_logerel = math.log(jet_constituent_p4.energy / jet_p4.energy)
        part_deltaR = metric_dR_or_angle(jet_constituent_p4, jet_p4)
        part_charge = jet_constituent_q
        part_isElectron = 1.0 if jet_constituent_abs_pdgId == 11 else 0.0
        part_isMuon = 1.0 if jet_constituent_abs_pdgId == 13 else 0.0
        part_isPhoton = 1.0 if jet_constituent_abs_pdgId == 22 else 0.0
        part_isChargedHadron = 1.0 if jet_constituent_abs_pdgId == 211 else 0.0
        part_isNeutralHadron = 1.0 if jet_constituent_abs_pdgId == 130 else 0.0
        part_isProton = 1.0 if jet_constituent_abs_pdgId == 2212 else 0.0
        part_d0 = 0.0
        part_d0err = 0.0
        part_dz = 0.0
        part_dzerr = 0.0
        if abs(part_charge) > 0.5 and part_d0 > -99.0 and part_dz > -99.0:
            part_d0 = math.tanh(jet_constituent_d0)
            part_d0err = jet_constituent_d0 / max(0.01 * jet_constituent_d0, jet_constituent_d0err)
            part_dz = math.tanh(jet_constituent_dz)
            part_dzerr = jet_constituent_dz / max(0.01 * jet_constituent_dz, jet_constituent_dzerr)

        part_features = [
            part_deta,
            part_dphi,
            part_logpt,
            part_loge,
            part_logptrel,
            part_logerel,
            part_deltaR,
        ]
        if idx == 0:
            is_one_hot_encoded = [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ]
        if use_pdgId:
            part_features.extend(
                [
                    part_charge,
                    part_isElectron,
                    part_isMuon,
                    part_isPhoton,
                    part_isChargedHadron,
                    part_isNeutralHadron,
                    part_isProton,
                ]
            )
            if idx == 0:
                is_one_hot_encoded.extend(
                    [
                        False,
                        True,
                        True,
                        True,
                        True,
                        True,
                        True,
                    ]
                )
        if use_lifetime:
            part_features.extend(
                [
                    part_d0,
                    part_d0err,
                    part_dz,
                    part_dzerr,
                ]
            )
            if idx == 0:
                is_one_hot_encoded.extend(
                    [
                        False,
                        False,
                        False,
                        False,
                    ]
                )
        assert len(part_features) == len(is_one_hot_encoded)
        jet_constituent_features.append(part_features)

    x_tensor = torch.tensor(jet_constituent_features, dtype=torch.float32)
    x_tensor = torch.nn.functional.pad(x_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0)

    v_tensor = torch.tensor(jet_constituent_p4s_zipped, dtype=torch.float32)
    v_tensor = torch.nn.functional.pad(v_tensor, (0, 0, 0, max_cands - num_jet_constituents), "constant", 0.0)

    node_mask_tensor = torch.ones(num_jet_constituents, dtype=torch.float32)
    node_mask_tensor = torch.nn.functional.pad(node_mask_tensor, (0, max_cands - num_jet_constituents), "constant", 0.0)
    node_mask_tensor = torch.unsqueeze(node_mask_tensor, dim=-1)

    x_tensor = torch.swapaxes(x_tensor, 0, 1)
    x_is_one_hot_encoded = is_one_hot_encoded
    v_tensor = torch.swapaxes(v_tensor, 0, 1)
    v_is_one_hot_encoded = [False, False, False, False]
    node_mask_tensor = torch.swapaxes(node_mask_tensor, 0, 1)

    return x_tensor, x_is_one_hot_encoded, v_tensor, v_is_one_hot_encoded, node_mask_tensor


def read_cut(cuts, key):
    if key in cuts.keys():
        return cuts[key]
    else:
        return -1.0


class ParticleTransformerDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, max_jets=-1, cache_size=1, **kwargs):
        """
        Args:
            filelist: List of .pt files or a single directory containing .pt files.
            max_num_files: Maximum number of files to load.
            max_jets: Maximum number of jets (entries) to load.
            cache_size: Number of shards to keep in memory.
        """
        print("<ParticleTransformerDataset::ParticleTransformerDataset>:")
        
        preselection = kwargs.get("preselection", {})
        self.min_jet_theta = read_cut(preselection, "min_jet_theta")
        self.max_jet_theta = read_cut(preselection, "max_jet_theta")
        self.min_jet_pt = read_cut(preselection, "min_jet_pt")
        self.max_jet_pt = read_cut(preselection, "max_jet_pt")
        print(" min_jet_theta = %1.3f" % self.min_jet_theta)
        print(" max_jet_theta = %1.3f" % self.max_jet_theta)
        print(" min_jet_pt = %1.3f" % self.min_jet_pt)
        print(" max_jet_pt = %1.3f" % self.max_jet_pt)

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
        self.valid_indices = [] # List of (shard_idx, local_idx) or per-shard valid indices
        self.total_jets = 0
        
        self.x_is_one_hot_encoded = None
        self.v_is_one_hot_encoded = None

        print("Indexing shards...")
        for i, file in enumerate(filelist):
            if max_jets != -1 and self.total_jets >= max_jets:
                break
                
            print(f"Indexing shard {i}/{len(filelist)}: {file}")
            # Standard load to get metadata and kinematics
            data = torch.load(file, map_location="cpu", mmap=True)
            
            # Apply cuts if they exist in the file
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
                self.x_is_one_hot_encoded = data.get("x_is_one_hot_encoded", None)
                self.v_is_one_hot_encoded = data.get("v_is_one_hot_encoded", None)
            
            del data
            gc.collect()

        self.cache_size = cache_size
        self.cache = {} # path -> data
        self.cache_order = [] # LRU
        
        print(f"Dataset contains {self.total_jets} entries in {len(self.filelist)} shards.")

    def __len__(self):
        return self.total_jets

    def _load_shard(self, shard_idx):
        file_path = self.filelist[shard_idx]
        if file_path in self.cache:
            # Move to end of LRU
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            return self.cache[file_path]
        
        # Load new shard
        if len(self.cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()
            
        data = torch.load(file_path, map_location="cpu", mmap=True)
        # Ensure correct shapes
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
            
        # Find which shard idx belongs to
        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        shard_offset = idx - self.cumulative_lengths[shard_idx]
        
        # Map to original index in the shard
        local_idx = self.valid_indices[shard_idx][shard_offset].item()
        
        data = self._load_shard(shard_idx)
        
        return (
            {
                "v": data["v"][local_idx].clone(),
                "v_is_one_hot_encoded": self.v_is_one_hot_encoded,
                "x": data["x"][local_idx].clone(),
                "x_is_one_hot_encoded": self.x_is_one_hot_encoded,
                "mask": data["mask"][local_idx].clone(),
            },
            data["y"][local_idx].clone(),
            data["weight"][local_idx].clone(),
        )
