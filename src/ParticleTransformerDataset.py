import torch
from torch.utils.data import Dataset
import glob
import os
import bisect
import gc

class ParticleTransformerDataset(Dataset):
    def __init__(self, filelist, max_num_files=-1, cache_size=1, **kwargs):
        """
        Args:
            filelist: List of .pt files or a single directory containing .pt files.
            max_num_files: Maximum number of files to load.
            cache_size: Number of shards to keep in memory.
        """
        print("<ParticleTransformerDataset::ParticleTransformerDataset>:")
        
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

        self.filelist = filelist
        print(f"Found {len(self.filelist)} .pt files.")

        self.shard_lengths = []
        self.cumulative_lengths = [0]
        self.total_jets = 0
        
        self.x_is_one_hot_encoded = None
        self.v_is_one_hot_encoded = None

        print("Indexing shards...")
        for file in self.filelist:
            # Standard load to get metadata
            data = torch.load(file, map_location="cpu")
            n = data["x"].size(0)
            self.shard_lengths.append(n)
            self.total_jets += n
            self.cumulative_lengths.append(self.total_jets)
            
            if self.x_is_one_hot_encoded is None:
                self.x_is_one_hot_encoded = data["x_is_one_hot_encoded"]
                self.v_is_one_hot_encoded = data["v_is_one_hot_encoded"]
            
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
            
        data = torch.load(file_path, map_location="cpu")
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
        local_idx = idx - self.cumulative_lengths[shard_idx]
        
        data = self._load_shard(shard_idx)
        
        return (
            {
                "v": data["v"][local_idx],
                "v_is_one_hot_encoded": self.v_is_one_hot_encoded,
                "x": data["x"][local_idx],
                "x_is_one_hot_encoded": self.x_is_one_hot_encoded,
                "mask": data["mask"][local_idx],
            },
            data["y"][local_idx],
            data["weight"][local_idx],
        )
