#!/usr/bin/env python3
import torch
import os
import glob
import random
import argparse
from tqdm import tqdm
import shutil

def shuffle_dataset(input_files, output_dir, num_buckets=32, tmp_dir="tmp_shuffle"):
    print(f"Starting global shuffle of {len(input_files)} files into {num_buckets} buckets.")
    
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create bucket directories
    bucket_dirs = [os.path.join(tmp_dir, f"bucket_{i}") for i in range(num_buckets)]
    for d in bucket_dirs:
        os.makedirs(d, exist_ok=True)

    # Pass 1: Distribute jets into buckets
    print("Pass 1: Distributing jets into buckets...")
    for file_idx, file_path in enumerate(tqdm(input_files, desc="Distributing")):
        data = torch.load(file_path, map_location="cpu")
        num_jets = data["x"].size(0)
        
        # Generate random bucket assignments for all jets in this file
        bucket_assignments = torch.randint(0, num_buckets, (num_jets,))
        
        # Split and save chunks
        for b_idx in range(num_buckets):
            mask = (bucket_assignments == b_idx)
            if mask.any():
                chunk = {
                    "x": data["x"][mask],
                    "v": data["v"][mask],
                    "mask": data["mask"][mask],
                    "y": data["y"][mask],
                    "weight": data["weight"][mask],
                    "x_is_one_hot_encoded": data["x_is_one_hot_encoded"],
                    "v_is_one_hot_encoded": data["v_is_one_hot_encoded"],
                }
                torch.save(chunk, os.path.join(bucket_dirs[b_idx], f"chunk_{file_idx}.pt"))
        
        # Clear data from memory
        del data

    # Pass 2: Shuffle buckets and save final shards
    print("\nPass 2: Shuffling buckets and saving final shards...")
    for b_idx in range(num_buckets):
        bucket_path = bucket_dirs[b_idx]
        chunk_files = glob.glob(os.path.join(bucket_path, "chunk_*.pt"))
        if not chunk_files:
            continue
            
        all_x, all_v, all_mask, all_y, all_weight = [], [], [], [], []
        x_is_oh, v_is_oh = None, None
        
        for cf in chunk_files:
            c_data = torch.load(cf, map_location="cpu")
            all_x.append(c_data["x"])
            all_v.append(c_data["v"])
            all_mask.append(c_data["mask"])
            all_y.append(c_data["y"])
            all_weight.append(c_data["weight"])
            if x_is_oh is None:
                x_is_oh = c_data["x_is_one_hot_encoded"]
                v_is_oh = c_data["v_is_one_hot_encoded"]
        
        # Concatenate
        full_x = torch.cat(all_x, dim=0)
        full_v = torch.cat(all_v, dim=0)
        full_mask = torch.cat(all_mask, dim=0)
        full_y = torch.cat(all_y, dim=0)
        full_weight = torch.cat(all_weight, dim=0)
        
        # Shuffle internally
        perm = torch.randperm(full_x.size(0))
        
        final_shard = {
            "x": full_x[perm],
            "v": full_v[perm],
            "mask": full_mask[perm],
            "y": full_y[perm],
            "weight": full_weight[perm],
            "x_is_one_hot_encoded": x_is_oh,
            "v_is_one_hot_encoded": v_is_oh,
        }
        
        output_path = os.path.join(output_dir, f"shuffled_dataset_{b_idx}.pt")
        print(f"Saving bucket {b_idx} ({full_x.size(0)} jets) to {output_path}")
        torch.save(final_shard, output_path)
        
        # Clean up bucket chunks immediately to save space
        shutil.rmtree(bucket_path)

    # Clean up tmp_dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    print("\nGlobal shuffle complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Globally shuffle sharded .pt datasets.")
    parser.add_argument("--input_pattern", required=True, help="Glob pattern for input .pt files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save shuffled shards.")
    parser.add_argument("--num_buckets", type=int, default=32, help="Number of buckets for shuffling.")
    parser.add_argument("--tmp_dir", default="tmp_shuffle", help="Temporary directory for bucketing.")
    
    args = parser.parse_args()
    
    input_files = sorted(glob.glob(args.input_pattern))
    if not input_files:
        print(f"No files found matching {args.input_pattern}")
    else:
        shuffle_dataset(input_files, args.output_dir, args.num_buckets, args.tmp_dir)
