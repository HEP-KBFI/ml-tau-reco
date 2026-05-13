#!/usr/bin/env python3
import argparse
import awkward as ak
import vector
import torch
import math
import multiprocessing
from tqdm import tqdm
import os
import sys
import numpy as np
import pyarrow.parquet as pq
import psutil

# Ensure src is in the path so we can import ParticleTransformerDataset and hpsAlgoTools
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from hpsAlgoTools import (
    comp_angle3d,
    comp_deltaEta,
    comp_deltaPhi,
    comp_deltaTheta,
    comp_deltaR_etaphi,
    comp_deltaR_thetaphi,
)

METRICS = {
    "eta-phi": (comp_deltaR_etaphi, comp_deltaEta),
    "theta-phi": (comp_deltaR_thetaphi, comp_deltaTheta),
    "angle3d": (comp_angle3d, comp_deltaTheta),
}


def process_jet_chunk_vectorized(
    data_chunk,
    metric_name,
    max_cands,
    use_pdgId,
    use_lifetime,
):
    # This function processes a chunk of jets at once using vectorized operations.

    # 1. Prepare vectors
    jet_p4s = vector.awk(
        ak.zip(
            {
                "px": data_chunk.reco_jet_p4s.x,
                "py": data_chunk.reco_jet_p4s.y,
                "pz": data_chunk.reco_jet_p4s.z,
                "mass": data_chunk.reco_jet_p4s.tau,
            }
        )
    )
    cand_p4s = vector.awk(
        ak.zip(
            {
                "px": data_chunk.reco_cand_p4s.x,
                "py": data_chunk.reco_cand_p4s.y,
                "pz": data_chunk.reco_cand_p4s.z,
                "mass": data_chunk.reco_cand_p4s.tau,
            }
        )
    )

    # Truncate candidates
    cand_p4s = cand_p4s[:, :max_cands]

    # 2. Compute basic features
    if metric_name == "eta-phi":
        deta = np.abs(cand_p4s.eta - jet_p4s.eta)
        dphi = np.abs((cand_p4s.phi - jet_p4s.phi + np.pi) % (2 * np.pi) - np.pi)
    elif metric_name == "theta-phi":
        deta = np.abs(cand_p4s.theta - jet_p4s.theta)
        dphi = np.abs((cand_p4s.phi - jet_p4s.phi + np.pi) % (2 * np.pi) - np.pi)
    elif metric_name == "angle3d":
        deta = np.abs(cand_p4s.theta - jet_p4s.theta)
        # unit vectors dot product
        p_jet = np.sqrt(jet_p4s.x**2 + jet_p4s.y**2 + jet_p4s.z**2)
        p_cand = np.sqrt(cand_p4s.x**2 + cand_p4s.y**2 + cand_p4s.z**2)
        dot = (jet_p4s.x * cand_p4s.x + jet_p4s.y * cand_p4s.y + jet_p4s.z * cand_p4s.z) / (p_jet * p_cand)
        dot = ak.where(dot > 1.0, 1.0, dot)
        dot = ak.where(dot < -1.0, -1.0, dot)
        dphi = np.arccos(dot)

    logpt = np.log(cand_p4s.pt)
    loge = np.log(cand_p4s.energy)
    logptrel = np.log(cand_p4s.pt / jet_p4s.pt)
    logerel = np.log(cand_p4s.energy / jet_p4s.energy)

    if metric_name != "angle3d":
        dR = np.sqrt(deta**2 + dphi**2)
    else:
        dR = dphi

    features = [deta, dphi, logpt, loge, logptrel, logerel, dR]

    # 3. PDG ID features
    if use_pdgId:
        pdgId = np.abs(data_chunk.reco_cand_pdg[:, :max_cands])
        charge = data_chunk.reco_cand_charge[:, :max_cands]
        features.append(charge)
        features.append(ak.values_astype(pdgId == 11, np.float32))
        features.append(ak.values_astype(pdgId == 13, np.float32))
        features.append(ak.values_astype(pdgId == 22, np.float32))
        features.append(ak.values_astype(pdgId == 211, np.float32))
        features.append(ak.values_astype(pdgId == 130, np.float32))

    # 4. Lifetime features
    if use_lifetime:
        d0s = data_chunk.reco_cand_dxy[:, :max_cands]
        d0errs = data_chunk.reco_cand_dxy_err[:, :max_cands]
        dzs = np.abs(data_chunk.reco_cand_dz[:, :max_cands])
        dzerrs = data_chunk.reco_cand_dz_err[:, :max_cands]

        charge = data_chunk.reco_cand_charge[:, :max_cands]
        mask_lt = np.abs(charge) > 0.5

        part_d0 = ak.where(mask_lt, np.tanh(d0s), 0.0)
        denom_d0 = np.maximum(0.01 * d0s, d0errs)
        denom_d0 = ak.where(denom_d0 == 0, 1.0, denom_d0)
        part_d0err = ak.where(mask_lt, d0s / denom_d0, 0.0)

        part_dz = ak.where(mask_lt, np.tanh(dzs), 0.0)
        denom_dz = np.maximum(0.01 * dzs, dzerrs)
        denom_dz = ak.where(denom_dz == 0, 1.0, denom_dz)
        part_dzerr = ak.where(mask_lt, dzs / denom_dz, 0.0)

        features.extend([part_d0, part_d0err, part_dz, part_dzerr])

    # 5. Padding and stacking
    def pad_and_to_numpy(a):
        return ak.to_numpy(ak.fill_none(ak.pad_none(a, max_cands, axis=1, clip=True), 0.0))

    x_tensor = np.stack([pad_and_to_numpy(f) for f in features], axis=1)

    v_tensor = np.stack(
        [
            pad_and_to_numpy(cand_p4s.x),
            pad_and_to_numpy(cand_p4s.y),
            pad_and_to_numpy(cand_p4s.z),
            pad_and_to_numpy(cand_p4s.energy),
        ],
        axis=1,
    )

    node_mask = ak.values_astype(ak.pad_none(ak.ones_like(cand_p4s.x), max_cands, axis=1, clip=True), np.float32)
    node_mask_tensor = pad_and_to_numpy(node_mask)[:, np.newaxis, :]

    # 6. Labels and weights
    y = np.where(ak.to_numpy(data_chunk.gen_jet_tau_decaymode) != -1, 1, 0)
    weights = (
        ak.to_numpy(data_chunk.weight) if "weight" in data_chunk.fields else np.ones(len(data_chunk), dtype=np.float32)
    )

    # Return as tensors
    return (
        torch.tensor(x_tensor, dtype=torch.float32),
        torch.tensor(v_tensor, dtype=torch.float32),
        torch.tensor(node_mask_tensor, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long).unsqueeze(-1),
        torch.tensor(weights, dtype=torch.float32).unsqueeze(-1),
    )


def worker_fn(args):
    (
        file_path,
        rg_range,
        file_columns,
        metric,
        max_cands,
        use_pdgId,
        use_lifetime,
        min_jet_theta,
        max_jet_theta,
        min_jet_pt,
        max_jet_pt,
    ) = args
    try:
        data_rg = ak.from_parquet(file_path, row_groups=rg_range, columns=file_columns)
        if len(data_rg) == 0:
            return None

        # Apply preselection
        jet_p4s = vector.awk(
            ak.zip(
                {
                    "px": data_rg.reco_jet_p4s.x,
                    "py": data_rg.reco_jet_p4s.y,
                    "pz": data_rg.reco_jet_p4s.z,
                    "mass": data_rg.reco_jet_p4s.tau,
                }
            )
        )

        mask = (
            ((min_jet_theta < 0.0) | (jet_p4s.theta >= min_jet_theta))
            & ((max_jet_theta < 0.0) | (jet_p4s.theta <= max_jet_theta))
            & ((min_jet_pt < 0.0) | (jet_p4s.pt >= min_jet_pt))
            & ((max_jet_pt < 0.0) | (jet_p4s.pt <= max_jet_pt))
        )

        masked_data_rg = data_rg[mask]
        if len(masked_data_rg) == 0:
            return None

        return process_jet_chunk_vectorized(masked_data_rg, metric, max_cands, use_pdgId, use_lifetime)
    except Exception as e:
        print(f"Error processing row groups {rg_range} in {file_path}: {e}")
        return None


def get_memory_info():
    mem = psutil.virtual_memory()
    return mem.available


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet datasets to PyTorch tensors for Particle Transformer training."
    )
    parser.add_argument("--input_files", nargs="+", required=True, help="List of input parquet files.")
    parser.add_argument("--output_file", required=True, help="Path to save the converted dataset (.pt).")
    parser.add_argument(
        "--max_num_files", type=int, default=-1, help="Maximum number of files to process per category (sig/bgr)."
    )
    parser.add_argument("--max_cands", type=int, default=50, help="Maximum number of constituents per jet.")
    parser.add_argument("--metric", default="eta-phi", choices=METRICS.keys(), help="Metric to use.")
    parser.add_argument("--no_pdgId", action="store_false", dest="use_pdgId", help="Do not use PDG IDs as features.")
    parser.add_argument(
        "--no_lifetime", action="store_false", dest="use_lifetime", help="Do not use lifetime features."
    )
    parser.set_defaults(use_pdgId=True, use_lifetime=True)
    parser.add_argument("--min_jet_theta", type=float, default=-1.0)
    parser.add_argument("--max_jet_theta", type=float, default=-1.0)
    parser.add_argument("--min_jet_pt", type=float, default=-1.0)
    parser.add_argument("--max_jet_pt", type=float, default=-1.0)
    parser.add_argument(
        "--num_workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers."
    )
    parser.add_argument("--chunk_size", type=int, default=50000, help="Number of jets per parallel chunk.")
    parser.add_argument(
        "--max_jets_per_file", type=int, default=-1, help="Maximum number of jets to process per file (for testing)."
    )
    parser.add_argument(
        "--max_jets_per_output", type=int, default=1000000, help="Maximum number of jets per output .pt file."
    )

    args = parser.parse_args()

    filelist = args.input_files
    if args.max_num_files != -1:
        num_sig_files = 0
        num_bgr_files = 0
        restricted_filelist = []
        for file in filelist:
            if "zh" in file.lower():
                if num_sig_files < args.max_num_files:
                    restricted_filelist.append(file)
                    num_sig_files += 1
            elif "qq" in file.lower() or "z" in file.lower():
                if num_bgr_files < args.max_num_files:
                    restricted_filelist.append(file)
                    num_bgr_files += 1
            else:
                restricted_filelist.append(file)
        filelist = restricted_filelist
        print(f"Restricting to {len(filelist)} files.")

    all_x = []
    all_v = []
    all_mask = []
    all_y = []
    all_weight = []
    current_jets_in_memory = 0
    output_count = 0

    # Identify necessary columns to save memory
    columns = ["reco_jet_p4s", "reco_cand_p4s", "gen_jet_tau_decaymode"]
    if args.use_pdgId:
        columns.extend(["reco_cand_pdg", "reco_cand_charge"])
    if args.use_lifetime:
        if "reco_cand_charge" not in columns:
            columns.append("reco_cand_charge")
        columns.extend(["reco_cand_dxy", "reco_cand_dxy_err", "reco_cand_dz", "reco_cand_dz_err"])

    x_is_oh = [False] * 7
    if args.use_pdgId:
        x_is_oh.extend([False, True, True, True, True, True])
    if args.use_lifetime:
        x_is_oh.extend([False, False, False, False])
    v_is_oh = [False, False, False, False]

    def save_current_tensors():
        nonlocal output_count, current_jets_in_memory
        if not all_x:
            return

        print(f"\nStacking tensors for {current_jets_in_memory} jets...")
        data_to_save = {
            "x": torch.cat(all_x, dim=0),
            "v": torch.cat(all_v, dim=0),
            "mask": torch.cat(all_mask, dim=0),
            "y": torch.cat(all_y, dim=0),
            "weight": torch.cat(all_weight, dim=0),
            "x_is_one_hot_encoded": torch.tensor(x_is_oh, dtype=torch.bool),
            "v_is_one_hot_encoded": torch.tensor(v_is_oh, dtype=torch.bool),
        }

        base, ext = os.path.splitext(args.output_file)
        out_file = f"{base}_{output_count}{ext}"
        output_dir = os.path.dirname(out_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"Saving to {out_file}...")
        torch.save(data_to_save, out_file)

        output_count += 1
        all_x.clear()
        all_v.clear()
        all_mask.clear()
        all_y.clear()
        all_weight.clear()
        current_jets_in_memory = 0

    total_jets_to_process = 0
    for file in filelist:
        meta = pq.read_metadata(file)
        if args.max_jets_per_file != -1:
            total_jets_to_process += min(meta.num_rows, args.max_jets_per_file)
        else:
            total_jets_to_process += meta.num_rows

    est_mem_gb = (total_jets_to_process * 4600) / (1024**3)
    avail_mem_gb = get_memory_info() / (1024**3)
    print(f"Total jets to process: {total_jets_to_process}")
    print(f"Estimated total memory for all tensors: {est_mem_gb:.2f} GB")
    print(f"Available memory: {avail_mem_gb:.2f} GB")

    if args.num_workers > 1:
        pool = multiprocessing.Pool(processes=args.num_workers)
    else:
        pool = None

    for file in filelist:
        print(f"Opening file {file}")
        parquet_file = pq.ParquetFile(file)
        num_row_groups = parquet_file.metadata.num_row_groups

        file_columns = columns.copy()
        if "weight" in parquet_file.schema.names:
            file_columns.append("weight")

        num_jets_in_file = parquet_file.metadata.num_rows
        num_jets_to_process = num_jets_in_file
        if args.max_jets_per_file != -1:
            num_jets_to_process = min(num_jets_in_file, args.max_jets_per_file)

        print(f"File contains {num_jets_in_file} jets, processing up to {num_jets_to_process}")

        jets_processed_in_file = 0

        # Estimate rows per row group to set rg_per_task
        avg_rows_per_rg = num_jets_in_file / num_row_groups if num_row_groups > 0 else 1
        rg_per_task = max(1, int(args.chunk_size / avg_rows_per_rg))

        tasks = (
            (
                file,
                list(range(i, min(i + rg_per_task, num_row_groups))),
                file_columns,
                args.metric,
                args.max_cands,
                args.use_pdgId,
                args.use_lifetime,
                args.min_jet_theta,
                args.max_jet_theta,
                args.min_jet_pt,
                args.max_jet_pt,
            )
            for i in range(0, num_row_groups, rg_per_task)
        )

        if pool is not None:
            results_iter = pool.imap(worker_fn, tasks)
        else:
            results_iter = map(worker_fn, tasks)

        pbar = tqdm(total=num_jets_to_process, desc=f"Processing {os.path.basename(file)}")
        for res in results_iter:
            if res is None:
                continue
            x_t, v_t, m_t, y_t, w_t = res

            # Truncate if we exceeded max_jets_per_file
            if jets_processed_in_file + len(x_t) > num_jets_to_process:
                keep = num_jets_to_process - jets_processed_in_file
                if keep <= 0:
                    break
                x_t, v_t, m_t, y_t, w_t = x_t[:keep], v_t[:keep], m_t[:keep], y_t[:keep], w_t[:keep]

            all_x.append(x_t)
            all_v.append(v_t)
            all_mask.append(m_t)
            all_y.append(y_t)
            all_weight.append(w_t)
            current_jets_in_memory += len(x_t)
            jets_processed_in_file += len(x_t)
            pbar.update(len(x_t))

            if current_jets_in_memory >= args.max_jets_per_output:
                save_current_tensors()

            if jets_processed_in_file >= num_jets_to_process:
                break
        pbar.close()

    if pool is not None:
        pool.close()
        pool.join()

    # Save any remaining tensors
    if all_x:
        save_current_tensors()

    if output_count == 0:
        print("No jets selected! Exiting.")
        return

    print(f"Successfully saved {output_count} files.")
    print("Done.")


if __name__ == "__main__":
    main()
