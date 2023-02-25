import os
import glob
import hydra
import vector
import numpy as np
import awkward as ak
import seaborn as sns
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from general import load_all_data


def load_samples(sig_dir: str, bkg_dir: str):
    sig_data = load_all_data(sig_dir)
    bkg_data = load_all_data(bkg_dir)
    return sig_data, bkg_data


def visualize_weights(weight_matrix, pt_bin_edges, eta_bin_edges, output_path):
    pt_labels = [f"{label:9.0f}" for label in (pt_bin_edges[1:] + pt_bin_edges[:-1]) / 2]
    eta_labels = [f"{label:9.2f}" for label in (eta_bin_edges[1:] + eta_bin_edges[:-1]) / 2]
    sns.set(rc={"figure.figsize": (16, 9)})
    heatmap = sns.heatmap(weight_matrix)
    heatmap.set_xticks(range(len(pt_labels)))
    heatmap.set_xticklabels(pt_labels)
    heatmap.set_yticks(range(len(eta_labels)))
    heatmap.set_yticklabels(eta_labels)
    plt.ylabel(r"$\eta$")
    plt.xlabel(r"$p_T$")
    for i, label in enumerate(heatmap.xaxis.get_ticklabels()):
        if i % 5 != 0:
            label.set_visible(False)
    for i, label in enumerate(heatmap.yaxis.get_ticklabels()):
        if i % 5 != 0:
            label.set_visible(False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def create_matrix(data, eta_bin_edges, pt_bin_edges):
    p4s = vector.awk(
        ak.zip(
            {
                "mass": data.gen_jet_p4s.tau,
                "x": data.gen_jet_p4s.x,
                "y": data.gen_jet_p4s.y,
                "z": data.gen_jet_p4s.z,
            }
        )
    )

    pt = p4s.pt.to_numpy()
    eta = p4s.eta.to_numpy()
    matrix = np.histogram2d(eta, pt, bins=(eta_bin_edges, pt_bin_edges))[0]
    return matrix


def get_weight_matrix(target_matrix, comp_matrix):
    weights = np.minimum(target_matrix, comp_matrix) / target_matrix
    return np.nan_to_num(weights, nan=0.0)


def process_files(weight_matrix, eta_bin_edges, pt_bin_edges, data_dir, use_multiprocessing=True):
    data_paths = glob.glob(os.path.join(data_dir, "*.parquet"))
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(
            process_single_file, zip(data_paths, repeat(weight_matrix), repeat(eta_bin_edges), repeat(pt_bin_edges))
        )
    else:
        for input_path in data_paths:
            process_single_file(
                input_path=input_path, weight_matrix=weight_matrix, eta_bin_edges=eta_bin_edges, pt_bin_edges=pt_bin_edges
            )


def process_single_file(input_path, weight_matrix, eta_bin_edges, pt_bin_edges):
    data = ak.from_parquet(input_path)
    p4s = vector.awk(
        ak.zip(
            {
                "mass": data.gen_jet_p4s.tau,
                "x": data.gen_jet_p4s.x,
                "y": data.gen_jet_p4s.y,
                "z": data.gen_jet_p4s.z,
            }
        )
    )
    eta_values = p4s.eta.to_numpy()
    pt_values = p4s.pt.to_numpy()
    eta_bin = np.digitize(eta_values, bins=eta_bin_edges) - 1
    pt_bin = np.digitize(pt_values, bins=pt_bin_edges) - 1
    matrix_loc = np.concatenate([eta_bin.reshape(-1, 1), pt_bin.reshape(-1, 1)], axis=1)
    weights = ak.from_iter([weight_matrix[tuple(loc)] for loc in matrix_loc])
    merged_info = {field: data[field] for field in data.fields}
    merged_info.update({"weight": weights})
    print(f"Adding weights to {input_path}")
    ak.to_parquet(ak.Record(merged_info), input_path)


@hydra.main(config_path="../config", config_name="weighting", version_base=None)
def main(cfg: DictConfig):
    sig_data, bkg_data = load_samples(sig_dir=cfg.samples.ZH_Htautau.output_dir, bkg_dir=cfg.samples.QCD.output_dir)
    eta_bin_edges = np.linspace(
        cfg.weighting.variables.eta.range[0], cfg.weighting.variables.eta.range[1], cfg.weighting.variables.eta.n_bins
    )
    pt_bin_edges = np.linspace(
        cfg.weighting.variables.pt.range[0], cfg.weighting.variables.pt.range[1], cfg.weighting.variables.pt.n_bins
    )
    sig_matrix = create_matrix(sig_data, eta_bin_edges, pt_bin_edges)
    bkg_matrix = create_matrix(bkg_data, eta_bin_edges, pt_bin_edges)
    sig_weights = get_weight_matrix(target_matrix=sig_matrix, comp_matrix=bkg_matrix)
    bkg_weights = get_weight_matrix(target_matrix=bkg_matrix, comp_matrix=sig_weights)
    sig_output_path = os.path.join(cfg.samples.ZH_Htautau.output_dir, "signal_weights.png")
    visualize_weights(sig_weights, pt_bin_edges, eta_bin_edges, sig_output_path)
    bkg_output_path = os.path.join(cfg.samples.QCD.output_dir, "bkg_weights.png")
    visualize_weights(bkg_weights, pt_bin_edges, eta_bin_edges, bkg_output_path)
    process_files(
        weight_matrix=sig_weights,
        eta_bin_edges=eta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.ZH_Htautau.output_dir,
        use_multiprocessing=cfg.use_multiprocessing,
    )
    process_files(
        weight_matrix=bkg_weights,
        eta_bin_edges=eta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.QCD.output_dir,
        use_multiprocessing=cfg.use_multiprocessing,
    )


if __name__ == "__main__":
    main()
