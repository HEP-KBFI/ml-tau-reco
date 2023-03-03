import os
import glob
import hydra
import vector
import numpy as np
import mplhep as hep
import awkward as ak
import seaborn as sns
import multiprocessing
from itertools import repeat
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from general import load_all_data

hep.style.use(hep.styles.CMS)


def load_samples(sig_dir: str, bkg_dir: str):
    sig_data = load_all_data(sig_dir)
    bkg_data = load_all_data(bkg_dir)
    return sig_data, bkg_data


def visualize_weights(weight_matrix, x_bin_edges, y_bin_edges, output_path, ylabel=r"$\eta$", xlabel=r"$p_T$"):
    x_labels = [f"{label:9.0f}" for label in (x_bin_edges[1:] + x_bin_edges[:-1]) / 2]
    y_labels = [f"{label:9.2f}" for label in (y_bin_edges[1:] + y_bin_edges[:-1]) / 2]
    sns.set(rc={"figure.figsize": (16, 9)})
    heatmap = sns.heatmap(weight_matrix)
    heatmap.set_xticks(range(len(x_labels)))
    heatmap.set_xticklabels(x_labels)
    heatmap.set_yticks(range(len(y_labels)))
    heatmap.set_yticklabels(y_labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    for i, label in enumerate(heatmap.xaxis.get_ticklabels()):
        if i % 5 != 0:
            label.set_visible(False)
    for i, label in enumerate(heatmap.yaxis.get_ticklabels()):
        if i % 5 != 0:
            label.set_visible(False)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def create_matrix(data, y_bin_edges, x_bin_edges, y_property, x_property):
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
    x_property_ = getattr(p4s, x_property).to_numpy()
    y_property_ = getattr(p4s, y_property).to_numpy()
    if y_property == "theta":
        y_property_ = np.rad2deg(y_property_)
    matrix = np.histogram2d(y_property_, x_property_, bins=(y_bin_edges, x_bin_edges))[0]
    normalized_matrix = matrix / np.sum(matrix)
    return normalized_matrix


def get_weight_matrix(target_matrix, comp_matrix):
    weights = np.minimum(target_matrix, comp_matrix) / target_matrix
    return np.nan_to_num(weights, nan=0.0)


def process_files(weight_matrix, theta_bin_edges, pt_bin_edges, data_dir, use_multiprocessing=True):
    data_paths = glob.glob(os.path.join(data_dir, "*.parquet"))
    if use_multiprocessing:
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(
            process_single_file, zip(data_paths, repeat(weight_matrix), repeat(theta_bin_edges), repeat(pt_bin_edges))
        )
    else:
        for input_path in data_paths:
            process_single_file(
                input_path=input_path,
                weight_matrix=weight_matrix,
                theta_bin_edges=theta_bin_edges,
                pt_bin_edges=pt_bin_edges,
            )


def get_weights(data, weight_matrix, theta_bin_edges, pt_bin_edges):
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
    theta_values = np.rad2deg(p4s.theta.to_numpy())
    pt_values = p4s.p.to_numpy()
    theta_bin = np.digitize(theta_values, bins=(theta_bin_edges[1:] + theta_bin_edges[:-1]) / 2) - 1
    pt_bin = np.digitize(pt_values, bins=(pt_bin_edges[1:] + pt_bin_edges[:-1]) / 2) - 1
    matrix_loc = np.concatenate([theta_bin.reshape(-1, 1), pt_bin.reshape(-1, 1)], axis=1)
    weights = ak.from_iter([weight_matrix[tuple(loc)] for loc in matrix_loc])
    return weights


def process_single_file(input_path, weight_matrix, theta_bin_edges, pt_bin_edges):
    data = ak.from_parquet(input_path)
    weights = get_weights(data, weight_matrix, theta_bin_edges, pt_bin_edges)
    merged_info = {field: data[field] for field in data.fields}
    merged_info.update({"weight": weights})
    print(f"Adding weights to {input_path}")
    ak.to_parquet(ak.Record(merged_info), input_path)


def plot_weighting_results(sig_data, bkg_data, sig_weights, bkg_weights, output_dir):
    bkg_p4s = vector.awk(
        ak.zip(
            {
                "mass": bkg_data.gen_jet_p4s.tau,
                "x": bkg_data.gen_jet_p4s.x,
                "y": bkg_data.gen_jet_p4s.y,
                "z": bkg_data.gen_jet_p4s.z,
            }
        )
    )
    sig_p4s = vector.awk(
        ak.zip(
            {
                "mass": sig_data.gen_jet_p4s.tau,
                "x": sig_data.gen_jet_p4s.x,
                "y": sig_data.gen_jet_p4s.y,
                "z": sig_data.gen_jet_p4s.z,
            }
        )
    )
    plot_distributions(
        sig_values=sig_p4s.pt,
        bkg_values=bkg_p4s.pt,
        bkg_weights=np.ones(len(bkg_p4s.pt)) / len(bkg_p4s.pt),
        sig_weights=np.ones(len(sig_p4s.pt)) / len(sig_p4s.pt),
        output_path=os.path.join(output_dir, "pt_normalized_unweighted.png"),
        xlabel=r"$p_T$",
    )
    plot_distributions(
        sig_values=sig_p4s.pt,
        bkg_values=bkg_p4s.pt,
        bkg_weights=bkg_weights / sum(bkg_weights),
        sig_weights=sig_weights / sum(sig_weights),
        output_path=os.path.join(output_dir, "pt_normalized_weighted.png"),
        xlabel=r"$p_T$",
    )
    plot_distributions(
        sig_values=sig_p4s.eta,
        bkg_values=bkg_p4s.eta,
        bkg_weights=np.ones(len(bkg_p4s.pt)) / len(bkg_p4s.pt),
        sig_weights=np.ones(len(sig_p4s.pt)) / len(sig_p4s.pt),
        output_path=os.path.join(output_dir, "eta_normalized_unweighted.png"),
        xlabel=r"$\eta$",
    )
    plot_distributions(
        sig_values=sig_p4s.eta,
        bkg_values=bkg_p4s.eta,
        bkg_weights=bkg_weights / sum(bkg_weights),
        sig_weights=sig_weights / sum(sig_weights),
        output_path=os.path.join(output_dir, "eta_normalized_weighted.png"),
        xlabel=r"$\eta$",
    )


def plot_distributions(sig_values, bkg_values, bkg_weights, sig_weights, output_path, xlabel=r"$p_T$"):
    bkg_hist, bin_edges = np.histogram(bkg_values, weights=bkg_weights, bins=100)
    sig_hist = np.histogram(sig_values, weights=sig_weights, bins=bin_edges)[0]
    hep.histplot(bkg_hist, bins=bin_edges, histtype="step", label="BKG", hatch="//")
    hep.histplot(sig_hist, bins=bin_edges, histtype="step", label="SIG", hatch="\\\\")
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")


@hydra.main(config_path="../config", config_name="weighting", version_base=None)
def main(cfg: DictConfig):
    sig_data, bkg_data = load_samples(sig_dir=cfg.samples.ZH_Htautau.output_dir, bkg_dir=cfg.samples.QCD.output_dir)
    eta_bin_edges = np.linspace(
        cfg.weighting.variables.eta.range[0], cfg.weighting.variables.eta.range[1], cfg.weighting.variables.eta.n_bins
    )
    theta_bin_edges = np.linspace(
        cfg.weighting.variables.theta.range[0], cfg.weighting.variables.theta.range[1], cfg.weighting.variables.theta.n_bins
    )
    pt_bin_edges = np.linspace(
        cfg.weighting.variables.pt.range[0], cfg.weighting.variables.pt.range[1], cfg.weighting.variables.pt.n_bins
    )
    sig_matrix = create_matrix(sig_data, eta_bin_edges, pt_bin_edges, y_property="eta", x_property="pt")
    bkg_matrix = create_matrix(bkg_data, eta_bin_edges, pt_bin_edges, y_property="eta", x_property="pt")
    sig_weights = get_weight_matrix(target_matrix=sig_matrix, comp_matrix=bkg_matrix)
    bkg_weights = get_weight_matrix(target_matrix=bkg_matrix, comp_matrix=sig_matrix)
    sig_output_path = os.path.join(cfg.samples.ZH_Htautau.output_dir, "signal_weights.png")
    visualize_weights(sig_weights, pt_bin_edges, eta_bin_edges, sig_output_path)
    output_dir = os.path.abspath(os.path.join(cfg.samples.QCD.output_dir, os.pardir))
    bkg_output_path = os.path.join(output_dir, "bkg_weights.png")
    visualize_weights(bkg_weights, pt_bin_edges, eta_bin_edges, bkg_output_path)

    sig_matrix_p_theta = create_matrix(sig_data, theta_bin_edges, pt_bin_edges, y_property="theta", x_property="p")
    bkg_matrix_p_theta = create_matrix(bkg_data, theta_bin_edges, pt_bin_edges, y_property="theta", x_property="p")
    sig_weights_p_theta = get_weight_matrix(target_matrix=sig_matrix_p_theta, comp_matrix=bkg_matrix_p_theta)
    bkg_weights_p_theta = get_weight_matrix(target_matrix=bkg_matrix_p_theta, comp_matrix=sig_matrix_p_theta)
    sig_output_path_p_theta = os.path.join(output_dir, "signal_weights_p_theta.png")
    visualize_weights(
        weight_matrix=sig_weights_p_theta,
        x_bin_edges=pt_bin_edges,
        y_bin_edges=theta_bin_edges,
        output_path=sig_output_path_p_theta,
        ylabel=r"$\theta$",
        xlabel="p",
    )
    bkg_output_path_p_theta = os.path.join(cfg.samples.QCD.output_dir, "bkg_weights_p_theta.png")
    visualize_weights(
        weight_matrix=bkg_weights_p_theta,
        x_bin_edges=pt_bin_edges,
        y_bin_edges=theta_bin_edges,
        output_path=bkg_output_path_p_theta,
        ylabel=r"$\theta$",
        xlabel="p",
    )
    process_files(
        weight_matrix=sig_weights_p_theta,
        theta_bin_edges=theta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.ZH_Htautau.output_dir,
        use_multiprocessing=cfg.use_multiprocessing,
    )
    process_files(
        weight_matrix=bkg_weights_p_theta,
        theta_bin_edges=theta_bin_edges,
        pt_bin_edges=pt_bin_edges,
        data_dir=cfg.samples.QCD.output_dir,
        use_multiprocessing=cfg.use_multiprocessing,
    )
    sig_weights = get_weights(sig_data, sig_weights_p_theta, theta_bin_edges, pt_bin_edges)
    bkg_weights = get_weights(bkg_data, bkg_weights_p_theta, theta_bin_edges, pt_bin_edges)
    plot_weighting_results(
        sig_data=sig_data,
        bkg_data=bkg_data,
        sig_weights=sig_weights,
        bkg_weights=bkg_weights,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
