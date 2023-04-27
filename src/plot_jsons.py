import os
import json
import hydra
import matplotlib
import numpy as np
import mplhep as hep
import plotting as pl
from general import load_json
import calculate_metrics as cm
import matplotlib.pyplot as plt
from metrics_tools import Histogram

hep.style.use(hep.styles.CMS)


@hydra.main(config_path="../config", config_name="json_plotting", version_base=None)
def plot_json(cfg):
    default_dir = os.path.expandvars(cfg.input_dir)
    os.makedirs(cfg.plotting.output_dir, exist_ok=True)
    input_dir = {
        algorithm: (default_dir if cfg.plotting_algorithms[algorithm] == "" else cfg.plotting_algorithms[algorithm])
        for algorithm in cfg.plotting_algorithms
    }
    if cfg.plotting_metrics.fakerate:
        fakerates = {}
        for algorithm in cfg.plotting_algorithms:
            if algorithm not in ["SimpleDNN", "FastCMSTau", "HPS_with_quality_cuts"]:
                fakerates[algorithm] = {}
                for metric_entry in cfg.metrics.efficiency.variables:
                    metric = metric_entry.name
                    input_path = os.path.join(input_dir[algorithm], f"{metric}_fakerates_{algorithm}.json")
                    fakerates[algorithm][metric] = load_json(input_path)
        plot_eff_fake(fakerates, "fakerates", cfg, cfg.plotting.output_dir)
    if cfg.plotting_metrics.efficiency:
        efficiencies = {}
        for algorithm in cfg.plotting_algorithms:
            if algorithm not in ["SimpleDNN", "FastCMSTau", "HPS_with_quality_cuts"]:
                efficiencies[algorithm] = {}
                for metric_entry in cfg.metrics.efficiency.variables:
                    metric = metric_entry.name
                    input_path = os.path.join(input_dir[algorithm], f"{metric}_efficiencies_{algorithm}.json")
                    efficiencies[algorithm][metric] = load_json(input_path)
        plot_eff_fake(efficiencies, "efficiency", cfg, cfg.plotting.output_dir)
    if cfg.plotting_metrics.ROC:
        input_path = os.path.join(default_dir, "roc.json")
        roc_info = load_json(input_path)
        for algo, dir_ in input_dir.items():
            if dir_ != default_dir:
                algo_input = os.path.join(dir_, "roc.json")
                algo_roc_info = load_json(algo_input)
                roc_info["efficiencies"][algo] = algo_roc_info["efficiencies"][algo]
                roc_info["fakerates"][algo] = algo_roc_info["fakerates"][algo]
            x = np.array(roc_info["efficiencies"][algo])
            y = np.array(roc_info["fakerates"][algo])
            print(f"Algorithm {algo} \t eff: {min(x[x > 0])} \t fake: {min(y[x > 0])}")
            print(f"Algorithm {algo} \t eff: {max(x[x > 0])} \t fake: {max(y[x > 0])}")
            print("______________________________________")
        roc_plotting_info = {}
        roc_plotting_info["efficiencies"] = {algo: roc_info["efficiencies"][algo] for algo in cfg.plotting_algorithms}
        roc_plotting_info["fakerates"] = {algo: roc_info["fakerates"][algo] for algo in cfg.plotting_algorithms}
        cm.plot_roc(roc_plotting_info["efficiencies"], roc_plotting_info["fakerates"], cfg.plotting.output_dir)
    # if cfg.plotting_metrics.efficiency and cfg.plotting_metrics.fakerate:
    #     for algorithm in cfg.plotting_algorithms:

    #         find_corresponding_fakerate(efficiencies, fakerates)
    # find_corresponding_fakerate(efficiencies['pt'], fakerates['pt'], efficiency_wp)
    if cfg.plotting_metrics.tauClassifier:
        algo_names = {algorithm: algorithm for algorithm in cfg.plotting_algorithms}
        algo_names["FastCMSTau"] = "JINST 17 (2022) P07023"
        algo_names["HPS"] = "HPS cut-based"
        algo_names["HPS_DeepTau"] = "HPS + DeepTau"
        for algorithm in cfg.plotting_algorithms:
            classifier_input_dir = os.path.join(input_dir[algorithm], algorithm, "tauClassifier.json")
            tauClassifiers = load_json(classifier_input_dir)
            wp_info_path = os.path.join(input_dir[algorithm], algorithm, "working_points.json")
            medium_wp = load_json(wp_info_path)["Medium"]
            output_dir = os.path.join(cfg.plotting.output_dir, algorithm)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{algorithm}_tauClassifier.pdf")
            cm.plot_algo_tauClassifiers(
                tauClassifiers["tauClassifiers"],
                output_path,
                medium_wp,
                plot_train=algorithm != "HPS" and algorithm != "HPS_with_quality_cuts",
                algo_name=algo_names[algorithm],
            )
    if cfg.plotting_metrics.decaymode:
        for algorithm in cfg.plotting_algorithms:
            dm_input_path = os.path.join(input_dir[algorithm], algorithm, "decaymode_reconstruction.json")
            decaymodes_info = load_json(dm_input_path)
            dm_output_dir = os.path.join(cfg.plotting.output_dir, algorithm)
            dm_output_path = os.path.join(dm_output_dir, "decaymode_reconstruction.pdf")
            os.makedirs(dm_output_dir, exist_ok=True)
            print_category_fractions(decaymodes_info["gen"], decaymodes_info["categories"], algorithm)
            pl.plot_decaymode_correlation_matrix_removed_row(
                true_cats=decaymodes_info["gen"],
                pred_cats=decaymodes_info["reco"],
                categories=decaymodes_info["categories"],
                output_path=dm_output_path,
                y_label=r"Reconstructed $\tau$ decay mode",
                x_label=r"Generated $\tau$ decay mode",
                figsize=None,
            )
    if cfg.plotting_metrics.HPS_comparison:
        efficiencies = {}
        fakerates = {}
        output_dir = os.path.join(cfg.plotting.output_dir, "Comparison_HPS_cuts")
        os.makedirs(output_dir, exist_ok=True)
        for algo in cfg.HPS_comparison:
            input_path = os.path.join(cfg.HPS_comparison[algo], "roc.json")
            HPS_comp_roc_info = load_json(input_path)
            if algo == "HPS":
                algorithm = "HPS_NOquality"
            else:
                algorithm = "HPS_quality"
            efficiencies[algorithm] = HPS_comp_roc_info["efficiencies"][algo]
            fakerates[algorithm] = HPS_comp_roc_info["fakerates"][algo]
        cm.plot_roc(efficiencies, fakerates, output_dir, ylim=(1e-3, 1), xlim=(0.5, 0.95), title="cut-based HPS algorithm")

    # if cfg.plotting_metrics.energy_resolution:
    #     for algorithm in cfg.plotting_algorithms:
    #     plot_energy_resolution


def print_category_fractions(gen_decaymodes, categories, algorithm):
    print(f"Decaymode fractions for {algorithm}")
    for category in set(gen_decaymodes):
        number_cat_entires = sum(np.array(gen_decaymodes) == category)
        total_entries = len(gen_decaymodes)
        fraction = number_cat_entires / total_entries
        print(f"{category}: {fraction}")


def save_wps(efficiencies, classifier_cuts, algorithm_output_dir):
    working_points = {"Loose": 0.4, "Medium": 0.6, "Tight": 0.8}
    wp_file_path = os.path.join(algorithm_output_dir, "working_points.json")
    wp_values = {}
    for wp_name, wp_value in working_points.items():
        diff = abs(np.array(efficiencies) - wp_value)
        idx = np.argmin(diff)
        if not diff[idx] / wp_value > 0.3:
            cut = classifier_cuts[idx]
        else:
            cut = -1
        wp_values[wp_name] = cut
    with open(wp_file_path, "wt") as out_file:
        json.dump(wp_values, out_file, indent=4)
    return wp_values["Medium"]


def plot_eff_fake(eff_fake_data, key, cfg, output_dir):
    markers = ["o", "^", "s", "v", "*", "P"]
    if key == "fakerates":
        metrics = cfg.metrics.fakerate.variables
    else:
        metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_{key}.pdf")
        fig, ax = plt.subplots(figsize=(12, 12))
        algorithms = eff_fake_data.keys()
        algo_names = {algorithm: algorithm for algorithm in algorithms}
        algo_names["FastCMSTau"] = "JINST 17 (2022) P07023"
        algo_names["HPS"] = "HPS cut-based"
        algo_names["HPS_DeepTau"] = "HPS + DeepTau"
        for i, algorithm in enumerate(algorithms):
            if metric.name == "theta":
                numerator_ = np.rad2deg(np.array(eff_fake_data[algorithm][metric.name]["numerator"]))
                denominator_ = np.rad2deg(np.array(eff_fake_data[algorithm][metric.name]["denominator"]))
                numerator = 90 - np.abs(numerator_ - 90)
                denominator = 90 - np.abs(denominator_ - 90)
            else:
                numerator = eff_fake_data[algorithm][metric.name]["numerator"]
                denominator = eff_fake_data[algorithm][metric.name]["denominator"]
            bin_edges = np.linspace(metric.x_range[0], metric.x_range[1], num=metric.n_bins + 1)
            numerator_hist = Histogram(numerator, bin_edges, "numerator")
            denominator_hist = Histogram(denominator, bin_edges, "denominator")
            resulting_hist = numerator_hist / denominator_hist
            plt.errorbar(
                resulting_hist.bin_centers,
                resulting_hist.binned_data,
                xerr=resulting_hist.bin_halfwidths,
                yerr=resulting_hist.uncertainties,
                ms=20,
                marker=markers[i],
                linestyle="None",
                label=algo_names[algorithm],
            )
        plt.grid()
        plt.legend()
        matplotlib.rcParams["axes.unicode_minus"] = False
        if metric.name == "pt":
            if key == "fakerates":
                plt.xlabel(r"$p_T^{gen\mathrm{-}jet}\,\, [GeV]$")
            else:
                plt.xlabel(r"$p_T^{gen\mathrm{-}\tau_h}\,\, [GeV]$")
        elif metric.name == "eta":
            if key == "fakerates":
                plt.xlabel(r"$\eta^{gen\mathrm{-}jet}\,\, [GeV]$")
            else:
                plt.xlabel(r"$\eta^{gen\mathrm{-}\tau_h}\,\, [GeV]$")
        elif metric.name == "theta":
            if key == "fakerates":
                plt.xlabel(r"$\theta^{gen\mathrm{-}jet}\,\, [ ^{o} ]$")
            else:
                plt.xlabel(r"$\theta^{gen\mathrm{-}\tau_h}\,\, [ ^{o} ]$")
        if key == "fakerates":
            plt.ylabel(r"$P_{misid}$")
            plt.ylim((5e-6, 2e-2))
            plt.yscale("log")
        else:
            plt.ylabel(r"$\varepsilon_{\tau}$")
        plt.savefig(output_path, format="pdf")
        plt.close("all")


def calculate_bin_centers(edges: list) -> np.array:
    bin_widths = np.array([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
    bin_centers = []
    for i in range(len(edges) - 1):
        bin_centers.append(edges[i] + (bin_widths[i] / 2))
    return np.array(bin_centers), bin_widths / 2


if __name__ == "__main__":
    plot_json()
