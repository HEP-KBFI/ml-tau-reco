"""
https://github.com/HEP-KBFI/ml-tau-reco/issues/10

src/metrics.py  \
  --model outputs/hps/signal.parquet:outputs/hps/bkg.parquet:HPS \
  --model outputs/hps_deeptau/signal.parquet:outputs/hps_deeptau/bkg.parquet:HPS-DeepTau \
  ...
"""
import os
import json
import yaml
import hydra
import vector
import numpy as np
import awkward as ak
import mplhep as hep
import plotting as pl
import matplotlib.pyplot as plt
from metrics_tools import Histogram
from general import get_reduced_decaymodes, load_data_from_paths

hep.style.use(hep.styles.CMS)


def plot_eff_fake(eff_fake_data, key, cfg, output_dir, cut):
    metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_{key}.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        algorithms = eff_fake_data.keys()
        for algorithm in algorithms:
            eff_fake_numerator = eff_fake_data[algorithm]["numerator"]
            eff_fake_numerator = eff_fake_numerator[eff_fake_numerator.tauClassifier > cut]
            eff_fake_denominator = eff_fake_data[algorithm]["denominator"]
            eff_fake_p4_num = vector.awk(
                ak.zip(
                    {
                        "mass": eff_fake_numerator.gen_jet_p4s.tau,
                        "x": eff_fake_numerator.gen_jet_p4s.x,
                        "y": eff_fake_numerator.gen_jet_p4s.y,
                        "z": eff_fake_numerator.gen_jet_p4s.z,
                    }
                )
            )
            eff_fake_p4_denom = vector.awk(
                ak.zip(
                    {
                        "mass": eff_fake_denominator.gen_jet_p4s.tau,
                        "x": eff_fake_denominator.gen_jet_p4s.x,
                        "y": eff_fake_denominator.gen_jet_p4s.y,
                        "z": eff_fake_denominator.gen_jet_p4s.z,
                    }
                )
            )
            eff_fake_var_denom = getattr(eff_fake_p4_denom, metric.name).to_numpy()
            eff_fake_var_num = getattr(eff_fake_p4_num, metric.name).to_numpy()
            bin_edges = np.linspace(min(eff_fake_var_denom), max(eff_fake_var_denom), metric.n_bins + 1)
            denom_hist = Histogram(eff_fake_var_denom, bin_edges, "denominator")
            num_hist = Histogram(eff_fake_var_num, bin_edges, "numerator")
            eff_fake = num_hist / denom_hist
            plt.plot(eff_fake.bin_centers, eff_fake.data, label=algorithm)
            # plt.errorbar(eff_fake.bin_centers, eff_fake.data, yerr=eff_fake.uncertainties, label=algorithm)
        plt.grid()
        plt.legend()
        plt.xlabel(f"gen_jet_{metric.name}")
        plt.ylabel(key)
        if key == "fakerates":
            plt.yscale("log")
        plt.title(f"tauClassifier > {cut}")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close("all")


def plot_energy_resolution(sig_data, algorithm_output_dir):
    output_path = os.path.join(algorithm_output_dir, "energy_resolution.png")
    gen_tau_vis_energies = sig_data.gen_jet_tau_vis_energy
    reco_tau_energies = vector.awk(
        ak.zip(
            {
                "mass": sig_data.tau_p4s.tau,
                "x": sig_data.tau_p4s.x,
                "y": sig_data.tau_p4s.y,
                "z": sig_data.tau_p4s.z,
            }
        )
    ).energy
    gen_tau_vis_energies = gen_tau_vis_energies.to_numpy()
    reco_tau_energies = reco_tau_energies.to_numpy()
    pl.plot_regression_confusion_matrix(
        y_true=gen_tau_vis_energies,
        y_pred=reco_tau_energies,
        output_path=output_path,
        left_bin_edge=np.min(gen_tau_vis_energies),
        right_bin_edge=np.max(gen_tau_vis_energies),
        y_label="Reconstructed tau energy",
        x_label="GenTau vis energy",
        title="Energy resolution",
    )


def plot_decaymode_reconstruction(sig_data, algorithm_output_dir, cfg):
    output_path = os.path.join(algorithm_output_dir, "decaymode_reconstruction.png")
    gen_tau_decaymodes = get_reduced_decaymodes(sig_data.gen_jet_tau_decaymode.to_numpy())
    reco_tau_decaymodes = get_reduced_decaymodes(sig_data.tau_decaymode.to_numpy())
    mapping = {
        0: r"$\pi^{\pm}$",
        1: r"$\pi^{\pm}\pi^{0}$",
        2: r"$\pi^{\pm}\pi^{0}\pi^{0}$",
        10: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}$",
        11: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}\pi^{0}$",
        15: "Other",
    }
    classifier_cut = cfg.metrics.WPs.HPS_wo_quality_cuts.Medium
    gen_tau_decaymodes_ = gen_tau_decaymodes[sig_data.tauClassifier > classifier_cut]
    reco_tau_decaymodes_ = reco_tau_decaymodes[sig_data.tauClassifier > classifier_cut]
    categories = [value for value in mapping.values()]
    pl.plot_classification_confusion_matrix(
        true_cats=gen_tau_decaymodes_, pred_cats=reco_tau_decaymodes_, categories=categories, output_path=output_path
    )


def plot_roc(efficiencies, fakerates, output_dir):
    output_path = os.path.join(output_dir, "ROC.png")
    algorithms = efficiencies.keys()
    fig, ax = plt.subplots(figsize=(12, 12))
    for algorithm in algorithms:
        plt.plot(efficiencies[algorithm], fakerates[algorithm], label=algorithm)
    plt.grid()
    plt.legend()
    plt.ylabel("Fakerate")
    plt.xlabel("Efficiency")
    plt.ylim((1e-5, 1))
    plt.yscale("log")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")


def plot_tauClassifier_correlation(sig_data, output_dir):
    p4s = vector.awk(
        ak.zip(
            {
                "mass": sig_data["reco_jet_p4s"].tau,
                "x": sig_data["reco_jet_p4s"].x,
                "y": sig_data["reco_jet_p4s"].y,
                "z": sig_data["reco_jet_p4s"].z,
            }
        )
    )
    tc = sig_data["tauClassifier"]
    for var in ["eta", "pt", "phi"]:
        variable = getattr(p4s, var)
        plt.scatter(variable, tc, alpha=0.3, marker="x")
        plt.title(var)
        output_path = os.path.join(output_dir, f"tauClassifier_corr_{var}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close("all")


def get_data_masks(data, ref_obj):
    ref_p4 = vector.awk(
        ak.zip(
            {
                "mass": data[ref_obj].tau,
                "x": data[ref_obj].x,
                "y": data[ref_obj].y,
                "z": data[ref_obj].z,
            }
        )
    )

    tau_p4 = vector.awk(
        ak.zip(
            {
                "mass": data.tau_p4s.tau,
                "x": data.tau_p4s.x,
                "y": data.tau_p4s.y,
                "z": data.tau_p4s.z,
            }
        )
    )
    # Denominator
    ref_var_pt_mask = ref_p4.pt > 20
    ref_var_theta_mask1 = abs(np.rad2deg(ref_p4.theta)) < 170
    ref_var_theta_mask2 = abs(np.rad2deg(ref_p4.theta)) > 10
    denominator_mask = ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2

    # Numerator
    tau_pt_mask = tau_p4.pt > 20
    tau_theta_mask1 = abs(np.rad2deg(tau_p4.theta)) < 170
    tau_theta_mask2 = abs(np.rad2deg(tau_p4.theta)) > 10
    numerator_mask = tau_pt_mask * tau_theta_mask1 * tau_theta_mask2

    full_numerator_mask = numerator_mask * denominator_mask
    return full_numerator_mask, denominator_mask


def calculate_efficiencies_fakerates(raw_numerator_data, denominator_data, classifier_cuts, variant=False):
    eff_fakes = []
    if variant:
        n_all = len(raw_numerator_data[raw_numerator_data.tauClassifier > 0])
    else:
        n_all = len(denominator_data)
    for cut in classifier_cuts:
        n_passing_cuts = len(raw_numerator_data[raw_numerator_data.tauClassifier > cut])
        eff_fake = n_passing_cuts / n_all
        eff_fakes.append(eff_fake)
    return eff_fakes


@hydra.main(config_path="../config", config_name="metrics", version_base=None)
def plot_all_metrics(cfg):
    algorithms = [algo for algo in cfg.algorithms if cfg.algorithms[algo].compare]
    assert len(algorithms) != 0, "No algorithms chosen for comparison"
    output_dir = cfg.plotting.output_dir
    os.makedirs(output_dir, exist_ok=True)
    efficiencies = {}
    fakerates = {}
    eff_data = {}
    fake_data = {}
    tauClassifiers = {algo: {} for algo in algorithms}
    for algorithm in algorithms:
        sig_input_dir = os.path.expandvars(cfg.algorithms[algorithm].sig_ntuples_dir)
        bkg_input_dir = os.path.expandvars(cfg.algorithms[algorithm].bkg_ntuples_dir)
        # sig_paths = [
        #     os.path.join(sig_input_dir, os.path.basename(path)) for path in cfg.datasets.test.paths if "ZH_Htautau" in path
        # ]
        # bkg_paths = [
        #     os.path.join(bkg_input_dir, os.path.basename(path)) for path in cfg.datasets.test.paths if "QCD" in path
        # ]

        if algorithm != "FastCMSTau" and algorithm != "SimpleDNN":
            sig_paths = [
                os.path.join(sig_input_dir, os.path.basename(path))
                for path in cfg.datasets.test.paths
                if "ZH_Htautau" in path
            ]
            bkg_paths = [
                os.path.join(bkg_input_dir, os.path.basename(path)) for path in cfg.datasets.test.paths if "QCD" in path
            ]
        else:
            with open("/home/laurits/ml-tau-reco/config/datasets/test.yaml_", "r") as stream:
                data = yaml.safe_load(stream)
            all_paths = data["test"]["paths"]
            sig_paths = [os.path.join(sig_input_dir, os.path.basename(path)) for path in all_paths if "ZH_Htautau" in path]
            bkg_paths = [os.path.join(bkg_input_dir, os.path.basename(path)) for path in all_paths if "QCD" in path]

        sig_data = load_data_from_paths(sig_paths, n_files=cfg.plotting.n_files)
        bkg_data = load_data_from_paths(bkg_paths, n_files=cfg.plotting.n_files)
        tauClassifiers[algorithm] = {"sig": sig_data.tauClassifier, "bkg": bkg_data.tauClassifier}
        numerator_mask_e, denominator_mask_e = get_data_masks(sig_data, ref_obj="gen_jet_tau_p4s")
        numerator_mask_f, denominator_mask_f = get_data_masks(bkg_data, ref_obj="gen_jet_p4s")
        raw_numerator_data_e, denominator_data_e = sig_data[numerator_mask_e], sig_data[denominator_mask_e]
        raw_numerator_data_f, denominator_data_f = bkg_data[numerator_mask_f], bkg_data[denominator_mask_f]
        # Also need to calculate workingpoints
        classifier_cuts = np.linspace(start=0, stop=1, num=1001)
        efficiencies[algorithm] = calculate_efficiencies_fakerates(raw_numerator_data_e, denominator_data_e, classifier_cuts)
        fakerates[algorithm] = calculate_efficiencies_fakerates(raw_numerator_data_f, denominator_data_f, classifier_cuts)
        eff_data[algorithm] = {"numerator": raw_numerator_data_e, "denominator": denominator_data_e}
        fake_data[algorithm] = {"numerator": raw_numerator_data_f, "denominator": denominator_data_f}
        algorithm_output_dir = os.path.join(output_dir, algorithm)
        os.makedirs(algorithm_output_dir, exist_ok=True)
        save_wps(efficiencies[algorithm], classifier_cuts, algorithm_output_dir)
        plot_energy_resolution(raw_numerator_data_e, algorithm_output_dir)
        plot_decaymode_reconstruction(raw_numerator_data_e, algorithm_output_dir, cfg)
    plot_roc(efficiencies, fakerates, output_dir)
    classifier_cut = cfg.metrics.WPs.HPS_wo_quality_cuts.Medium
    plot_eff_fake(eff_data, key="efficiencies", cfg=cfg, output_dir=output_dir, cut=classifier_cut)
    plot_eff_fake(fake_data, key="fakerates", cfg=cfg, output_dir=output_dir, cut=classifier_cut)
    plot_tauClassifiers(tauClassifiers, "sig", os.path.join(output_dir, "tauClassifier_sig.png"))
    plot_tauClassifiers(tauClassifiers, "bkg", os.path.join(output_dir, "tauClassifier_bkg.png"))


def plot_tauClassifiers(tauClassifiers, dtype, output_path):
    for name, tC in tauClassifiers.items():
        bin_edges = np.linspace(0, 1, 21)
        hist = np.histogram(tC[dtype], bins=bin_edges)[0]
        hep.histplot(hist, bins=bin_edges, histtype="step", label=name, ls="--")
        plt.xlabel("tauClassifier")
        plt.yscale("log")
        plt.legend()
    plt.savefig(output_path, bbox_inches="tight")


def save_wps(efficiencies, classifier_cuts, algorithm_output_dir):
    working_points = {"Loose": 0.4, "Medium": 0.6, "Tight": 0.8}
    wp_file_path = os.path.join(algorithm_output_dir, "working_points.json")
    wp_values = {}
    for wp_name, wp_value in working_points.items():
        diff = abs(np.array(efficiencies) - wp_value)
        idx = np.argmin(diff)
        if not diff[idx] / wp_value > 0.1:
            cut = classifier_cuts[idx]
        else:
            cut = -1
        wp_values[wp_name] = cut
    with open(wp_file_path, "wt") as out_file:
        json.dump(wp_values, out_file, indent=4)


if __name__ == "__main__":
    plot_all_metrics()
