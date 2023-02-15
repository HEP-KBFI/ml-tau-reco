"""
https://github.com/HEP-KBFI/ml-tau-reco/issues/10

src/metrics.py  \
  --model outputs/hps/signal.parquet:outputs/hps/bkg.parquet:HPS \
  --model outputs/hps_deeptau/signal.parquet:outputs/hps_deeptau/bkg.parquet:HPS-DeepTau \
  ...
"""
import os
import hydra
import vector
import mplhep
import numpy as np
import awkward as ak
import plotting as pl
import matplotlib.pyplot as plt
from metrics_tools import Histogram
from general import load_all_data, get_reduced_decaymodes

mplhep.style.use(mplhep.styles.CMS)


def plot_eff_fake(eff_fake_data, key, cfg, output_dir, cut):
    metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_{key}.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        algorithms = eff_fake_data.keys()
        for algorithm in algorithms:
            eff_fake_numerator = eff_fake_data[algorithm]['numerator']
            eff_fake_numerator = eff_fake_numerator[eff_fake_numerator.tauClassifier > cut]
            eff_fake_denominator = eff_fake_data[algorithm]['denominator']
            eff_fake_p4_num = vector.awk(
                ak.zip(
                    {
                        "mass": eff_fake_numerator.gen_jet_tau_p4s.tau,
                        "x": eff_fake_numerator.gen_jet_tau_p4s.x,
                        "y": eff_fake_numerator.gen_jet_tau_p4s.y,
                        "z": eff_fake_numerator.gen_jet_tau_p4s.z,
                    }
                )
            )
            eff_fake_p4_denom = vector.awk(
                ak.zip(
                    {
                        "mass": eff_fake_denominator.gen_jet_tau_p4s.tau,
                        "x": eff_fake_denominator.gen_jet_tau_p4s.x,
                        "y": eff_fake_denominator.gen_jet_tau_p4s.y,
                        "z": eff_fake_denominator.gen_jet_tau_p4s.z,
                    }
                )
            )
            eff_fake_var_denom = getattr(eff_fake_p4_denom, metric.name)
            eff_fake_var_num = getattr(eff_fake_p4_num, metric.name)
            bin_edges = np.linspace(min(eff_fake_var_denom), max(eff_fake_var_denom), num=4)#metric.n_bins+1)
            denom_hist = Histogram(eff_fake_var_denom, bin_edges, 'denominator')
            num_hist = Histogram(eff_fake_var_num, bin_edges, 'denominator')
            eff_fake = num_hist/denom_hist
            plt.plot(eff_fake.bin_centers, eff_fake.data, label=algorithm)
        plt.grid()
        plt.legend()
        plt.xlabel(metric.name)
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
        left_bin_edge=np.min([gen_tau_vis_energies, reco_tau_energies]),
        right_bin_edge=np.max([gen_tau_vis_energies, reco_tau_energies]),
        y_label="Reconstructed tau energy",
        x_label="GenTau vis energy",
        title="Energy resolution",
    )


def plot_decaymode_reconstruction(sig_data, algorithm_output_dir):
    output_path = os.path.join(algorithm_output_dir, "decaymode_reconstruction.png")
    gen_tau_decaymodes = get_reduced_decaymodes(sig_data.gen_jet_tau_decaymode.to_numpy())
    reco_tau_decaymodes = get_reduced_decaymodes(sig_data.tau_decaymode.to_numpy())
    # Mapping of decaymodes needed, not all classes classified, such as [14: 'ThreeProngNPiZero']
    mapping = {
        0: r"$\pi^{\pm}$",
        1: r"$\pi^{\pm}\pi^{0}$",
        2: r"$\pi^{\pm}\pi^{0}\pi^{0}$",
        10: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}$",
        11: r"$\pi^{\pm}\pi^{\mp}\pi^{\pm}\pi^{0}$",
        15: "Other",
    }
    gen_tau_mask = gen_tau_decaymodes != -1
    reco_tau_mask = reco_tau_decaymodes != -1
    gen_tau_decaymodes_ = gen_tau_decaymodes[gen_tau_mask * reco_tau_mask]
    reco_tau_decaymodes_ = reco_tau_decaymodes[gen_tau_mask * reco_tau_mask]
    categories = [value for value in mapping.values()]
    pl.plot_classification_confusion_matrix(
        true_cats=gen_tau_decaymodes_, pred_cats=reco_tau_decaymodes_, categories=categories, output_path=output_path
    )


def calculate_eff_fake(data, ref_obj, cfg, tau_classifier_cut):
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
    tau_classifier_mask = data.tauClassifier > tau_classifier_cut
    # Need to also have some cuts for the generator tau, like abs(eta) > 2.4 and pt > 20.
    var_eff_fake = {}
    for variable in cfg.metrics.efficiency.variables:
        name = variable.name
        ref_var_ = getattr(ref_p4, name)
        bin_edges = np.linspace(variable.x_range[0], variable.x_range[1], num=variable.n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        ref_var_mask = ref_var_ != -1
        ref_var_pt_mask = ref_p4.pt > 20
        ref_var_eta_mask = abs(ref_p4.eta) < 2.5
        denominator = ref_var_[ref_var_mask * ref_var_pt_mask * ref_var_eta_mask]
        numerator = ref_var_[ref_var_mask * tau_classifier_mask * ref_var_pt_mask * ref_var_eta_mask]
        numerator_ = np.histogram(numerator, bins=bin_edges)[0]
        denominator_ = np.histogram(denominator, bins=bin_edges)[0]
        eff_fake = numerator_ / denominator_
        var_eff_fake[name] = {
            "x_values": bin_centers,
            "y_values": eff_fake,
            "eff_fake": sum(numerator_) / sum(denominator_),
        }
    return var_eff_fake


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


def calculate_efficiencies_fakerates(raw_numerator_data, denominator_data, variant=False):
    classifier_cuts = np.linspace(start=0, stop=1, num=1001)
    eff_fakes = []
    if variant:
        n_all = len(raw_numerator_data[raw_numerator_data.tauClassifier > 0])
    else:
        n_all = len(denominator_data)
    for cut in classifier_cuts:
        n_passing_cuts = len(raw_numerator_data[raw_numerator_data.tauClassifier > cut])
        eff_fake = n_passing_cuts/n_all
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
    for algorithm in algorithms:
        sig_input_dir = os.path.expandvars(cfg.algorithms[algorithm].sig_ntuples_dir)
        bkg_input_dir = os.path.expandvars(cfg.algorithms[algorithm].bkg_ntuples_dir)
        print(f"Loading signal data for {algorithm} from {sig_input_dir}")
        sig_data = load_all_data(sig_input_dir, n_files=cfg.plotting.n_files)
        print(f"Loading background data for {algorithm} from {bkg_input_dir}")
        bkg_data = load_all_data(bkg_input_dir, n_files=cfg.plotting.n_files)
        numerator_mask_e, denominator_mask_e = get_data_masks(sig_data, ref_obj="gen_jet_tau_p4s")
        numerator_mask_f, denominator_mask_f = get_data_masks(bkg_data, ref_obj="reco_jet_p4s")
        raw_numerator_data_e, denominator_data_e = sig_data[numerator_mask_e], sig_data[denominator_mask_e]
        raw_numerator_data_f, denominator_data_f = bkg_data[numerator_mask_f], bkg_data[denominator_mask_f]
        # Also need to calculate workingpoints
        efficiencies[algorithm] = calculate_efficiencies_fakerates(raw_numerator_data_e, denominator_data_e)
        fakerates[algorithm] = calculate_efficiencies_fakerates(raw_numerator_data_f, denominator_data_f)
        eff_data[algorithm] = {
            "numerator": raw_numerator_data_e,
            "denominator": denominator_data_e
        }
        fake_data[algorithm] = {
            "numerator": raw_numerator_data_f,
            "denominator": denominator_data_f
        }
        algorithm_output_dir = os.path.join(output_dir, algorithm)
        os.makedirs(algorithm_output_dir, exist_ok=True)
        plot_energy_resolution(sig_data, algorithm_output_dir)
        plot_decaymode_reconstruction(sig_data, algorithm_output_dir)
    cut = 0.5
    plot_eff_fake(eff_data, key="efficiencies", cfg=cfg, output_dir=output_dir, cut=cut)
    plot_eff_fake(fake_data, key="fakerates", cfg=cfg, output_dir=output_dir, cut=cut)
    plot_genvistau_gentau_correlation(sig_data, output_dir)
    plot_roc(efficiencies, fakerates, output_dir)


def plot_genvistau_gentau_correlation(sig_data, output_dir):
    vis_tau_pt = vector.awk(
        ak.zip(
            {
                "mass": sig_data.gen_jet_tau_p4s.tau,
                "x": sig_data.gen_jet_tau_p4s.x,
                "y": sig_data.gen_jet_tau_p4s.y,
                "z": sig_data.gen_jet_tau_p4s.z,
            }
        )
    ).pt.to_numpy()
    gen_jet_pt = vector.awk(
        ak.zip(
            {
                "mass": sig_data.gen_jet_p4s.tau,
                "x": sig_data.gen_jet_p4s.x,
                "y": sig_data.gen_jet_p4s.y,
                "z": sig_data.gen_jet_p4s.z,
            }
        )
    ).pt.to_numpy()
    mask = vis_tau_pt != 0
    vis_tau_pt_ = vis_tau_pt[mask]
    gen_jet_pt_ = gen_jet_pt[mask]
    output_path = os.path.join(output_dir, "validate_ntuple_genVisTauPt_vs_genJetPt.png")
    pl.plot_regression_confusion_matrix(
        y_true=gen_jet_pt_,
        y_pred=vis_tau_pt_,
        output_path=output_path,
        left_bin_edge=np.min([vis_tau_pt, gen_jet_pt]),
        right_bin_edge=np.max([vis_tau_pt, gen_jet_pt]),
        y_label=r"$p_T^{\tau_{vis}}$",
        x_label=r"$p_T^{genJet}$",
        title="",
    )


if __name__ == "__main__":
    plot_all_metrics()
