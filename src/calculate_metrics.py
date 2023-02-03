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
import numpy as np
import awkward as ak
import plotting as pl
import matplotlib.pyplot as plt
from general import load_all_data


def plot_eff_fake(algorithm_metrics, key, cfg, output_dir, cut):
    metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_{key}.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        for algorithm, values in algorithm_metrics.items():
            plt.plot(values[cut][metric.name]['x_values'], values[cut][metric.name]['y_values'], label=algorithm)
        plt.grid()
        plt.legend()
        plt.xlabel(metric.name)
        plt.ylabel(key)
        plt.title(f"tauClassifier > {cut}")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close('all')


def plot_energy_resolution(sig_data, algorithm_output_dir):
    output_path = os.path.join(algorithm_output_dir, "energy_resolution.png")
    gen_tau_vis_energies = sig_data.gen_jet_tau_vis_energy
    reco_tau_energies = vector.awk(
        ak.zip(
            {
                "mass": sig_data.tau_p4.tau,
                "x": sig_data.tau_p4.x,
                "y": sig_data.tau_p4.y,
                "z": sig_data.tau_p4.z,
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
        title="Energy resolution"
    )


def plot_decaymode_reconstruction(sig_data, algorithm_output_dir):
    output_path = os.path.join(algorithm_output_dir, "decaymode_reconstruction.png")
    gen_tau_decaymodes = sig_data.gen_jet_tau_decaymode.to_numpy()
    reco_tau_decaymodes = sig_data.tau_decaymode.to_numpy()
    print(gen_tau_decaymodes)
    print(gen_tau_decaymodes.shape)
    print(type(reco_tau_decaymodes))
    print(reco_tau_decaymodes.shape)
    # Mapping of decaymodes needed, not all classes classified, such as [14: 'ThreeProngNPiZero']
    mapping = {
        0: "\\pi^{\\pm}",
        1: "\\pi^{\\pm}\\pi^{0}",
        2: "\\pi^{\\pm}\\pi^{0}\\pi^{0}",
        10: "\\pi^{\\pm}\\pi^{\\mp}\\pi^{\\pm}",
        11: "\\pi^{\\pm}\\pi^{\\mp}\\pi^{\\pm}\\pi^{0}",
        15: "Other"
    }
    categories = [value for value in mapping.values()]
    pl.plot_classification_confusion_matrix(
        true_cats=gen_tau_decaymodes,
        pred_cats=reco_tau_decaymodes,
        categories=categories,
        output_path=output_path
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
    tau_p4 = vector.awk(
        ak.zip(
            {
                "mass": data.tau_p4.tau,
                "x": data.tau_p4.x,
                "y": data.tau_p4.y,
                "z": data.tau_p4.z,
            }
        )
    )
    tau_classifier_mask = data.tauClassifier > tau_classifier_cut
    # Need to also have some cuts for the generator tau, like abs(eta) > 2.4 and pt > 20.
    var_eff_fake = {}
    for variable in cfg.metrics.efficiency.variables:
        name = variable.name
        x_range = variable.x_range
        ref_var_ = getattr(ref_p4, name)
        bin_edges = np.linspace(variable.x_range[0], variable.x_range[1], num=variable.n_bins)
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

        ref_var_mask = ref_var_ != -1
        denominator = ref_var_
        numerator = ref_var_[tau_classifier_mask]

        numerator_ = np.histogram(numerator, bins=bin_edges)[0]
        denominator_ = np.histogram(denominator, bins=bin_edges)[0]
        eff_fake = numerator_/denominator_
        var_eff_fake[name] = {
            "x_values": bin_centers,
            "y_values": eff_fake
        }
    return var_eff_fake


def plot_roc(efficiencies, fakerates, cfg, output_dir, classifier_cuts):
    metrics = cfg.metrics.efficiency.variables
    for metric in metrics:
        output_path = os.path.join(output_dir, f"{metric.name}_ROC.png")
        fig, ax = plt.subplots(figsize=(12, 12))
        for (algorithm, efficiency_histos), (algorithm_, fakerate_histos) in zip(efficiencies.items(), fakerates.items()):
            fakerates = [np.nanmean(fakerate_histos[cut][metric.name]['y_values']) for cut in classifier_cuts]
            efficiencies = [np.nanmean(efficiency_histos[cut][metric.name]['y_values']) for cut in classifier_cuts]
            plt.plot(fakerates, efficiencies, label=algorithm)
        plt.grid()
        plt.legend()
        plt.xlabel("Fakerate")
        plt.ylabel("Efficiency")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close('all')


@hydra.main(config_path="../config", config_name="metrics", version_base=None)
def plot_all_metrics(cfg):
    algorithms = [algo for algo in cfg.algorithms if cfg.algorithms[algo].compare]
    assert len(algorithms) != 0, "No algorithms chosen for comparison"
    output_dir = cfg.plotting.output_dir
    os.makedirs(output_dir, exist_ok=True)
    efficiencies = {}
    fakerates = {}
    classifier_cuts = np.linspace(start=0, stop=1, num=51)
    for algorithm in algorithms:
        sig_input_dir = cfg.algorithms[algorithm].sig_ntuples_dir
        bkg_input_dir = cfg.algorithms[algorithm].bkg_ntuples_dir
        sig_data = load_all_data(sig_input_dir)
        bkg_data = load_all_data(bkg_input_dir)
        efficiencies[algorithm] = {}
        fakerates[algorithm] = {}
        for cut in classifier_cuts:
            efficiencies[algorithm][cut] = calculate_eff_fake(sig_data, "gen_jet_p4s", cfg, cut)
            fakerates[algorithm][cut] = calculate_eff_fake(bkg_data, "reco_jet_p4s", cfg, cut)
        algorithm_output_dir = os.path.join(output_dir, algorithm)
        os.makedirs(algorithm_output_dir, exist_ok=True)
        plot_energy_resolution(sig_data, algorithm_output_dir)
        # plot_decaymode_reconstruction(sig_data, algorithm_output_dir)
    plot_eff_fake(efficiencies, key="efficiencies", cfg=cfg, output_dir=output_dir, cut=0.96)
    plot_eff_fake(fakerates, key="fakerates", cfg=cfg, output_dir=output_dir, cut=0.96)
    plot_roc(efficiencies, fakerates, cfg, output_dir, classifier_cuts)


if __name__ == '__main__':
    plot_all_metrics()
