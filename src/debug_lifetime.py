import os
import glob
import hydra
import vector
import numpy as np
import general as g
import mplhep as hep
import awkward as ak
import edm4hep_to_ntuple as nt
import matplotlib.pyplot as plt
from omegaconf import DictConfig

hep.style.use(hep.styles.CMS)

c = 29979245800 # mm/s
tau_lifetime = 2.903e-13 # s
tau_mass = 1.77 # GeV


def plot(values, title, output_path, xlim):
    fig, ax = plt.subplots()
    rms = np.sqrt(ak.sum(values**2) * (1/(len(values))))
    bins = np.linspace(-1*xlim, xlim, num=101)
    hist, bin_edges = np.histogram(values, bins=bins)
    hep.histplot(hist, bin_edges)
    plt.xlabel("mm", loc='center')
    plt.title(title)
    textstr = f"RMS={'{:0.3e}'.format(rms)}"
    props = {"boxstyle": "round", "facecolor":'white', "alpha": 0.5}
    ax.text(0.6, 0.8, textstr, transform=ax.transAxes, fontsize=16, verticalalignment="top", bbox=props)
    plt.xlim((-1*xlim, xlim))
    plt.savefig(output_path)


def process_data(data, output_dir):
    tau_mask = data['gen_jet_tau_decaymode'] > 0
    data = data[tau_mask]
    gen_jet_tau_p4s = g.reinitialize_p4(data['gen_jet_tau_p4s'])
    gen_jet_tau_gamma = np.sqrt(1 + (gen_jet_tau_p4s.p/tau_mass)**2)
    expected_traveldistance = gen_jet_tau_gamma * c * tau_lifetime
    track_mask = abs(data['reco_cand_charge']) > 0
    tau_descendant_mask = data['reco_cand_matched_gen_energy'] / g.reinitialize_p4(data['reco_cand_p4s']).energy > 0.1
    suitable_cands_mask = track_mask * tau_descendant_mask
    gen_DV_dist = np.sqrt(
                        data['gen_jet_tau_decay_vertex_x']**2\
                      + data['gen_jet_tau_decay_vertex_y']**2\
                      + data['gen_jet_tau_decay_vertex_z']**2
    )
    reco_pca_dist_from_gen_PV = np.sqrt(
            data['reco_cand_PCA_x'][suitable_cands_mask]**2\
          + data['reco_cand_PCA_y'][suitable_cands_mask]**2\
          + data['reco_cand_PCA_z'][suitable_cands_mask]**2
    )
    plot(
        values=ak.flatten(data['reco_cand_PV_x'][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_x",
        output_path=os.path.join(output_dir, "PV_x.png"),
        xlim=0.5e-5
    )
    plot(
        values=ak.flatten(data['reco_cand_PV_y'][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_y",
        output_path=os.path.join(output_dir, "PV_y.png"),
        xlim=0.5e-7
    )
    plot(
        values=ak.flatten(data['reco_cand_PV_z'][suitable_cands_mask], axis=-1),
        title="Reco Cand PV_z",
        output_path=os.path.join(output_dir, "PV_z.png"),
        xlim=1e-2
    )


@hydra.main(config_path="../config", config_name="lifetime_verification", version_base=None)
def study_lifetime_resolution(cfg: DictConfig) -> None:
    data = g.load_all_data(cfg.input_dir, n_files=50)
    output_dir = os.path.join(cfg.input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    process_data(data, output_dir)


if __name__ == "__main__":
    study_lifetime_resolution()