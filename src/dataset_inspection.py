import awkward as ak
import hydra
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from omegaconf import DictConfig
import edm4hep_to_ntuple as nt
import mplhep as hep
from general import load_all_data
import plotting as pl

hep.style.use(hep.styles.CMS)


@hydra.main(config_path="../config", config_name="data_inspection", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_data = {}
    sample_arrays = {}
    for sample in cfg.samples_to_process:
        input_paths = glob.glob(os.path.join(cfg.samples[sample].input_dir, '*.root'))
        n_files = cfg.n_files_per_sample
        arrays = []
        for input_path in input_paths[:n_files]:
            arrays.append(nt.load_single_file_contents(input_path, cfg.tree_path, cfg.branches))
        arrays = ak.concatenate(arrays)
        sample_arrays[sample] = arrays
        # sample_data[sample] = nt.process_input_file(arrays)
    output_dir = os.path.expandvars(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plot_gen(sample_arrays['ZH_Htautau'], {}, output_dir)


########################################################################################################
########################################################################################################
###############                           PLOT GEN PART                            #####################
########################################################################################################
########################################################################################################


def plot_gen(signal_arrays, signal_data, output_dir):
    mc_particles, mc_p4 = nt.calculate_p4('MCParticles', signal_arrays)
    reco_particles, reco_p4 = nt.calculate_p4('MergedRecoParticles', signal_arrays)
    plot_Z_H_vars(mc_particles, mc_p4, output_dir)
    plot_H_pt(mc_particles, mc_p4, output_dir)
    plot_tau_vars(mc_particles, mc_p4, output_dir)
    plot_tau_vis_pt(mc_particles, mc_p4, output_dir)
    qg_jets = plot_quark_gluon_jet_multiplicity(mc_particles, mc_p4, output_dir)
    plot_genjet_vars(qg_jets, output_dir)
    plot_lepton_multiplicities(mc_particles, mc_p4, output_dir)


def is_qg_jet(jet):
    lep_jet_elem = [11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24]
    non_lep_elems = list(set(np.abs(jet)) - set(lep_jet_elem))
    return len(non_lep_elems) > 0


def plot_quark_gluon_jet_multiplicity(mc_particles, mc_p4, output_dir):
    stable_mc_p4, stable_mc_particles = nt.get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets, gen_jet_constituent_indices = nt.cluster_gen_jets(stable_mc_p4)
    events = []
    is_qg_jets = []
    for eidx in range(len(stable_mc_particles.PDG)):
        jets = []
        is_qg_jet_ = []
        for jet_idx in range(len(gen_jet_constituent_indices[eidx])):
            jet_PDG = []
            for p_idx in range(len(gen_jet_constituent_indices[eidx][jet_idx])):
                jet_PDG.append(stable_mc_particles.PDG[eidx][p_idx])
            jets.append(jet_PDG)
            is_qg_jet_.append(is_qg_jet(jet_PDG))
        events.append(jets)
        is_qg_jets.append(is_qg_jet_)
    gen_qg_jet_multiplicity = ak.sum(is_qg_jets, axis=1)
    electron_multiplicity_path = os.path.join(output_dir, "n_gen_qg_jets.png")
    pl.plot_histogram(
        entries=gen_qg_jet_multiplicity,
        output_path=electron_multiplicity_path,
        left_bin_edge=min(gen_qg_jet_multiplicity),
        right_bin_edge=max(gen_qg_jet_multiplicity),
        x_label="# q/g jets",
        title="# q/g jets"
    )
    qg_jets = gen_jets[is_qg_jets]
    return qg_jets


def plot_lepton_multiplicities(mc_particles, mc_p4, output_dir):
    stable_mask = mc_particles.generatorStatus == 1
    decaying_mask = mc_particles.generatorStatus == 2
    electron_mask = abs(mc_particles.PDG) == 11
    muon_mask = abs(mc_particles.PDG) == 13
    tau_mask = abs(mc_particles.PDG) == 15
    n_electrons = ak.num(mc_particles.PDG[stable_mask*electron_mask], axis=1)
    n_muons = ak.num(mc_particles.PDG[stable_mask*muon_mask], axis=1)
    n_taus = ak.num(mc_particles.PDG[decaying_mask*tau_mask], axis=1)
    electron_multiplicity_path = os.path.join(output_dir, "n_gen_stable_electrons.png")
    muon_multiplicity_path = os.path.join(output_dir, "n_gen_stable_muons.png")
    tau_multiplicity_path = os.path.join(output_dir, "n_gen_taus.png")
    pl.plot_histogram(
        entries=n_electrons,
        output_path=electron_multiplicity_path,
        left_bin_edge=min(n_electrons),
        right_bin_edge=max(n_electrons),
        x_label="# electrons",
        title="# electrons"
    )
    pl.plot_histogram(
        entries=n_muons,
        output_path=muon_multiplicity_path,
        left_bin_edge=min(n_muons),
        right_bin_edge=max(n_muons),
        x_label="# muons",
        title="# muons"
    )
    pl.plot_histogram(
        entries=n_taus,
        output_path=tau_multiplicity_path,
        left_bin_edge=min(n_taus),
        right_bin_edge=max(n_taus),
        x_label="# taus",
        title="# taus"
    )


def plot_H_pt(mc_particles, mc_p4, output_dir):
    initial_H_pt = ak.from_iter([
        mc_p4[i][min(ak.where(mc_particles.PDG[i] == 25)[0])].pt for i in range(len(mc_particles.PDG))
    ])
    initial_H_pt_output_path = os.path.join(output_dir, "initial_H_pt.png")
    pl.plot_histogram(
        entries=initial_H_pt,
        output_path=initial_H_pt_output_path,
        left_bin_edge=min(initial_H_pt),
        right_bin_edge=max(initial_H_pt),
        x_label=r"$p_T^H$",
        title="H pt"
    )


def plot_tau_vis_pt(mc_particles, mc_p4, output_dir):
    # Currently includes also the non-direct Higgs descantant taus. The ratio of direct and non-direct is rougly 44-1000
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)
    all_events_tau_vis_pts = []
    for e_idx in range(len(mc_particles.PDG[tau_mask])):
        daughter_mask = mc_particles.daughters_end[tau_mask][e_idx] < ak.num(mc_particles.daughters_begin[e_idx], axis=0)
        n_daughters = len(mc_particles.daughters_begin[tau_mask][e_idx][daughter_mask])
        tau_vis_pts = []
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][e_idx][daughter_mask][d_idx],
                mc_particles.daughters_end[tau_mask][e_idx][daughter_mask][d_idx],
            )
            pts = mc_p4.pt[e_idx][daughter_indices]
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            tau_vis_pts.append(ak.sum(pts[vis_particle_map], axis=0))
        all_events_tau_vis_pts.append(tau_vis_pts)
    all_events_tau_vis_pts = ak.from_iter(all_events_tau_vis_pts)
    tau_vis_pts_flat = ak.flatten(all_events_tau_vis_pts, axis=-1)
    vis_tau_pt_output_path = os.path.join(output_dir, "vis_tau_pt.png")
    pl.plot_histogram(
        entries=tau_vis_pts_flat,
        output_path=vis_tau_pt_output_path,
        left_bin_edge=min(tau_vis_pts_flat),
        right_bin_edge=max(tau_vis_pts_flat),
        x_label=r"$p_T^{vis-\tau}$",
        title="vis tau pT"
    )


def plot_genjet_vars(gen_jets, output_dir):
    gen_jets_flat = ak.flatten(gen_jets, axis=-1)

    gen_jet_pt = ak.from_iter([gjf.pt for gjf in gen_jets_flat])
    pt_output_path = os.path.join(output_dir, "gen_jet_pt.png")
    pl.plot_histogram(
        entries=gen_jet_pt,
        output_path=pt_output_path,
        left_bin_edge=min(gen_jet_pt),
        right_bin_edge=max(gen_jet_pt),
        x_label=r"$p_T^{genJet}$",
        title="genJet pT"
    )

    gen_jet_eta = ak.from_iter([gjf.eta for gjf in gen_jets_flat])
    eta_output_path = os.path.join(output_dir, "gen_jet_eta.png")
    pl.plot_histogram(
        entries=gen_jet_eta,
        output_path=eta_output_path,
        left_bin_edge=min(gen_jet_eta),
        right_bin_edge=max(gen_jet_eta),
        x_label=r"$\eta^{genJet}$",
        title="genJet eta"
    )

    gen_jet_theta = ak.from_iter([gjf.theta for gjf in gen_jets_flat])
    theta_output_path = os.path.join(output_dir, "gen_jet_theta.png")
    pl.plot_histogram(
        entries=gen_jet_theta,
        output_path=theta_output_path,
        left_bin_edge=min(gen_jet_theta),
        right_bin_edge=max(gen_jet_theta),
        x_label=r"$\theta^{genJet}$",
        title="genJet theta"
    )


def plot_Z_H_vars(mc_particles, mc_p4, output_dir):
    initial_Z = ak.from_iter([min(ak.where(mc_particles.PDG[i] == 23)[0]) for i in range(len(mc_particles.PDG))])
    initial_H = ak.from_iter([min(ak.where(mc_particles.PDG[i] == 25)[0]) for i in range(len(mc_particles.PDG))])
    Z_H_pair_mass = [(mc_p4[i][initial_Z[i]] + mc_p4[i][initial_H[i]]).mass for i in range(len(mc_particles.PDG))]
    Z_H_pair_pt = [(mc_p4[i][initial_Z[i]] + mc_p4[i][initial_H[i]]).pt for i in range(len(mc_particles.PDG))]
    mass_output_path = os.path.join(output_dir, "ZH_inv_mass.png")
    pl.plot_histogram(
        entries=Z_H_pair_mass,
        output_path=mass_output_path,
        left_bin_edge=min(Z_H_pair_mass),
        right_bin_edge=max(Z_H_pair_mass),
        x_label=r"$m_{inv}^{ZH}$",
        title="ZH pair mass"
    )
    pt_output_path = os.path.join(output_dir, "ZH_pt.png")
    pl.plot_histogram(
        entries=Z_H_pair_pt,
        output_path=pt_output_path,
        left_bin_edge=min(Z_H_pair_pt),
        right_bin_edge=max(Z_H_pair_pt),
        x_label=r"$p_{T}^{ZH}$",
        title="ZH pair pT"
    )


def plot_tau_vars(mc_particles, mc_p4, output_dir):
    tau_energy = ak.from_iter([[
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][0]].energy,
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][1]].energy
    ] for i in range(len(mc_particles.PDG))])
    tau_energy = ak.flatten(tau_energy, axis=-1)
    energy_output_path = os.path.join(output_dir, "tau_energy.png")
    pl.plot_histogram(
        entries=tau_energy,
        output_path=energy_output_path,
        left_bin_edge=min(tau_energy),
        right_bin_edge=max(tau_energy),
        x_label=r"$E^{\tau}$",
        title="tau energy"
    )

    tau_pt = ak.from_iter([[
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][0]].pt,
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][1]].pt
    ] for i in range(len(mc_particles.PDG))])
    tau_pt = ak.flatten(tau_pt, axis=-1)
    pt_output_path = os.path.join(output_dir, "tau_pt.png")
    pl.plot_histogram(
        entries=tau_pt,
        output_path=pt_output_path,
        left_bin_edge=min(tau_pt),
        right_bin_edge=max(tau_pt),
        x_label=r"$p_T^{\tau}$",
        title="tau pt"
    )

    tau_eta = ak.from_iter([[
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][0]].eta,
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][1]].eta
    ] for i in range(len(mc_particles.PDG))])
    tau_eta = ak.flatten(tau_eta, axis=-1)
    eta_output_path = os.path.join(output_dir, "tau_eta.png")
    pl.plot_histogram(
        entries=tau_eta,
        output_path=eta_output_path,
        left_bin_edge=min(tau_eta),
        right_bin_edge=max(tau_eta),
        x_label=r"$\eta^{\tau}$",
        title="tau eta"
    )

    tau_theta = ak.from_iter([[
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][0]].theta,
        mc_p4[i][ak.where(abs(mc_particles.PDG[i]) == 15)[0][1]].theta
    ] for i in range(len(mc_particles.PDG))])
    tau_theta = ak.flatten(tau_energy, axis=-1)
    theta_output_path = os.path.join(output_dir, "tau_theta.png")
    pl.plot_histogram(
        entries=tau_theta,
        output_path=theta_output_path,
        left_bin_edge=min(tau_theta),
        right_bin_edge=max(tau_theta),
        x_label=r"$\theta^{\tau}$",
        title="tau theta"
    )


if __name__ == '__main__':
    main()
