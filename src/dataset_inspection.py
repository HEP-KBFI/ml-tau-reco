import os
import glob
import hydra
import vector
import numpy as np
import mplhep as hep
import awkward as ak
import plotting as pl
import seaborn as sns
import edm4hep_to_ntuple as nt
import matplotlib.pyplot as plt
from omegaconf import DictConfig

hep.style.use(hep.styles.CMS)


@hydra.main(config_path="../config", config_name="data_inspection", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_arrays = {}
    for sample in cfg.samples_to_process:
        input_paths = glob.glob(os.path.join(cfg.samples[sample].input_dir, "*.root"))
        if cfg.n_files_per_sample == -1:
            n_files = None
        else:
            n_files = cfg.n_files_per_sample
        arrays = []
        for input_path in input_paths[:n_files]:
            arrays.append(nt.load_single_file_contents(input_path, cfg.tree_path, cfg.branches))
        arrays = ak.concatenate(arrays)
        sample_arrays[sample] = arrays
    output_dir = os.path.expandvars(cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    sig_data = sample_arrays["ZH_Htautau"]
    bkg_data = sample_arrays["QCD"]
    plot_gen(sig_data, output_dir)
    plot_reco(sig_data, output_dir)


def plot_genvistau_gentau_correlation(tau_gen_jet_p4s, gen_jets, mask, output_dir):
    vis_tau_pt = tau_gen_jet_p4s.pt.to_numpy()
    gen_jet_pt = gen_jets.pt.to_numpy()
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



def get_data_masks(gen_jets, tau_gen_jet_p4s):
    ref_var_pt_mask = tau_gen_jet_p4s.pt > 20
    ref_var_theta_mask1 = abs(np.rad2deg(tau_gen_jet_p4s.theta)) < 170
    ref_var_theta_mask2 = abs(np.rad2deg(tau_gen_jet_p4s.theta)) > 10
    mask = ref_var_pt_mask * ref_var_theta_mask1 * ref_var_theta_mask2
    return mask


########################################################################################################
########################################################################################################
###############                           PLOT GEN PART                            #####################
########################################################################################################
########################################################################################################


def plot_gen(signal_arrays, output_dir):
    mc_particles, mc_p4 = nt.calculate_p4("MCParticles", signal_arrays)
    reco_particles, reco_p4 = nt.calculate_p4("MergedRecoParticles", signal_arrays)
    plot_Z_H_vars(mc_particles, mc_p4, output_dir)
    plot_H_pt(mc_particles, mc_p4, output_dir)
    tau_mask, mask_addition = nt.get_hadronically_decaying_hard_tau_masks(mc_particles)
    plot_tau_vars(mc_particles, mc_p4, tau_mask, mask_addition, output_dir)
    plot_tau_vis_pt(mc_particles, mc_p4, tau_mask, mask_addition, output_dir)
    qg_jets = plot_quark_gluon_jet_multiplicity(mc_particles, mc_p4, output_dir)
    plot_genjet_vars(qg_jets, output_dir)
    plot_lepton_multiplicities(mc_particles, mc_p4, output_dir)
    ### 
    reco_particles, reco_p4 = nt.clean_reco_particles(reco_particles=reco_particles, reco_p4=reco_p4)
    reco_jets, reco_jet_constituent_indices = nt.cluster_jets(reco_p4)
    stable_mc_p4, stable_mc_particles = nt.get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets = nt.cluster_jets(stable_mc_p4)[0]
    reco_indices, gen_indices = nt.get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jet_constituent_indices = ak.from_iter([reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = vector.awk(ak.zip({"energy": reco_jets.t, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z}))
    gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
    gen_jets = vector.awk(ak.zip({"energy": gen_jets.t, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z}))

    tau_gen_jet_p4s_fill_value = vector.awk(
        ak.zip(
            {
                "mass": [0.0],
                "x": [0.0],
                "y": [0.0],
                "z": [0.0],
            }
        )
    )[0]
    best_combos = nt.get_all_tau_best_combinations(mc_p4, gen_jets, tau_mask, mask_addition)
    vis_tau_p4s = nt.get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4)
    tau_gen_jet_p4s = nt.get_matched_gen_tau_property(gen_jets, best_combos, vis_tau_p4s, dummy_value=tau_gen_jet_p4s_fill_value)
    tau_gen_jet_p4s = vector.awk(ak.zip({
        "mass": tau_gen_jet_p4s.tau,
        "px": tau_gen_jet_p4s.x,
        "py": tau_gen_jet_p4s.y,
        "pz": tau_gen_jet_p4s.z
    }))
    gen_jets_ = ak.flatten(gen_jets, axis=-1)
    tau_gen_jet_p4s_ = ak.flatten(tau_gen_jet_p4s, axis=-1)
    mask = get_data_masks(gen_jets_, tau_gen_jet_p4s)
    plot_genvistau_gentau_correlation(tau_gen_jet_p4s, gen_jets_, mask, output_dir)


def is_qg_jet(jet):
    lep_jet_elem = [11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24]
    non_lep_elems = list(set(np.abs(jet)) - set(lep_jet_elem))
    return len(non_lep_elems) > 0


def plot_quark_gluon_jet_multiplicity(mc_particles, mc_p4, output_dir):
    stable_mc_p4, stable_mc_particles = nt.get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets, gen_jet_constituent_indices = nt.cluster_jets(stable_mc_p4)
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
        title="# q/g jets",
    )
    qg_jets = gen_jets[is_qg_jets]
    return qg_jets


def plot_lepton_multiplicities(mc_particles, mc_p4, output_dir):
    stable_mask = mc_particles.generatorStatus == 1
    decaying_mask = mc_particles.generatorStatus == 2
    electron_mask = abs(mc_particles.PDG) == 11
    muon_mask = abs(mc_particles.PDG) == 13
    tau_mask = abs(mc_particles.PDG) == 15
    n_electrons = ak.num(mc_particles.PDG[stable_mask * electron_mask], axis=1)
    n_muons = ak.num(mc_particles.PDG[stable_mask * muon_mask], axis=1)
    n_taus = ak.num(mc_particles.PDG[decaying_mask * tau_mask], axis=1)
    electron_multiplicity_path = os.path.join(output_dir, "n_gen_stable_electrons.png")
    muon_multiplicity_path = os.path.join(output_dir, "n_gen_stable_muons.png")
    tau_multiplicity_path = os.path.join(output_dir, "n_gen_taus.png")
    pl.plot_histogram(
        entries=n_electrons,
        output_path=electron_multiplicity_path,
        left_bin_edge=min(n_electrons),
        right_bin_edge=max(n_electrons),
        x_label="# electrons",
        title="# electrons",
    )
    pl.plot_histogram(
        entries=n_muons,
        output_path=muon_multiplicity_path,
        left_bin_edge=min(n_muons),
        right_bin_edge=max(n_muons),
        x_label="# muons",
        title="# muons",
    )
    pl.plot_histogram(
        entries=n_taus,
        output_path=tau_multiplicity_path,
        left_bin_edge=min(n_taus),
        right_bin_edge=max(n_taus),
        x_label="# taus",
        title="# taus",
    )


def plot_H_pt(mc_particles, mc_p4, output_dir):
    initial_H_pt = ak.from_iter(
        [mc_p4[i][min(ak.where(mc_particles.PDG[i] == 25)[0])].pt for i in range(len(mc_particles.PDG))]
    )
    initial_H_pt_output_path = os.path.join(output_dir, "initial_H_pt.png")
    pl.plot_histogram(
        entries=initial_H_pt,
        output_path=initial_H_pt_output_path,
        left_bin_edge=min(initial_H_pt),
        right_bin_edge=max(initial_H_pt),
        x_label=r"$p_T^H$",
        title="H pt",
    )


def plot_tau_vis_pt(mc_particles, mc_p4, tau_mask, mask_addition, output_dir):
    all_events_tau_vis_pts = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        tau_vis_pts = []
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            p4s = mc_p4[e_idx][daughter_indices]
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            p4s = p4s[vis_particle_map]
            summed_vis_tau = vector.awk(
                ak.zip(
                    {
                        "px": [ak.sum(p4s.x, axis=-1)],
                        "py": [ak.sum(p4s.y, axis=-1)],
                        "pz": [ak.sum(p4s.z, axis=-1)],
                        "mass": [ak.sum(p4s.tau, axis=-1)],
                    }
                )
            ).pt[0]
            tau_vis_pts.append(summed_vis_tau)
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
        title="vis tau pT",
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
        title="genJet pT",
    )

    gen_jet_eta = ak.from_iter([gjf.eta for gjf in gen_jets_flat])
    eta_output_path = os.path.join(output_dir, "gen_jet_eta.png")
    pl.plot_histogram(
        entries=gen_jet_eta,
        output_path=eta_output_path,
        left_bin_edge=min(gen_jet_eta),
        right_bin_edge=max(gen_jet_eta),
        x_label=r"$\eta^{genJet}$",
        title="genJet eta",
    )

    gen_jet_theta = np.rad2deg([gjf.theta for gjf in gen_jets_flat])
    theta_output_path = os.path.join(output_dir, "gen_jet_theta.png")
    pl.plot_histogram(
        entries=gen_jet_theta,
        output_path=theta_output_path,
        left_bin_edge=min(gen_jet_theta),
        right_bin_edge=max(gen_jet_theta),
        x_label=r"$\theta^{genJet}$",
        title="genJet theta",
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
        title="ZH pair mass",
    )
    pt_output_path = os.path.join(output_dir, "ZH_pt.png")
    pl.plot_histogram(
        entries=Z_H_pair_pt,
        output_path=pt_output_path,
        left_bin_edge=min(Z_H_pair_pt),
        right_bin_edge=max(Z_H_pair_pt),
        x_label=r"$p_{T}^{ZH}$",
        title="ZH pair pT",
    )


def plot_tau_vars(mc_particles, mc_p4, tau_mask, mask_addition, output_dir):
    tau_p4 = mc_p4[tau_mask][mask_addition]
    tau_energy = tau_p4.energy
    tau_energy = ak.flatten(tau_energy, axis=-1)
    energy_output_path = os.path.join(output_dir, "tau_energy.png")
    pl.plot_histogram(
        entries=tau_energy,
        output_path=energy_output_path,
        left_bin_edge=min(tau_energy),
        right_bin_edge=max(tau_energy),
        x_label=r"$E^{\tau}$",
        title="tau energy",
    )

    tau_pt = tau_p4.pt
    tau_pt = ak.flatten(tau_pt, axis=-1)
    pt_output_path = os.path.join(output_dir, "tau_pt.png")
    pl.plot_histogram(
        entries=tau_pt,
        output_path=pt_output_path,
        left_bin_edge=min(tau_pt),
        right_bin_edge=max(tau_pt),
        x_label=r"$p_T^{\tau}$",
        title="tau pt",
    )

    tau_eta = tau_p4.eta
    tau_eta = ak.flatten(tau_eta, axis=-1)
    eta_output_path = os.path.join(output_dir, "tau_eta.png")
    pl.plot_histogram(
        entries=tau_eta,
        output_path=eta_output_path,
        left_bin_edge=min(tau_eta),
        right_bin_edge=max(tau_eta),
        x_label=r"$\eta^{\tau}$",
        title="tau eta",
    )

    tau_theta = tau_p4.eta
    tau_theta = ak.flatten(tau_energy, axis=-1)
    theta_output_path = os.path.join(output_dir, "tau_theta.png")
    pl.plot_histogram(
        entries=tau_theta,
        output_path=theta_output_path,
        left_bin_edge=min(tau_theta),
        right_bin_edge=max(tau_theta),
        x_label=r"$\theta^{\tau}$",
        title="tau theta",
    )


########################################################################################################
########################################################################################################
###############                           PLOT RECO PART                           #####################
########################################################################################################
########################################################################################################


def plot_reco(signal_arrays, output_dir):
    mc_particles, mc_p4 = nt.calculate_p4(p_type="MCParticles", arrs=signal_arrays)
    reco_particles, reco_p4 = nt.calculate_p4(p_type="MergedRecoParticles", arrs=signal_arrays)
    tau_mask, mask_addition = nt.get_hadronically_decaying_hard_tau_masks(mc_particles)
    reco_jets, reco_jet_constituent_indices = nt.cluster_jets(reco_p4)
    stable_mc_p4, stable_mc_particles = nt.get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets = nt.cluster_jets(stable_mc_p4)[0]
    reco_indices, gen_indices = nt.get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jets = ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = vector.awk(ak.zip({"energy": reco_jets.t, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z}))
    gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
    gen_jets = vector.awk(ak.zip({"energy": gen_jets.t, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z}))
    plot_jet_response(reco_jets, gen_jets, output_dir)
    plot_met_response(stable_mc_p4, reco_p4, output_dir)
    plot_particle_multiplicity_around_gen_vis_tau(
        mc_particles, mc_p4, reco_particles, reco_p4, tau_mask, mask_addition, output_dir, cone_radius=0.4
    )


def plot_jet_response(reco_jets, gen_jets, output_dir):
    flat_reco_jets = ak.flatten(reco_jets.pt, axis=1).to_numpy()
    flat_gen_jets = ak.flatten(gen_jets.pt, axis=1).to_numpy()
    jet_response = flat_reco_jets / flat_gen_jets
    jet_response_output_path = os.path.join(output_dir, "jet_response.png")
    pl.plot_histogram(
        entries=jet_response,
        output_path=jet_response_output_path,
        left_bin_edge=min(jet_response),
        right_bin_edge=max(jet_response),
        x_label=r"$p_T^{recoJet} / p_T^{genJet}$",
        title="jet response",
    )


def plot_met_response(stable_mc_p4, reco_p4, output_dir):
    event_gen_p4 = vector.awk(
        ak.zip(
            {
                "px": ak.sum(stable_mc_p4.px, axis=-1),
                "py": ak.sum(stable_mc_p4.py, axis=-1),
                "pz": ak.sum(stable_mc_p4.pz, axis=-1),
                "energy": ak.sum(stable_mc_p4.energy, axis=-1),
            }
        )
    )
    event_reco_p4 = vector.awk(
        ak.zip(
            {
                "px": ak.sum(reco_p4.px, axis=-1),
                "py": ak.sum(reco_p4.py, axis=-1),
                "pz": ak.sum(reco_p4.pz, axis=-1),
                "energy": ak.sum(reco_p4.energy, axis=-1),
            }
        )
    )
    reco_met = event_reco_p4.et
    gen_met = event_gen_p4.et
    met_response = reco_met / gen_met
    met_response2 = reco_met - gen_met
    met_response_output_path = os.path.join(output_dir, "met_response.png")
    met_response_output_path2 = os.path.join(output_dir, "met_response2.png")
    pl.plot_histogram(
        entries=met_response,
        output_path=met_response_output_path,
        left_bin_edge=min(met_response),
        right_bin_edge=max(met_response),
        x_label=r"$p_T^{reco MET} / p_T^{gen MET}$",
        title="MET response",
    )
    pl.plot_histogram(
        entries=met_response2,
        output_path=met_response_output_path2,
        left_bin_edge=min(met_response2),
        right_bin_edge=max(met_response2),
        x_label=r"$p_T^{reco MET} - p_T^{gen MET}$",
        title="MET response",
    )


def plot_particle_multiplicity_around_gen_vis_tau(
    mc_particles, mc_p4, reco_particles, reco_p4, tau_mask, mask_addition, output_dir, cone_radius=0.4
):

    stable_mc_p4, stable_mc_particles = nt.get_stable_mc_particles(mc_particles, mc_p4)
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)

    all_cone_gen_particle_pdgs = []
    all_cone_reco_particle_pdgs = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        event_cone_reco_particle_pdgs = []
        event_cone_gen_particle_pdgs = []
        event_reco_particle_PDGs = reco_particles["type"][e_idx]
        event_reco_p4s = reco_p4[e_idx]
        event_gen_particle_PDGs = stable_mc_particles.PDG[e_idx]
        event_gen_p4s = stable_mc_p4[e_idx]
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            p4s = mc_p4[e_idx][daughter_indices]
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            p4s = p4s[vis_particle_map]
            vis_tau_p4 = vector.awk(
                ak.zip(
                    {
                        "px": [ak.sum(p4s.x, axis=-1)],
                        "py": [ak.sum(p4s.y, axis=-1)],
                        "pz": [ak.sum(p4s.z, axis=-1)],
                        "mass": [ak.sum(p4s.tau, axis=-1)],
                    }
                )
            )[0]
            reco_in_cone = find_dr_between_vis_tau_and_cands(vis_tau_p4, cand_p4s=event_reco_p4s, cone_radius=cone_radius)
            cone_reco_particle_pdgs = event_reco_particle_PDGs[reco_in_cone]
            gen_in_cone = find_dr_between_vis_tau_and_cands(vis_tau_p4, cand_p4s=event_gen_p4s, cone_radius=cone_radius)
            cone_gen_particle_pdgs = event_gen_particle_PDGs[gen_in_cone]
            event_cone_reco_particle_pdgs.append(cone_reco_particle_pdgs)
            event_cone_gen_particle_pdgs.append(cone_gen_particle_pdgs)
        all_cone_gen_particle_pdgs.append(event_cone_gen_particle_pdgs)
        all_cone_reco_particle_pdgs.append(event_cone_reco_particle_pdgs)
    all_cone_gen_particle_pdgs = ak.from_iter(all_cone_gen_particle_pdgs)
    all_cone_reco_particle_pdgs = ak.from_iter(all_cone_reco_particle_pdgs)
    flat_cone_reco_particle_pdgs = ak.flatten(all_cone_reco_particle_pdgs, axis=1)
    flat_cone_gen_particle_pdgs = ak.flatten(all_cone_gen_particle_pdgs, axis=1)
    gen_particle_pdg_set = set(ak.flatten(abs(flat_cone_gen_particle_pdgs), axis=-1))
    reco_particle_pdg_set = set(ak.flatten(abs(flat_cone_reco_particle_pdgs), axis=-1))
    particle_info = {
        11: [r"$e^{\pm}$", "electron"],
        13: [r"$\mu^{\pm}$", "muon"],
        22: [r"$\gamma$", "gamma"],
        111: [r"$\pi^0$", "piZero"],
        211: [r"$\pi^{\pm}$", "piPlusMinus"],
        2112: ["n", "neutron"],
        2212: ["p", "proton"],
        321: [r"$K^{\pm}$", "kaonPlusMinus"],
        130: [r"$K^0_L$", "K_0_long"],
        310: [r"$K^0_S$", "K_0_short"],
        311: [r"$K^0$", "K_0"],
    }
    for reco_pdg in list(reco_particle_pdg_set)[2:]:
        particle_multiplicities = ak.num(flat_cone_reco_particle_pdgs[abs(flat_cone_reco_particle_pdgs) == reco_pdg], axis=1)
        particle_info_entry = particle_info[reco_pdg]
        plot_particle_multiplicity(particle_multiplicities, particle_info_entry, output_dir, particles_origin="reco")
    for gen_pdg in gen_particle_pdg_set:
        particle_multiplicities = ak.num(flat_cone_gen_particle_pdgs[abs(flat_cone_gen_particle_pdgs) == gen_pdg], axis=1)
        particle_info_entry = particle_info[gen_pdg]
        plot_particle_multiplicity(particle_multiplicities, particle_info_entry, output_dir, particles_origin="gen")
    reco_multi_path = os.path.join(output_dir, "reco_full_cone_multiplicity.png")
    plot_full_entry_multiplicity_matrix(flat_cone_reco_particle_pdgs, reco_multi_path, particle_info)
    gen_multi_path = os.path.join(output_dir, "gen_full_cone_multiplicity.png")
    plot_full_entry_multiplicity_matrix(flat_cone_gen_particle_pdgs, gen_multi_path, particle_info)


def plot_full_entry_multiplicity_matrix(flat_cone_pdgs, output_path, particle_info, n_tau_jets=20, figsize=(16, 9)):
    pdg_matrix = []
    for cone_pdgs in flat_cone_pdgs:
        pdg_matrix.append(np.array([abs(cone_pdgs).to_list().count(key_) for key_ in particle_info.keys()]))
    pdg_matrix = np.array(pdg_matrix[:n_tau_jets])
    pdg_matrix = pdg_matrix.astype("float")
    pdg_matrix[pdg_matrix == 0] = "nan"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pdg_matrix.T, linewidth=0.5, cmap="viridis", annot=True)
    ax.set_yticklabels([particle_info[pdg][0] for pdg in particle_info.keys()])
    ax.set(xticklabels=[])
    ax.set_xlabel("Taus")
    plt.savefig(output_path, bbox_inches="tight")


def plot_particle_multiplicity(particle_multiplicities, particle_info_entry, output_dir, particles_origin="gen"):
    output_path = os.path.join(output_dir, f"{particles_origin}_{particle_info_entry[1]}_multiplicity.png")
    pl.plot_histogram(
        entries=particle_multiplicities,
        output_path=output_path,
        left_bin_edge=min(particle_multiplicities),
        right_bin_edge=max(particle_multiplicities),
        x_label=particle_info_entry[0],
        title=f"# {particle_info_entry[0]}",
        integer_bins=True,
    )


# @numba.njit
def find_dr_between_vis_tau_and_cands(vis_tau_p4, cand_p4s, cone_radius):
    drs = []
    for cand_p4 in cand_p4s:
        drs.append(nt.deltar(vis_tau_p4.eta, vis_tau_p4.phi, cand_p4.eta, cand_p4.phi) < cone_radius)
    return drs


if __name__ == "__main__":
    main()
