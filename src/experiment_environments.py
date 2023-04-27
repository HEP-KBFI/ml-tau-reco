import os
import glob
import numba
import uproot
import hydra
import vector
import numpy as np
import general as g
import awkward as ak
import edm4hep_to_ntuple as nt
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use(hep.styles.CMS)


def remove_leptonic_jets(gen_jets, gen_jet_constituent_indices, stable_mc_particles):
    gen_num_ptcls_per_jet = ak.num(gen_jet_constituent_indices, axis=-1)
    gen_jet_pdgs = nt.get_jet_constituent_property(
        stable_mc_particles.PDG, gen_jet_constituent_indices, gen_num_ptcls_per_jet
    )
    mask = []
    for gj_pdg in gen_jet_pdgs:
        sub_mask = []
        for gjp in gj_pdg:
            if (15 in np.abs(gjp)) or (13 in np.abs(gjp)):
                sub_mask.append(False)
            else:
                sub_mask.append(True)
        mask.append(sub_mask)
    mask = ak.Array(mask)
    return gen_jets[mask], gen_jet_constituent_indices[mask]


def construct_non_tau_remnants_mask(mc_particles):
    tau_mask, mask_addition = nt.get_hadronically_decaying_hard_tau_masks(mc_particles)
    gen_tau_daughters = nt.find_tau_daughters_all_generations(mc_particles, tau_mask, mask_addition)
    mc_particles_shape = ak.ones_like(mc_particles.PDG, dtype=bool)
    masking_array = []
    for i, mcs in enumerate(mc_particles_shape):
        np.asarray(mcs)[gen_tau_daughters[i]] = False
        masking_array.append(mcs)
    masking_array = ak.Array(masking_array)
    return masking_array


@numba.njit
def get_particle_drs(tau_eta, tau_phi, particle_etas, particle_phis):
    particle_drs = []
    for i in range(len(particle_etas)):
        particle_drs.append(nt.deltar(tau_eta, tau_phi, particle_etas[i], particle_phis[i]))
    return particle_drs


@numba.njit
def get_all_event_drs(tau_eta, tau_phi, particle_etas, particle_phis):
    all_event_drs = []
    for i in range(len(tau_eta)):
        te = tau_eta[i]
        tp = tau_phi[i]
        pe = particle_etas[i]
        pp = particle_phis[i]
        particles_dr_from_tau = get_particle_drs(te, tp, pe, pp)
        all_event_drs.append(particles_dr_from_tau)
    return all_event_drs


def get_isolation_particle_mask(flattened_per_tau_particle_isolation_p4, flattened_taus_p4s, iso_cone_radius=0.5):
    tau_eta = flattened_taus_p4s.eta
    tau_phi = flattened_taus_p4s.phi
    particle_etas = flattened_per_tau_particle_isolation_p4.eta
    particle_phis = flattened_per_tau_particle_isolation_p4.phi
    all_event_drs = ak.Array(get_all_event_drs(tau_eta, tau_phi, particle_etas, particle_phis))
    inside_cone = all_event_drs <= iso_cone_radius
    return inside_cone


def compute_tau_isolation_and_n_constituents(mc_particles, mc_p4):
    tau_mask, mask_addition = nt.get_hadronically_decaying_hard_tau_masks(mc_particles)
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (abs(mc_particles["PDG"]) != 16)
    stability_mask = stable_pythia_mask * neutrino_mask
    non_tau_remnants_mask = construct_non_tau_remnants_mask(mc_particles)
    summing_mask = (mc_particles.PDG == 22) + (abs(mc_particles.charge) > 0)
    partial_mask = stability_mask * non_tau_remnants_mask * summing_mask
    isolation_particles_p4 = mc_p4[partial_mask]
    vis_tau_p4s = nt.get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4)
    # Duplicate all event isolation_particle_p4 for each tau in that given event
    event_isolation_particle_p4 = ak.from_iter(
        [[isolation_particles_p4[j] for i in range(len(vis_tau_p4s[j]))] for j in range(len(vis_tau_p4s))]
    )
    flattened_per_tau_particle_isolation_p4 = g.reinitialize_p4(ak.flatten(event_isolation_particle_p4, axis=1))
    flattened_taus_p4s = g.reinitialize_p4(ak.flatten(vis_tau_p4s, axis=1))
    iso_particle_mask = get_isolation_particle_mask(flattened_per_tau_particle_isolation_p4, flattened_taus_p4s)
    isolations = ak.sum(flattened_per_tau_particle_isolation_p4[iso_particle_mask].pt, axis=-1)
    n_constituents = ak.num(flattened_per_tau_particle_isolation_p4[iso_particle_mask].pt, axis=-1)
    return isolations, n_constituents


def plot_isolations(isolations, output_dir):
    output_path = os.path.join(output_dir, "isolations.pdf")
    bin_edges = np.linspace(-0.5, 50.5, 50)
    hatches = {h: s for h, s in zip(isolations.keys(), ["//", "\\\\"])}
    colors = {c: s for c, s in zip(isolations.keys(), ["red", "blue"])}
    for sample_name, iso in isolations.items():
        sample_H = np.histogram(iso, bins=bin_edges)[0]
        sample_H = sample_H / sum(sample_H)
        hep.histplot(sample_H, bin_edges, label=sample_name, hatch=hatches[sample_name], color=colors[sample_name])
    plt.legend()
    plt.ylabel("Relative yield / bin", fontdict={"size": 25})
    plt.xlabel(r"$\mathcal{I}_{\tau}$", fontdict={"size": 25})
    plt.savefig(output_path)
    plt.close("all")


def plot_n_constituents(n_constituents, output_dir):
    output_path = os.path.join(output_dir, "n_constituents.pdf")
    bin_edges = np.linspace(0, 30, 30)
    hatches = {h: s for h, s in zip(n_constituents.keys(), ["//", "\\\\"])}
    colors = {c: s for c, s in zip(n_constituents.keys(), ["red", "blue"])}
    for sample_name, nc in n_constituents.items():
        sample_H = np.histogram(np.clip(nc, bin_edges[0], bin_edges[-1]), bins=bin_edges)[0]
        sample_H = sample_H / sum(sample_H)
        hep.histplot(sample_H, bin_edges, label=sample_name, hatch=hatches[sample_name], color=colors[sample_name])
    plt.legend()
    plt.ylabel("Relative yield / bin", fontdict={"size": 25})
    plt.xlabel("Number of particles in isolation cone", fontdict={"size": 25})
    plt.savefig(output_path)
    plt.close("all")


@hydra.main(config_path="../config", config_name="environments", version_base=None)
def compare_different_environments(cfg: DictConfig) -> None:
    isolations = {}
    n_constituents = {}
    for sample_name, path in cfg.samples.items():
        if sample_name == "ee":
            n_files = int(np.ceil(cfg.max_events / 100))
            paths = glob.glob(os.path.join(cfg.samples.ee, "*.root"))[:n_files]
            arrays = []
            for path in paths:
                arrays.append(nt.load_single_file_contents(path))
            arrays = ak.concatenate(arrays)[: cfg.max_events]
            mc_particles, mc_p4 = nt.calculate_p4(p_type="MCParticles", arrs=arrays)
        else:
            mc_particles, mc_p4 = load_pp_file_contents(path, cfg.max_events, branch="GenNoPU")
        sample_isolations, sample_n_constituents = compute_tau_isolation_and_n_constituents(mc_particles, mc_p4)
        isolations[sample_name] = sample_isolations
        n_constituents[sample_name] = sample_n_constituents
    ##
    os.makedirs(cfg.output_dir, exist_ok=True)
    plot_isolations(isolations, cfg.output_dir)
    plot_n_constituents(n_constituents, cfg.output_dir)


def load_pp_file_contents(pp_path, max_events, branch="GenNoPU"):
    branches_of_interest = ["Px", "Py", "Pz", "Mass", "PID", "Charge", "M1", "M2", "D1", "D2", "Status"]
    branch_names = [f"{branch}.{var}" for var in branches_of_interest]
    with uproot.open(pp_path) as in_file:
        tree = in_file[f"Delphes/{branch}"]
        arrays = tree.arrays(branch_names)
    particles = ak.Record({k.replace(f"{branch}.", "").lower(): arrays[k] for k in arrays.fields})
    mc_particles = ak.Record(
        {
            "daughters_begin": particles.d1[:max_events],
            "daughters_end": particles.d2[:max_events],
            "parents_begin": particles.m1[:max_events],
            "parents_end": particles.m2[:max_events],
            "PDG": particles.pid[:max_events],
            "charge": particles.charge[:max_events],
            "generatorStatus": particles.status[:max_events],
        }
    )
    mc_p4 = vector.awk(
        ak.zip(
            {
                "mass": particles.mass,
                "x": particles.px,
                "y": particles.py,
                "z": particles.pz,
            }
        )
    )[:max_events]
    return mc_particles, mc_p4


if __name__ == "__main__":
    compare_different_environments()
