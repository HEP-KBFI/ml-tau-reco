"""Script for ntupelizing the EDM4HEP dataformat to ML friendly format. To run the script in our singularity image:
           ./scripts/run-env.sh  src/edm4hep_to_ntuple.py [args]
Call with 'python3'
"""

import os
import time
import glob
import numba
import uproot
import hydra
import vector
import fastjet
import numpy as np
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig


def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(ak.Record(data), output_path)


def load_single_file_contents(path: str, tree_path: str, branches: list = None) -> ak.Array:
    with uproot.open(path) as in_file:
        tree = in_file[tree_path]
        arrays = tree.arrays(branches)
        idx0 = "RecoMCTruthLink#0/RecoMCTruthLink#0.index"
        idx1 = "RecoMCTruthLink#1/RecoMCTruthLink#1.index"
        idx_recoparticle = tree.arrays(idx0)[idx0]
        idx_mc_particlesarticle = tree.arrays(idx1)[idx1]
        # index in the MergedRecoParticles collection
        arrays["idx_reco"] = idx_recoparticle
        # index in the MCParticles collection
        arrays["idx_mc"] = idx_mc_particlesarticle
    return arrays


def calculate_p4(p_type: str, arrs: ak.Array):
    particles = arrs[p_type]
    particles = ak.Record({k.replace(f"{p_type}.", ""): particles[k] for k in particles.fields})
    particle_p4 = vector.awk(
        ak.zip(
            {
                "mass": particles["mass"],
                "x": particles["momentum.x"],
                "y": particles["momentum.y"],
                "z": particles["momentum.z"],
            }
        )
    )
    return particles, particle_p4


def get_stable_mc_particles(mc_particles, mc_p4):
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    stable_mc_p4 = mc_p4[stable_pythia_mask]
    return stable_mc_p4


def cluster_jets(particles_p4):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    # This workaround here is also only temporary due to incorrect object
    # initialization, see: https://github.com/scikit-hep/fastjet/issues/174
    constituent_index = []
    jets = []
    for iev in range(len(particles_p4.pt)):
        cluster = fastjet.ClusterSequence(particles_p4[iev], jetdef)
        jets.append(vector.awk(cluster.inclusive_jets(min_pt=2.0)))
        ci = cluster.constituent_index(min_pt=2.0)
        constituent_index.append(ci)
    constituent_index = ak.from_iter(constituent_index)
    jets = ak.from_iter(jets)
    jets = vector.awk(ak.zip({"mass": jets["t"], "x": jets["x"], "y": jets["y"], "z": jets["z"]}))
    return jets, constituent_index


###############################################################################
###############################################################################
#####                 TAU DECAYMODE CALCULATION                         #######
###############################################################################
###############################################################################


def get_decaymode(pdg_ids):
    """Tau decaymodes are the following:
    decay_mode_mapping = {
        0: 'OneProng0PiZero',
        1: 'OneProng1PiZero',
        2: 'OneProng2PiZero',
        3: 'OneProng3PiZero',
        4: 'OneProngNPiZero',
        5: 'TwoProng0PiZero',
        6: 'TwoProng1PiZero',
        7: 'TwoProng2PiZero',
        8: 'TwoProng3PiZero',
        9: 'TwoProngNPiZero',
        10: 'ThreeProng0PiZero',
        11: 'ThreeProng1PiZero',
        12: 'ThreeProng2PiZero',
        13: 'ThreeProng3PiZero',
        14: 'ThreeProngNPiZero',
        15: 'RareDecayMode'
    }
    0: [0, 5, 10]
    1: [1, 6, 11]
    2: [2, 3, 4, 7, 8, 9, 12, 13, 14, 15]
    """
    pdg_ids = np.abs(np.array(pdg_ids))
    unique, counts = np.unique(pdg_ids, return_counts=True)
    p_counts = {i: 0 for i in [16, 111, 211, 13, 14, 12, 11, 22]}
    p_counts.update(dict(zip(unique, counts)))
    if check_rare_decaymode(pdg_ids):
        return 15
    if np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 0:
        return 0
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 1:
        return 1
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 2:
        return 2
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 3:
        return 3
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] > 3:
        return 4
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 0:
        return 5
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 1:
        return 6
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 2:
        return 7
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 3:
        return 8
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] > 3:
        return 9
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 0:
        return 10
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 1:
        return 11
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 2:
        return 12
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 3:
        return 13
    elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] > 3:
        return 14
    else:
        return 15


def check_rare_decaymode(pdg_ids):
    """The common particles in order are tau-neutrino, pi0, pi+, mu,
    mu-neutrino, electron-neutrino, electron, photon"""
    common_particles = [16, 111, 211, 13, 14, 12, 11, 22]
    return sum(np.in1d(pdg_ids, common_particles)) != len(pdg_ids)


def get_event_decaymodes(j, tau_mask, mc_particles):
    # Temporary fix because daughter indices is bigger than the nr of mc_particles
    daughter_mask = mc_particles.daughters_begin[tau_mask][j] < ak.num(mc_particles.daughters_begin[tau_mask][j], axis=0)
    return ak.from_iter(
        [
            get_decaymode(
                mc_particles.PDG[j][
                    range(mc_particles.daughters_begin[tau_mask][j][i], mc_particles.daughters_end[tau_mask][j][i])
                ]
            )
            for i in range(len(mc_particles.daughters_begin[tau_mask][j][daughter_mask]))
        ]
    )


def get_all_tau_decaymodes(mc_particles, tau_mask):
    return ak.from_iter([get_event_decaymodes(j, tau_mask, mc_particles) for j in range(len(mc_particles.PDG[tau_mask]))])


###############################################################################
###############################################################################
###############            TAU VISIBLE ENERGY                    ##############
###############################################################################
###############################################################################


def get_tau_vis_energy(mc_p4, mc_particles, tau_mask, i, j):
    range_ = range(mc_particles.daughters_begin[tau_mask][j][i], mc_particles.daughters_end[tau_mask][j][i])
    PDG_ids = np.abs(mc_particles.PDG[j][range_])
    energies = mc_p4.energy[j][range_]
    vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
    return ak.sum(energies[vis_particle_map])


def calculate_tau_visible_energies(j, tau_mask, mc_particles, mc_p4):
    # Temporary fix because daughter indices is bigger than the nr of mc_particles
    daughter_mask = mc_particles.daughters_begin[tau_mask][j] < ak.num(mc_particles.daughters_begin[tau_mask][j], axis=0)
    return ak.from_iter(
        [
            get_tau_vis_energy(mc_p4, mc_particles, tau_mask, i, j)
            for i in range(len(mc_particles.daughters_begin[tau_mask][j][daughter_mask]))
        ]
    )


def get_tau_energies(tau_mask, mc_particles, mc_p4):
    return ak.from_iter(
        [calculate_tau_visible_energies(j, tau_mask, mc_particles, mc_p4) for j in range(len(mc_particles.PDG[tau_mask]))]
    )


###############################################################################
###############################################################################
###############################################################################
###############################################################################


###############################################################################
###############################################################################
###############           MATCH TAUS WITH GEN JETS               ##############
###############################################################################
###############################################################################


def get_all_tau_best_combinations(mc_p4, gen_jets, tau_mask):
    mc_tau_vec = ak.zip(
        {
            "pt": mc_p4[tau_mask].pt,
            "eta": mc_p4[tau_mask].eta,
            "phi": mc_p4[tau_mask].phi,
            "energy": mc_p4[tau_mask].energy,
        }
    )
    gen_jets_p4 = ak.zip(
        {
            "pt": gen_jets.pt,
            "eta": gen_jets.eta,
            "phi": gen_jets.phi,
            "energy": gen_jets.energy,
        }
    )
    tau_indices, gen_indices = match_jets(mc_tau_vec, gen_jets_p4, 999.9)
    pairs = []
    for tau_idx, gen_idx in zip(tau_indices, gen_indices):
        pair = []
        for i in range(len(tau_idx)):
            pair.append([tau_idx[i], gen_idx[i]])
        pairs.append(pair)
    return ak.Array(pairs)


###############################################################################
###############################################################################
###############################################################################
###############################################################################


@numba.njit
def deltar(eta1, phi1, eta2, phi2):
    deta = np.abs(eta1 - eta2)
    dphi = deltaphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


@numba.njit
def deltaphi(phi1, phi2):
    return np.fmod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi


@numba.njit
def match_jets(jets1, jets2, deltaR_cut):
    iev = len(jets1)
    jet_inds_1_ev = []
    jet_inds_2_ev = []
    for ev in range(iev):
        j1 = jets1[ev]
        j2 = jets2[ev]

        jet_inds_1 = []
        jet_inds_2 = []
        for ij1 in range(len(j1)):
            drs = np.zeros(len(j2), dtype=np.float64)
            for ij2 in range(len(j2)):
                eta1 = j1.eta[ij1]
                eta2 = j2.eta[ij2]
                phi1 = j1.phi[ij1]
                phi2 = j2.phi[ij2]

                # Workaround for https://github.com/scikit-hep/vector/issues/303
                # dr = j1[ij1].deltaR(j2[ij2])
                dr = deltar(eta1, phi1, eta2, phi2)
                drs[ij2] = dr
            if len(drs) > 0:
                min_idx_dr = np.argmin(drs)
                if drs[min_idx_dr] < deltaR_cut:
                    jet_inds_1.append(ij1)
                    jet_inds_2.append(min_idx_dr)
        jet_inds_1_ev.append(jet_inds_1)
        jet_inds_2_ev.append(jet_inds_2)
    return jet_inds_1_ev, jet_inds_2_ev


def get_jet_constituent_p4s(reco_p4, constituent_idx, num_ptcls_per_jet):
    reco_p4_flat = reco_p4[ak.flatten(constituent_idx, axis=-1)]
    ret = ak.from_iter([ak.unflatten(reco_p4_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])
    return vector.awk(ak.zip({"x": ret.x, "y": ret.y, "z": ret.z, "mass": ret.tau}))


def get_jet_constituent_pdgs(reco_particles, constituent_idx, num_ptcls_per_jet):
    reco_pdg_flat = reco_particles["type"][ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [ak.unflatten(reco_pdg_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))]
    )


def get_jet_constituent_charges(reco_particles, constituent_idx, num_ptcls_per_jet):
    reco_charge_flat = reco_particles["charge"][ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [ak.unflatten(reco_charge_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))]
    )


def to_vector(jet):
    return vector.awk(
        ak.zip(
            {
                "pt": jet.pt,
                "eta": jet.eta,
                "phi": jet.phi,
                "energy": jet.energy,
            }
        )
    )


def to_fourvec(jet):
    return vector.awk(
        ak.zip(
            {
                "mass": jet.tau,
                "x": jet.x,
                "y": jet.y,
                "z": jet.z,
            }
        )
    )


def get_matched_gen_jet_p4(reco_jets, gen_jets):
    reco_jets = to_vector(reco_jets)
    gen_jets = to_vector(gen_jets)
    reco_indices, gen_indices = match_jets(reco_jets, gen_jets, deltaR_cut=0.3)
    return reco_indices, gen_indices
    # Muuta et reco jet arv ja gen jet arv oleks sama
    # return ak.from_iter([gen_jets[i][gen_idx] for i, gen_idx in enumerate(gen_indices)])


def get_matched_gen_tau_decaymode(gen_jets, best_combos, tau_decaymodes):
    gen_jet_full_info_array = []
    for event_id in range(len(gen_jets)):
        mapping = {i[1]: i[0] for i in best_combos[event_id]}
        gen_jet_info_array = []
        for i, gen_jet in enumerate(gen_jets[event_id]):
            if i in best_combos[event_id][:, 1]:
                if len(tau_decaymodes[event_id]) == 0:
                    value = -1
                else:
                    value = tau_decaymodes[event_id][mapping[i]]
                gen_jet_info_array.append(value)
            else:
                gen_jet_info_array.append(-1)
        gen_jet_full_info_array.append(gen_jet_info_array)
    return ak.Array(gen_jet_full_info_array)


def get_matched_gen_tau_vis_energy(gen_jets, best_combos, tau_energies):
    gen_jet_full_info_array = []
    for event_id in range(len(gen_jets)):
        mapping = {i[1]: i[0] for i in best_combos[event_id]}
        gen_jet_info_array = []
        for i, gen_jet in enumerate(gen_jets[event_id]):
            if i in best_combos[event_id][:, 1]:
                if len(tau_energies[event_id]) == 0:
                    value = -1
                else:
                    value = tau_energies[event_id][mapping[i]]
                gen_jet_info_array.append(value)
            else:
                gen_jet_info_array.append(-1)
        gen_jet_full_info_array.append(gen_jet_info_array)
    return ak.Array(gen_jet_full_info_array)


def get_gen_tau_jet_info(gen_jets, tau_mask, mc_particles, mc_p4):
    best_combos = get_all_tau_best_combinations(mc_p4, gen_jets, tau_mask)
    tau_energies = get_tau_energies(tau_mask, mc_particles, mc_p4)
    tau_decaymodes = get_all_tau_decaymodes(mc_particles, tau_mask)
    tau_gen_jet_vis_energies = get_matched_gen_tau_vis_energy(gen_jets, best_combos, tau_energies)
    tau_gen_jet_decaymodes = get_matched_gen_tau_decaymode(gen_jets, best_combos, tau_decaymodes)
    return tau_gen_jet_vis_energies, tau_gen_jet_decaymodes


def process_input_file(arrays: ak.Array):
    mc_particles, mc_p4 = calculate_p4(p_type="MCParticles", arrs=arrays)
    reco_particles, reco_p4 = calculate_p4(p_type="MergedRecoParticles", arrs=arrays)
    # reco_particles, reco_p4 = clean_reco_particles(reco_particles=reco_particles, reco_p4=reco_p4)
    reco_jets, reco_jet_constituent_indices = cluster_jets(reco_p4)
    # stable_pythia_mask = mc_particles["generatorStatus"] == 1
    # gen_jets, gen_jet_constituent_indices = cluster_jets(ak.Array(mc_p4[stable_pythia_mask]))
    gen_jets, gen_jet_constituent_indices = cluster_jets(mc_p4)
    reco_indices, gen_indices = get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jet_constituent_indices = ak.from_iter([reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = to_fourvec(ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)]))
    gen_jets = to_fourvec(ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)]))
    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)
    gen_jet_tau_vis_energy, gen_jet_tau_decaymode = get_gen_tau_jet_info(gen_jets, tau_mask, mc_particles, mc_p4)
    data = {
        "event_reco_candidates": ak.from_iter(
            [[reco_p4[i] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        ),
        "reco_cand_p4s": get_jet_constituent_p4s(reco_p4, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_cand_charge": get_jet_constituent_charges(reco_particles, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_cand_pdg": get_jet_constituent_pdgs(reco_particles, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_jet_p4s": reco_jets,
        "gen_jet_p4s": gen_jets,
        "gen_jet_tau_decaymode": gen_jet_tau_vis_energy,
        "gen_jet_tau_vis_energy": gen_jet_tau_decaymode,
    }
    data = {key: ak.flatten(value, axis=1) for key, value in data.items()}
    return data  # Testina saab kontrollida kas kõik sama shapega


def process_single_file(input_path: str, tree_path: str, branches: list, output_dir: str):
    # print(f"[{i}/{len(input_paths)}] Loading contents of {path}")
    start_time = time.time()
    arrays = load_single_file_contents(input_path, tree_path, branches)
    data = process_input_file(arrays)
    file_name = os.path.basename(input_path).replace(".root", ".parquet")
    output_ntuple_path = os.path.join(output_dir, file_name)
    save_record_to_file(data, output_ntuple_path)
    end_time = time.time()
    print(f"Finished processing in {end_time-start_time} s.")


@hydra.main(config_path="../config", config_name="ntupelizer")
def process_all_input_files(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    input_wcp = os.path.join(cfg.input_dir, "*.root")
    if cfg.test_run:
        n_files = 1
    else:
        n_files = None
    input_paths = glob.glob(input_wcp)[:n_files]
    pool = multiprocessing.Pool(processes=8)
    pool.starmap(process_single_file, zip(input_paths, repeat(cfg.tree_path), repeat(cfg.branches), repeat(cfg.output_dir)))


if __name__ == "__main__":
    process_all_input_files()
