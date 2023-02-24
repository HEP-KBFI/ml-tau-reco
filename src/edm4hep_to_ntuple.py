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
import general as g
import awkward as ak
import multiprocessing
from itertools import repeat
from omegaconf import DictConfig
from lifeTimeTools import findTrackPCAs


def save_record_to_file(data: dict, output_path: str) -> None:
    print(f"Saving to precessed data to {output_path}")
    ak.to_parquet(ak.Record(data), output_path)


def load_single_file_contents(
    path: str,
    tree_path: str = "events",
    branches: list = ["MCParticles", "MergedRecoParticles", "SiTracks_Refitted_1", "PrimaryVertices"],
) -> ak.Array:
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
                "px": particles["momentum.x"],
                "py": particles["momentum.y"],
                "pz": particles["momentum.z"],
            }
        )
    )
    return particles, particle_p4


def cluster_jets(particles_p4):
    jetdef = fastjet.JetDefinition2Param(fastjet.ee_genkt_algorithm, 0.4, -1)
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=20.0))
    jets = vector.awk(ak.zip({"energy": jets["t"], "x": jets["x"], "y": jets["y"], "z": jets["z"]}))
    constituent_index = cluster.constituent_index(min_pt=20.0)
    return jets, constituent_index


###############################################################################
###############################################################################
#####                 TAU DECAYMODE CALCULATION                         #######
###############################################################################
###############################################################################


def get_all_tau_decaymodes(mc_particles, tau_mask, mask_addition):
    all_decaymodes = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        event_decaymodes = []
        if n_daughters == 0:
            event_decaymodes.append(-1)
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            daughter_PDGs = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            event_decaymodes.append(g.get_decaymode(daughter_PDGs))
        all_decaymodes.append(event_decaymodes)
    return ak.from_iter(all_decaymodes)


###############################################################################
###############################################################################
###############              GET ALL TAU DAUGHTERS               ##############
###############################################################################
###############################################################################


def find_tau_daughters_all_generations(mc_particles, tau_mask, mask_addition):
    tau_daughters_all_events = []
    for event_idx in range(len(mc_particles.daughters_begin)):
        tau_daughter_indices = []
        for daughter_idx in range(len(mc_particles.daughters_begin[tau_mask][mask_addition][event_idx])):
            daughters = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][event_idx][daughter_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][event_idx][daughter_idx],
            )
            tau_daughter_indices.extend(daughters)
        event_tau_daughters_begin = mc_particles.daughters_begin[event_idx]
        event_tau_daughters_end = mc_particles.daughters_end[event_idx]
        all_tau_daughter_indices = get_event_tau_daughters(
            tau_daughter_indices, event_tau_daughters_begin, event_tau_daughters_end, mc_particles, event_idx
        )
        tau_daughters_all_events.append(all_tau_daughter_indices)
    return tau_daughters_all_events


def get_event_tau_daughters(
    tau_daughter_indices, event_tau_daughters_begin, event_tau_daughters_end, mc_particles, event_idx
):
    all_tau_daughter_indices = []
    all_tau_daughter_indices.extend(tau_daughter_indices)
    while len(tau_daughter_indices) != 0:
        new_tau_daughter_indices = []
        for tau_daughter_idx in tau_daughter_indices:
            daughter_mask = event_tau_daughters_end[tau_daughter_idx] < ak.num(
                mc_particles.daughters_begin[event_idx], axis=0
            )
            if len(mc_particles.daughters_begin[event_idx][tau_daughter_idx][daughter_mask]) > 0:
                daughters = list(
                    range(
                        event_tau_daughters_begin[tau_daughter_idx][daughter_mask][0],
                        event_tau_daughters_end[tau_daughter_idx][daughter_mask][0],
                    )
                )
                new_tau_daughter_indices.extend(daughters)
        tau_daughter_indices = new_tau_daughter_indices
        all_tau_daughter_indices.extend(new_tau_daughter_indices)
    return all_tau_daughter_indices


def get_jet_matched_constituent_gen_energy(
    arrays, reco_jet_constituent_indices, num_ptcls_per_jet, mc_p4, gen_tau_daughters
):
    maps = []
    for ridx, midx in zip(arrays["idx_reco"], arrays["idx_mc"]):
        maps.append(dict(zip(ridx, midx)))
    flat_indices = ak.flatten(reco_jet_constituent_indices, axis=-1)
    gen_energies = ak.from_iter(
        [
            ak.from_iter([mc_p4[ev_i][map_[i]].energy if i in gen_tau_daughters[ev_i] else 0 for i in ev])
            for ev_i, (ev, map_) in enumerate(zip(flat_indices, maps))
        ]
    )
    ret = ak.from_iter([ak.unflatten(gen_energies[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])
    return ret


###############################################################################
###############################################################################
###############           MATCH TAUS WITH GEN JETS               ##############
###############################################################################
###############################################################################


def get_all_tau_best_combinations(mc_p4, gen_jets, tau_mask, mask_addition):
    mc_tau_vec = ak.zip(
        {
            "pt": mc_p4[tau_mask][mask_addition].pt,
            "eta": mc_p4[tau_mask][mask_addition].eta,
            "phi": mc_p4[tau_mask][mask_addition].phi,
            "energy": mc_p4[tau_mask][mask_addition].energy,
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


def get_jet_constituent_property(property_, constituent_idx, num_ptcls_per_jet):
    reco_property_flat = property_[ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))]
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


def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0
    # photon, electron, muon
    if abs(pdgid) in [22, 11, 13, 15]:
        return pdgid
    # charged hadron
    if abs(charge) > 0:
        return 211
    # neutral hadron
    return 130


def to_fourvec(jet):
    return vector.awk(ak.zip({"tau": jet.tau, "x": jet.x, "y": jet.y, "z": jet.z}))


def get_matched_gen_jet_p4(reco_jets, gen_jets):
    reco_jets_ = to_vector(reco_jets)
    gen_jets_ = to_vector(gen_jets)
    reco_indices, gen_indices = match_jets(reco_jets_, gen_jets_, deltaR_cut=0.3)
    return reco_indices, gen_indices


def get_matched_gen_tau_property(gen_jets, best_combos, property_, dummy_value=-1):
    gen_jet_full_info_array = []
    for event_id in range(len(gen_jets)):
        mapping = {i[1]: i[0] for i in best_combos[event_id]}
        gen_jet_info_array = []
        for i, gen_jet in enumerate(gen_jets[event_id]):
            if len(best_combos[event_id]) > 0:
                if i in best_combos[event_id][:, 1]:
                    value = property_[event_id][mapping[i]]
                    gen_jet_info_array.append(value)
                else:
                    gen_jet_info_array.append(dummy_value)
            else:
                gen_jet_info_array.append(dummy_value)
        gen_jet_full_info_array.append(gen_jet_info_array)
    return ak.Array(gen_jet_full_info_array)


def get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4):
    all_events_tau_vis_p4s = []
    for e_idx in range(len(mc_particles.PDG[tau_mask][mask_addition])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][mask_addition][e_idx])
        tau_vis_p4s = []
        for d_idx in range(n_daughters):
            tau_vis_p4 = vector.awk(
                ak.zip(
                    {
                        "mass": [0.0],
                        "x": [0.0],
                        "y": [0.0],
                        "z": [0.0],
                    }
                )
            )[0]
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][mask_addition][e_idx][d_idx],
                mc_particles.daughters_end[tau_mask][mask_addition][e_idx][d_idx],
            )
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            for tau_daughter_p4 in mc_p4[e_idx][daughter_indices][vis_particle_map]:
                tau_vis_p4 = tau_vis_p4 + tau_daughter_p4
            tau_vis_p4s.append(tau_vis_p4)
        if len(tau_vis_p4s) > 0:
            all_events_tau_vis_p4s.append(tau_vis_p4s)
        else:
            all_events_tau_vis_p4s.append(
                vector.awk(
                    ak.zip(
                        {
                            "mass": [0.0],
                            "x": [0.0],
                            "y": [0.0],
                            "z": [0.0],
                        }
                    )
                )
            )
    all_events_tau_vis_p4s = ak.from_iter(all_events_tau_vis_p4s)
    all_events_tau_vis_p4s = vector.awk(
        ak.zip(
            {
                "mass": all_events_tau_vis_p4s.tau,
                "x": all_events_tau_vis_p4s.x,
                "y": all_events_tau_vis_p4s.y,
                "z": all_events_tau_vis_p4s.z,
            }
        )
    )
    return all_events_tau_vis_p4s


def get_gen_tau_jet_info(gen_jets, tau_mask, mask_addition, mc_particles, mc_p4):
    best_combos = get_all_tau_best_combinations(mc_p4, gen_jets, tau_mask, mask_addition)
    vis_tau_p4s = get_vis_tau_p4s(tau_mask, mask_addition, mc_particles, mc_p4)
    tau_energies = vis_tau_p4s.energy
    tau_decaymodes = get_all_tau_decaymodes(mc_particles, tau_mask, mask_addition)
    tau_charges = mc_particles.charge[tau_mask][mask_addition]
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
    gen_tau_jet_info = {
        "gen_jet_tau_vis_energy": get_matched_gen_tau_property(gen_jets, best_combos, tau_energies),
        "gen_jet_tau_decaymode": get_matched_gen_tau_property(gen_jets, best_combos, tau_decaymodes),
        "tau_gen_jet_charge": get_matched_gen_tau_property(gen_jets, best_combos, tau_charges),
        "tau_gen_jet_p4s": get_matched_gen_tau_property(
            gen_jets, best_combos, vis_tau_p4s, dummy_value=tau_gen_jet_p4s_fill_value
        ),
    }
    return gen_tau_jet_info


def get_stable_mc_particles(mc_particles, mc_p4):
    stable_pythia_mask = mc_particles["generatorStatus"] == 1
    neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (abs(mc_particles["PDG"]) != 16)
    particle_mask = stable_pythia_mask * neutrino_mask
    stable_mc_particles = ak.Array({field: ak.Array(mc_particles[field][particle_mask]) for field in mc_particles.fields})
    stable_mc_p4 = mc_p4[particle_mask]
    stable_mc_p4 = vector.awk(
        ak.zip(
            {
                "tau": stable_mc_p4["tau"],
                "px": stable_mc_p4["x"],
                "py": stable_mc_p4["y"],
                "pz": stable_mc_p4["z"],
            }
        )
    )
    return stable_mc_p4, stable_mc_particles


def get_reco_particle_pdg(reco_particles):
    reco_particle_pdg = []
    for i in range(len(reco_particles.charge)):
        charges = ak.flatten(reco_particles["charge"][i], axis=-1).to_numpy()
        pdgs = ak.flatten(reco_particles["type"][i], axis=-1).to_numpy()
        mapped_pdgs = ak.from_iter([map_pdgid_to_candid(pdgs[j], charges[j]) for j in range(len(pdgs))])
        reco_particle_pdg.append(mapped_pdgs)
    return ak.from_iter(reco_particle_pdg)


def clean_reco_particles(reco_particles, reco_p4):
    mask = reco_particles["type"] != 0
    reco_particles = ak.Record({k: reco_particles[k][mask] for k in reco_particles.fields})
    reco_p4 = vector.awk(
        ak.zip(
            {
                "px": reco_p4[mask].x,
                "py": reco_p4[mask].y,
                "pz": reco_p4[mask].z,
                "mass": reco_p4[mask].tau,
            }
        )
    )
    return reco_particles, reco_p4


def get_hadronically_decaying_hard_tau_masks(mc_particles):
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)
    mask_addition = []
    for i in range(len(mc_particles.PDG[tau_mask])):
        n_daughters = len(mc_particles.daughters_begin[tau_mask][i])
        daughter_mask = []
        for d in range(n_daughters):
            parent_idx = mc_particles.parents_begin[tau_mask][i][d]
            initial_tau = parent_idx < len(mc_particles.PDG[i])
            initial_tau_2 = mc_particles.daughters_end[tau_mask][i][d] < len(mc_particles.PDG[i])
            if initial_tau and initial_tau_2:
                parent_pdg = mc_particles.PDG[i][parent_idx]
                daughters_idx = range(
                    mc_particles.daughters_begin[tau_mask][i][d], mc_particles.daughters_end[tau_mask][i][d]
                )
                daughter_PDGs = mc_particles.PDG[i][daughters_idx]
                decaymode = g.get_decaymode(daughter_PDGs)
                if decaymode != 16 and abs(parent_pdg) == 15:
                    daughter_mask.append(True)
                else:
                    daughter_mask.append(False)
            else:
                daughter_mask.append(False)
        mask_addition.append(daughter_mask)
    mask_addition = ak.Array(mask_addition)
    return tau_mask, mask_addition


def process_input_file(arrays: ak.Array):
    mc_particles, mc_p4 = calculate_p4(p_type="MCParticles", arrs=arrays)
    reco_particles, reco_p4 = calculate_p4(p_type="MergedRecoParticles", arrs=arrays)
    reco_particles, reco_p4 = clean_reco_particles(reco_particles=reco_particles, reco_p4=reco_p4)
    reco_jets, reco_jet_constituent_indices = cluster_jets(reco_p4)
    stable_mc_p4, stable_mc_particles = get_stable_mc_particles(mc_particles, mc_p4)
    gen_jets = cluster_jets(stable_mc_p4)[0]
    reco_indices, gen_indices = get_matched_gen_jet_p4(reco_jets, gen_jets)
    reco_jet_constituent_indices = ak.from_iter([reco_jet_constituent_indices[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = ak.from_iter([reco_jets[i][idx] for i, idx in enumerate(reco_indices)])
    reco_jets = vector.awk(ak.zip({"energy": reco_jets.t, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z}))
    gen_jets = ak.from_iter([gen_jets[i][idx] for i, idx in enumerate(gen_indices)])
    gen_jets = vector.awk(ak.zip({"energy": gen_jets.t, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z}))
    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)
    tau_mask, mask_addition = get_hadronically_decaying_hard_tau_masks(mc_particles)
    gen_tau_jet_info = get_gen_tau_jet_info(gen_jets, tau_mask, mask_addition, mc_particles, mc_p4)
    gen_tau_daughters = find_tau_daughters_all_generations(mc_particles, tau_mask, mask_addition)
    event_reco_cand_p4s = ak.from_iter([[reco_p4[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_lifetime_infos = ak.from_iter([findTrackPCAs(arrays, i) for i in range(len(reco_jets))])
    event_dxy = ak.from_iter(event_lifetime_infos[i][:, 0] for i in range(len(reco_jets)))
    event_dz = ak.from_iter(event_lifetime_infos[i][:, 1] for i in range(len(reco_jets)))
    event_d3 = ak.from_iter(event_lifetime_infos[i][:, 2] for i in range(len(reco_jets)))
    event_d0 = ak.from_iter(event_lifetime_infos[i][:, 3] for i in range(len(reco_jets)))
    event_z0 = ak.from_iter(event_lifetime_infos[i][:, 4] for i in range(len(reco_jets)))
    event_PCA_p1 = ak.from_iter(event_lifetime_infos[i][:, 5] for i in range(len(reco_jets)))
    event_PCA_p2 = ak.from_iter(event_lifetime_infos[i][:, 6] for i in range(len(reco_jets)))
    event_PCA_tPCA = ak.from_iter(event_lifetime_infos[i][:, 7] for i in range(len(reco_jets)))

    event_reco_cand_dxy = ak.from_iter([[event_dxy[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_dz = ak.from_iter([[event_dz[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_d3 = ak.from_iter([[event_d3[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_d0 = ak.from_iter([[event_d0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_z0 = ak.from_iter([[event_z0[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))])
    event_reco_cand_PCA_p1 = ak.from_iter(
        [[event_PCA_p1[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_p2 = ak.from_iter(
        [[event_PCA_p2[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )
    event_reco_cand_PCA_tPCA = ak.from_iter(
        [[event_PCA_tPCA[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
    )

    reco_cand_dxy = get_jet_constituent_property(event_dxy, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_dz = get_jet_constituent_property(event_dz, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d3 = get_jet_constituent_property(event_d3, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_d0 = get_jet_constituent_property(event_d0, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_z0 = get_jet_constituent_property(event_z0, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_p1 = get_jet_constituent_property(event_PCA_p1, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_p2 = get_jet_constituent_property(event_PCA_p2, reco_jet_constituent_indices, num_ptcls_per_jet)
    reco_cand_PCA_tPCA = get_jet_constituent_property(event_PCA_tPCA, reco_jet_constituent_indices, num_ptcls_per_jet)
    ## Dummy values for sigma_z0 and sigma_d0 as per request to be always 0.035
    dummy_uncert_d0_z0 = ak.ones_like(reco_cand_d0) * 0.035
    event_dummy_uncert_d0_z0 = ak.ones_like(event_reco_cand_dxy) * 0.035
    ##
    reco_particle_pdg = get_reco_particle_pdg(reco_particles)
    data = {
        "event_reco_cand_p4s": vector.awk(
            ak.zip(
                {
                    "px": event_reco_cand_p4s.x,
                    "py": event_reco_cand_p4s.y,
                    "pz": event_reco_cand_p4s.z,
                    "mass": event_reco_cand_p4s.tau,
                }
            )
        ),
        "event_reco_cand_pdg": ak.from_iter(
            [[reco_particle_pdg[j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        ),
        "event_reco_cand_charge": ak.from_iter(
            [[reco_particles["charge"][j] for i in range(len(reco_jets[j]))] for j in range(len(reco_jets))]
        ),
        "reco_cand_p4s": get_jet_constituent_p4s(reco_p4, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_cand_charge": get_jet_constituent_property(
            reco_particles["charge"], reco_jet_constituent_indices, num_ptcls_per_jet
        ),
        "reco_cand_pdg": get_jet_constituent_property(reco_particle_pdg, reco_jet_constituent_indices, num_ptcls_per_jet),
        "reco_jet_p4s": vector.awk(
            ak.zip({"mass": reco_jets.mass, "px": reco_jets.x, "py": reco_jets.y, "pz": reco_jets.z})
        ),
        "event_reco_cand_dxy": event_reco_cand_dxy,
        "event_reco_cand_dz": event_reco_cand_dz,
        "event_reco_cand_d3": event_reco_cand_d3,
        "event_reco_cand_d0": event_reco_cand_d0,
        "event_reco_cand_z0": event_reco_cand_z0,
        "event_reco_cand_PCA_p1": event_reco_cand_PCA_p1,
        "event_reco_cand_PCA_p2": event_reco_cand_PCA_p2,
        "event_reco_cand_PCA_tPCA": event_reco_cand_PCA_tPCA,
        "event_cand_d0err": event_dummy_uncert_d0_z0,
        "event_cand_dzerr": event_dummy_uncert_d0_z0,
        "reco_cand_dxy": reco_cand_dxy,
        "reco_cand_dz": reco_cand_dz,
        "reco_cand_d3": reco_cand_d3,
        "reco_cand_d0": reco_cand_d0,
        "reco_cand_z0": reco_cand_z0,
        "reco_cand_PCA_p1": reco_cand_PCA_p1,
        "reco_cand_PCA_p2": reco_cand_PCA_p2,
        "reco_cand_PCA_tPCA": reco_cand_PCA_tPCA,
        "reco_cand_d0err": dummy_uncert_d0_z0,
        "reco_cand_dzerr": dummy_uncert_d0_z0,
        "gen_jet_p4s": vector.awk(ak.zip({"mass": gen_jets.mass, "px": gen_jets.x, "py": gen_jets.y, "pz": gen_jets.z})),
        "gen_jet_tau_decaymode": gen_tau_jet_info["gen_jet_tau_decaymode"],
        "gen_jet_tau_vis_energy": gen_tau_jet_info["gen_jet_tau_vis_energy"],
        "gen_jet_tau_charge": gen_tau_jet_info["tau_gen_jet_charge"],
        "gen_jet_tau_p4s": gen_tau_jet_info["tau_gen_jet_p4s"],
        "reco_cand_matched_gen_energy": get_jet_matched_constituent_gen_energy(
            arrays, reco_jet_constituent_indices, num_ptcls_per_jet, mc_p4, gen_tau_daughters
        ),
    }
    return data


def process_single_file(input_path: str, tree_path: str, branches: list, output_dir: str):
    # print(f"[{i}/{len(input_paths)}] Loading contents of {path}")
    start_time = time.time()
    arrays = load_single_file_contents(input_path, tree_path, branches)
    data = process_input_file(arrays)
    data = {key: ak.flatten(value, axis=1) for key, value in data.items()}
    file_name = os.path.basename(input_path).replace(".root", ".parquet")
    output_ntuple_path = os.path.join(output_dir, file_name)
    save_record_to_file(data, output_ntuple_path)
    end_time = time.time()
    print(f"Finished processing in {end_time-start_time} s.")


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def process_all_input_files(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    for sample in cfg.samples_to_process:
        output_dir = cfg.samples[sample].output_dir
        input_dir = cfg.samples[sample].input_dir
        os.makedirs(output_dir, exist_ok=True)
        input_wcp = os.path.join(input_dir, "*.root")
        if cfg.test_run:
            n_files = 1
        else:
            n_files = None
        input_paths = glob.glob(input_wcp)[:n_files]
        pool = multiprocessing.Pool(processes=8)
        pool.starmap(process_single_file, zip(input_paths, repeat(cfg.tree_path), repeat(cfg.branches), repeat(output_dir)))


if __name__ == "__main__":
    process_all_input_files()
