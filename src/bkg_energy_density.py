import os
import glob
import uproot
import hydra
import vector
import fastjet
import numpy as np
import general as g
import awkward as ak
import edm4hep_to_ntuple as nt
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import mplhep as hep
import json

hep.style.use(hep.styles.CMS)


def load_pp_file_contents(pp_path, branch, max_events):
    branches_of_interest = ['Px', 'Py', 'Pz', 'Mass', 'PID', 'Charge', 'M1', 'M2', 'D1', 'D2', 'Status']
    branch_names = [f"{branch}.{var}" for var in branches_of_interest]
    with uproot.open(pp_path) as in_file:
        tree = in_file[f"Delphes/{branch}"]
        arrays = tree.arrays(branch_names)
    particles = ak.Record({k.replace(f"{branch}.", "").lower(): arrays[k] for k in arrays.fields})
    mc_particles = ak.Record({
            'daughters_begin': particles.d1[:max_events],
            'daughters_end': particles.d2[:max_events],
            'parents_begin': particles.m1[:max_events],
            'parents_end': particles.m2[:max_events],
            'PDG': particles.pid[:max_events],
            'charge': particles.charge[:max_events],
            'generatorStatus': particles.status[:max_events]
    })
    mc_p4 = vector.awk(ak.zip({
        "mass": particles.mass,
        "x": particles.px,
        "y": particles.py,
        "z": particles.pz,
    }))[:max_events]
    # stable_pythia_mask = mc_particles["generatorStatus"] == 1
    # neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (abs(mc_particles["PDG"]) != 16)
    # stability_mask = stable_pythia_mask * neutrino_mask
    # mc_p4 = g.reinitialize_p4(mc_p4[stability_mask])
    return mc_p4


def load_ee_file_contents(ee_dir, max_events):
    n_files = int(np.ceil(max_events / 100))
    paths = glob.glob(os.path.join(ee_dir, '*.root'))[:n_files]
    arrays = []
    for path in paths:
        arrays.append(nt.load_single_file_contents(path))
    arrays = ak.concatenate(arrays)[:max_events]
    mc_particles, mc_p4 = nt.calculate_p4(p_type="MCParticles", arrs=arrays)
    # stable_pythia_mask = mc_particles["generatorStatus"] == 1
    # neutrino_mask = (abs(mc_particles["PDG"]) != 12) * (abs(mc_particles["PDG"]) != 14) * (abs(mc_particles["PDG"]) != 16)
    # stability_mask = stable_pythia_mask * neutrino_mask
    # mc_p4 = g.reinitialize_p4(mc_p4[stability_mask])
    return mc_p4


def p4_to_pseudojet(mc_p4):
    mc_p4_px = ak.Array([np.array(mp4, dtype=np.float64) for mp4 in mc_p4.px])
    mc_p4_py = ak.Array([np.array(mp4, dtype=np.float64) for mp4 in mc_p4.py])
    mc_p4_pz = ak.Array([np.array(mp4, dtype=np.float64) for mp4 in mc_p4.pz])
    mc_p4_E = ak.Array([np.array(mp4, dtype=np.float64) for mp4 in mc_p4.E])
    pseudojet_particles = []
    for i in range(len(mc_p4_px)):
        event_pseudojet_particles = []
        for j in range(len(mc_p4_px[i])):
            pseudojet_particle = fastjet.PseudoJet(
                mc_p4_px[i][j],
                mc_p4_py[i][j],
                mc_p4_pz[i][j],
                mc_p4_E[i][j]
            )
            event_pseudojet_particles.append(pseudojet_particle)
        pseudojet_particles.append(event_pseudojet_particles)
    return pseudojet_particles


def analyze_energy_density(mc_p4, cfg, collider_type):
    pseudojet_particles = p4_to_pseudojet(mc_p4)
    bge = fastjet.JetMedianBackgroundEstimator(
        fastjet.SelectorAbsRapMax(4.5),
        fastjet.JetDefinition(fastjet.kt_algorithm, 0.4),
        fastjet.AreaDefinition(fastjet.VoronoiAreaSpec(0.9))
    )
    rhos = []
    sigmas = []
    for i in range(len(pseudojet_particles)):
        bge.set_particles(pseudojet_particles[i])
        rhos.append(bge.rho())
        sigmas.append(bge.sigma())
        bge.reset()
    return rhos, sigmas


@hydra.main(config_path="../config", config_name="environments", version_base=None)
def compare_different_environments(cfg: DictConfig) -> None:
    ee_mc_p4 = load_ee_file_contents(cfg.samples.ee, max_events=cfg.max_events)
    pp_mc_p4_wPU = load_pp_file_contents(cfg.samples.pp, branch='PFParticlesAll', max_events=cfg.max_events)
    pp_mc_p4_noPU = load_pp_file_contents(cfg.samples.pp, branch='PFParticlesNoPU', max_events=cfg.max_events)
    ee_rhos, ee_sigmas = analyze_energy_density(ee_mc_p4, cfg, collider_type='ee')
    pp_wPU_rhos, pp_wPU_sigmas = analyze_energy_density(pp_mc_p4_wPU, cfg, collider_type='pp')
    pp_noPU_rhos, pp_noPU_sigmas = analyze_energy_density(pp_mc_p4_noPU, cfg, collider_type='pp')

    print("----------------------- EE ---------------------------------")
    print(f"RHO \t mean: {np.mean(ee_rhos)} \t std: {np.std(ee_rhos)}")
    print(f"SIGMA \t mean: {np.mean(ee_sigmas)} \t std: {np.std(ee_sigmas)}")

    print("----------------------- pp_wPU -----------------------------")
    print(f"RHO \t mean: {np.mean(pp_wPU_rhos)} \t std: {np.std(pp_wPU_rhos)}")
    print(f"SIGMA \t mean: {np.mean(pp_wPU_sigmas)} \t std: {np.std(pp_wPU_sigmas)}")

    print("----------------------- pp_noPU -----------------------------")
    print(f"RHO \t mean: {np.mean(pp_noPU_rhos)} \t std: {np.std(pp_noPU_rhos)}")
    print(f"SIGMA \t mean: {np.mean(pp_noPU_sigmas)} \t std: {np.std(pp_noPU_sigmas)}")

    info_dict = {
        'ee': {
            'rho': ee_rhos,
            'sigma': ee_sigmas
        },
        'pp_wPU': {
            'rho': pp_wPU_rhos,
            'sigma': pp_wPU_sigmas
        },
        'pp_noPU': {
            'rho': pp_noPU_rhos,
            'sigma': pp_noPU_sigmas
        }
    }
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_path = os.path.join(cfg.output_dir, 'bkg_energy_density.json')
    with open(output_path, 'wt') as in_file:
        json.dump(info_dict, in_file, indent=4)


if __name__ == "__main__":
    compare_different_environments()