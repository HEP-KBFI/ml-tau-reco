import os
import glob
import json
import hydra
import vector
import numpy as np
import awkward as ak
import edm4hep_to_ntuple as nt
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="genTau_inspector", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_arrays = {}
    input_paths = glob.glob(os.path.join(cfg.samples.ZH_Htautau.input_dir, "*.root"))
    if cfg.n_files_per_sample == -1:
        n_files = None
    else:
        n_files = cfg.n_files_per_sample
    arrays = []
    for input_path in input_paths[:n_files]:
        arrays.append(nt.load_single_file_contents(input_path, cfg.tree_path, cfg.branches))
    arrays = ak.concatenate(arrays)
    mc_particles, mc_p4 = nt.calculate_p4("MCParticles", arrays)
    gen_vis_tau_info = get_gen_vis_tau_info(mc_particles, mc_p4)
    output_path = os.path.expandvars(cfg.output_path)
    with open(output_path, 'wt') as out_file:
        json.dump(gen_vis_tau_info, out_file, indent=4)


def get_gen_vis_tau_info(mc_particles, mc_p4):
    tau_mask = (np.abs(mc_particles["PDG"]) == 15) & (mc_particles["generatorStatus"] == 2)
    all_event_taus = []
    for e_idx in range(len(mc_particles.PDG[tau_mask])):
        daughter_mask = mc_particles.daughters_end[tau_mask][e_idx] < ak.num(mc_particles.daughters_begin[e_idx], axis=0)
        n_daughters = len(mc_particles.daughters_begin[tau_mask][e_idx][daughter_mask])
        all_taus = []
        for d_idx in range(n_daughters):
            daughter_indices = range(
                mc_particles.daughters_begin[tau_mask][e_idx][daughter_mask][d_idx],
                mc_particles.daughters_end[tau_mask][e_idx][daughter_mask][d_idx],
            )
            p4s = mc_p4[e_idx][daughter_indices]
            PDG_ids = np.abs(mc_particles.PDG[e_idx][daughter_indices])
            vis_particle_map = (PDG_ids != 12) * (PDG_ids != 14) * (PDG_ids != 16)
            p4s_ = p4s[vis_particle_map]
            summed_vis_tau = vector.awk(
                ak.zip(
                    {
                        "px": [ak.sum(p4s_.x, axis=-1)],
                        "py": [ak.sum(p4s_.y, axis=-1)],
                        "pz": [ak.sum(p4s_.z, axis=-1)],
                        "mass": [ak.sum(p4s_.tau, axis=-1)],
                    }
                )
            )
            tau_info = {
                "eta": float(summed_vis_tau.eta[0]),
                "phi": float(summed_vis_tau.phi[0]),
                "pt": float(summed_vis_tau.pt[0]),
                "daughters": []
            }
            for p4, pdg in zip(p4s, PDG_ids):
                tau_info['daughters'].append({
                        "PDG": int(pdg),
                        "eta": float(p4.eta),
                        "phi": float(p4.phi),
                        "pt": float(p4.pt)
                    })
            all_taus.append(tau_info)
        all_event_taus.append(all_taus)
    return all_event_taus


if __name__ == '__main__':
    main()
