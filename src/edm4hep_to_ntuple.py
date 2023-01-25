import awkward as ak
import numpy as np
import sys
import uproot
import fastjet
import vector

# Remap various PDG-IDs to just photon, electron, muon, tau, charged hadron, neutral hadron
def map_pdgid_to_candid(pdgid, charge):
    if pdgid == 0:
        return 0

    # photon, electron, muon
    if pdgid in [22, 11, 13, 15]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("running:", sys.argv[0], infile, outfile)

    fi = uproot.open(infile)
    ev = fi["events"]

    this_file_arrs = ev.arrays(["MCParticles", "MergedRecoParticles"])

    idx0 = "RecoMCTruthLink#0/RecoMCTruthLink#0.index"
    idx1 = "RecoMCTruthLink#1/RecoMCTruthLink#1.index"

    idx_recoparticle = ev.arrays(idx0)[idx0]
    idx_mcparticle = ev.arrays(idx1)[idx1]

    # index in the MergedRecoParticles collection
    this_file_arrs["idx_reco"] = idx_recoparticle

    # index in the MCParticles collection
    this_file_arrs["idx_mc"] = idx_mcparticle

    arrs = [this_file_arrs]
    arrs = ak.concatenate(arrs)

    import pdb

    pdb.set_trace()

    print(arrs)

    reco_jet_p4s = np.zeros((100000, 4), dtype=np.float32)

    # Prepare 4-momentum of reco particles
    mrp = arrs["MergedRecoParticles"]
    mrp = ak.Record({k.replace("MergedRecoParticles.", ""): mrp[k] for k in mrp.fields})
    reco_p4 = vector.awk(
        ak.zip({"mass": mrp["mass"], "x": mrp["momentum.x"], "y": mrp["momentum.y"], "z": mrp["momentum.z"]})
    )

    # Cluster AK4 jets from PF particles with min pt 2 GeV
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    cluster = fastjet.ClusterSequence(reco_p4.to_xyzt(), jetdef)

    constituent_idx = cluster.constituent_index(min_pt=2.0)
    constituent_idx

    print(constituent_idx.to_list())

    p4_flat = reco_p4[ak.flatten(constituent_idx, axis=-1)]
    num_ptcls_per_jet = ak.num(constituent_idx, axis=-1)
    ret = ak.from_iter([ak.unflatten(p4_flat[i], num_ptcls_per_jet[i], axis=-1) for i in range(len(num_ptcls_per_jet))])

    ret2 = vector.awk(ak.zip({"x": ret.x, "y": ret.y, "z": ret.z, "t": ret.tau}))

    print(ret2)
    ak.to_parquet(ak.Record({"reco_pf_p4s": ret2}), outfile)
