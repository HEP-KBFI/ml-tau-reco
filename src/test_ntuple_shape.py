import sys
import awkward as ak


def test_data_shapes(file_path):
    data = ak.Array((ak.from_parquet(file_path).tolist()))
    data = data[0]  #  Just take first event temporarily


def level_one(data):
    event_reco_candidates = ak.num(data.event_reco_candidates, axis=0)
    reco_cand_p4s = ak.num(data.reco_cand_p4s, axis=0)
    reco_cand_charge = ak.num(data.reco_cand_charge, axis=0)
    reco_cand_pdg = ak.num(data.reco_cand_pdg, axis=0)
    reco_jet_p4s = ak.num(data.reco_jet_p4s, axis=0)
    gen_jet_p4s = ak.num(data.gen_jet_p4s, axis=0)
    gen_jet_tau_decaymode = ak.num(data.gen_jet_tau_decaymode, axis=0)
    gen_jet_tau_vis_energy = ak.num(data.gen_jet_tau_vis_energy, axis=0)
    assert event_reco_candidates == reco_cand_p4s
    assert reco_cand_p4s == reco_cand_charge
    assert reco_cand_pdg == reco_cand_charge
    assert reco_cand_pdg == reco_jet_p4s
    assert reco_jet_p4s == gen_jet_p4s
    assert gen_jet_tau_decaymode == gen_jet_p4s
    assert gen_jet_tau_vis_energy == gen_jet_tau_decaymode
    assert reco_cand_pdg == reco_jet_p4s
    assert reco_cand_charge == reco_jet_p4s


def level_two(data):
    reco_cand_p4s = sum(ak.num(data.reco_cand_p4s, axis=1))
    reco_cand_charge = sum(ak.num(data.reco_cand_charge, axis=1))
    reco_cand_pdg = sum(ak.num(data.reco_cand_pdg, axis=1))
    assert reco_cand_p4s == reco_cand_charge
    assert reco_cand_pdg == reco_cand_charge


if __name__ == "__main__":
    file_path = sys.argv[1]
    test_ntuple_shape(file_path)
