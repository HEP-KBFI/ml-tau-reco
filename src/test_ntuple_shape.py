"""Script for testing the output shapes of the saved data
Call with 'python3'

Usage:
    test_ntuple_shape.py --file_path=STR

Options:
    -f --file_path=STR       File path to be used for testing

"""

import docopt
import awkward as ak


def test_data_shapes(file_path):
    data = ak.Array((ak.from_parquet(file_path).tolist()))
    level_one(data)
    level_two(data)


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
    print("Level one correct")


def level_two(data):
    reco_cand_p4s = sum(ak.num(data.reco_cand_p4s, axis=1))
    reco_cand_charge = sum(ak.num(data.reco_cand_charge, axis=1))
    reco_cand_pdg = sum(ak.num(data.reco_cand_pdg, axis=1))
    assert reco_cand_p4s == reco_cand_charge
    assert reco_cand_pdg == reco_cand_charge
    print("Level two correct")


if __name__ == "__main__":
    try:
        arguments = docopt.docopt(__doc__)
        file_path = arguments["--file_path"]
        test_data_shapes(file_path)
    except docopt.DocoptExit as e:
        print(e)
