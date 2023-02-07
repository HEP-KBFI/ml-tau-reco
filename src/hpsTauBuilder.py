import awkward as ak
import json
import numpy as np
import os
import vector

from basicTauBuilder import BasicTauBuilder
from hpsAlgo import HPSAlgo
from hpsCand import buildCands
from hpsJet import buildJets
from hpsTau import Tau

# to-run: execute './scripts/run-env.sh python3 src/runBuilder.py --builder hps
#           --input /local/laurits/CLIC_data/ZH_Htautau --verbosity 2' in the ml-tau-reco directory


def data_to_p4s(data):
    retVal = list(vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "mass": data.tau})))
    return retVal


def data_to_p4s_x2(data):
    retVal = data_to_p4s(data)
    retVal = [list(p4s) for p4s in retVal]
    return retVal


def build_dummy_array(num=0, dtype=np.float):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            ak.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def get_decayMode(tau):
    retVal = None
    if tau.decayMode == "undefined":
        retVal = -1
    elif tau.decayMode == "1Prong0Pi0":
        retVal = 0
    elif tau.decayMode == "1Prong1Pi0":
        retVal = 1
    elif tau.decayMode == "1Prong2Pi0":
        retVal = 2
    elif tau.decayMode == "3Prong0Pi0":
        retVal = 10
    elif tau.decayMode == "3Prong1Pi0":
        retVal = 11
    else:
        raise ValueError("Invalid decayMode = '%s'" % tau.decayMode)
    return retVal


class HPSTauBuilder(BasicTauBuilder):
    def __init__(self, cfgFileName="./config/hpsAlgo_cfg.json", verbosity=0):
        super(BasicTauBuilder, self).__init__()
        if os.path.isfile(cfgFileName):
            cfgFile = open(cfgFileName, "r")
            cfg = json.load(cfgFile)
            if "HPSAlgo" not in cfg.keys():
                raise RuntimeError("Failed to parse config file %s !!")
            self._builderConfig = cfg["HPSAlgo"]
            self.hpsAlgo = HPSAlgo(self._builderConfig, verbosity)
            self.verbosity = verbosity
            cfgFile.close()
        else:
            raise RuntimeError("Failed to read config file %s !!")

    def processJets(self, data):
        if self.verbosity >= 3:
            print("data:")
            print(data.fields)

        jet_p4s = data_to_p4s(data["reco_jet_p4s"])
        jet_cand_p4s = data_to_p4s_x2(data["reco_cand_p4s"])
        jet_cand_pdg = data["reco_cand_pdg"]
        jet_cand_charge = data["reco_cand_charge"]

        event_cand_p4s = data_to_p4s_x2(data["event_reco_cand_p4s"])
        event_cand_pdg = data["event_reco_cand_pdg"]
        event_cand_charge = data["event_reco_cand_charge"]

        jets = buildJets(jet_p4s, jet_cand_p4s, jet_cand_pdg, jet_cand_charge)

        taus = []
        for idxJet, jet in enumerate(jets):
            if self.verbosity >= 2:
                print("Processing entry %i" % idxJet)
                jet.print()
            elif idxJet > 0 and (idxJet % 100) == 0:
                print("Processing entry %i" % idxJet)
            # CV: enable the following two lines for faster turn-around time when testing
            # if idxJet > 5:
            #    continue

            event_iso_cands = buildCands(event_cand_p4s[idxJet], event_cand_pdg[idxJet], event_cand_charge[idxJet])
            # CV: reverse=True argument needed in order to sort candidates in order of decreasing (and NOT increasing) pT)
            event_iso_cands.sort(key=lambda cand: cand.pt, reverse=True)
            if self.verbosity >= 4:
                print("event_iso_cands:")
                for cand in event_iso_cands:
                    cand.print()

            tau = self.hpsAlgo.buildTau(jet, event_iso_cands)
            if tau is None:
                if self.verbosity >= 2:
                    print("Failed to find tau associated to jet:")
                    jet.print()
                    print(" -> building dummy tau")
                # CV: build "dummy" tau to maintain 1-to-1 correspondence between taus and jets
                tau = Tau()
                tau.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
                tau.signal_cands = set()
                tau.iso_cands = set()
                tau.idDiscr = -1.0
                tau.q = 0.0
                tau.decayMode = "undefined"
                tau.barcode = -1
            if self.verbosity >= 2:
                tau.print()
            if self.verbosity >= 4 and idxJet > 10:
                raise ValueError("STOP.")
            taus.append(tau)

        retVal = {
            "tau_p4s": vector.awk(
                ak.zip(
                    {
                        "px": [tau.p4.px for tau in taus],
                        "py": [tau.p4.py for tau in taus],
                        "pz": [tau.p4.pz for tau in taus],
                        "mass": [tau.p4.mass for tau in taus],
                    }
                )
            ),
            "tauSigCand_p4s": ak.Array(
                [
                    vector.awk(
                        ak.zip(
                            {
                                "px": [cand.p4.px for cand in tau.signal_cands],
                                "py": [cand.p4.py for cand in tau.signal_cands],
                                "pz": [cand.p4.pz for cand in tau.signal_cands],
                                "mass": [cand.p4.mass for cand in tau.signal_cands],
                            }
                        )
                    )
                    if len(tau.signal_cands) >= 1
                    else build_dummy_array()
                    for tau in taus
                ]
            ),
            "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),
            "tau_charge": ak.Array([tau.q for tau in taus]),
            "tau_decaymode": ak.Array([get_decayMode(tau) for tau in taus]),
        }
        return retVal
