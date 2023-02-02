import awkward as ak
import json
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
    retVal = list(vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "tau": data.tau})))
    return retVal


def data_to_p4s_x2(data):
    retVal = data_to_p4s(data)
    retVal = [list(p4s) for p4s in retVal]
    return retVal


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
                jet.print()
            iso_cands = buildCands(event_cand_p4s[idxJet], event_cand_pdg[idxJet], event_cand_charge[idxJet])
            # CV: reverse=True argument needed in order to sort candidates in order of decreasing (and NOT increasing) pT)
            iso_cands.sort(key=lambda cand: cand.pt, reverse=True)
            if self.verbosity >= 4:
                print("iso_cands:")
                for cand in iso_cands:
                    cand.print()
            tau = self.hpsAlgo.buildTau(jet, iso_cands)
            if tau is None:
                print("Warning: Failed to find tau -> building dummy")
                # CV: build "dummy" tau to maintain 1-to-1 correspondence between taus and jets
                tau = Tau()
                tau.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
                tau.signalCands = []
                tau.isoCands = []
                tau.idDiscr = -1
                tau.q = 0
                tau.decayMode = "undefined"
            if self.verbosity >= 2:
                tau.print()
            if idxJet > 5:
                raise ValueError("STOP.")
            taus.append(tau)

        retVal = {
            "tauP4": ak.Array([tau.p4 for tau in taus]),
            "tauSigCandP4s": ak.Array([ak.Array([cand.p4 for cand in tau.signalCands]) for tau in taus]),
            "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),
            "tauCharge": ak.Array([tau.q for tau in taus]),
            "tauDmode": ak.Array([get_decayMode(tau) for tau in taus]),
        }
        return retVal
