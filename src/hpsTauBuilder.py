import awkward as ak
import json
import os
import vector

from basicTauBuilder import BasicTauBuilder
from hpsAlgo import HPSAlgo
from hpsCand import buildCands, isHigherPt
from hpsJet import buildJets
from hpsTau import Tau

# CV: to-run: execute './scripts/run-env.sh python3 src/runBuilder.py --builder hps' in the ml-tau-reco directory


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
    def __init__(self, cfgFileName="./config/hpsAlgo_cfg.json"):
        super(BasicTauBuilder, self).__init__()
        if os.path.isfile(cfgFileName):
            cfgFile = open(cfgFileName, "r")
            cfg = json.load(cfgFile)
            if "HPSAlgo" not in cfg.keys():
                raise RuntimeError("Failed to parse config file %s !!")
            self._builderConfig = cfg["HPSAlgo"]
            self.hpsAlgo = HPSAlgo(self._builderConfig)
            cfgFile.close()
        else:
            raise RuntimeError("Failed to read config file %s !!")

    def processJets(self, jets):
        jets = buildJets(jets["reco_jet_p4s"], jets["reco_cand_p4s"], jets["reco_cand_pdg"], jets["reco_cand_charge"])
        jets.sort(key=isHigherPt)

        iso_cands = buildCands(jets["event_reco_cand_p4s"], jets["event_reco_cand_pdg"], jets["event_reco_cand_charge"])
        iso_cands.sort(key=isHigherPt)

        taus = []
        for jet in jets:
            tau = self.hpsAlgo(jet, iso_cands)
            if tau is None:
                # CV: build "dummy" tau to maintain 1-to-1 correspondence between taus and jets
                tau = Tau()
                tau.p4 = vector(pt=0.0, phi=0.0, theta=0.0, mass=0.0)
                tau.signalCands = []
                tau.idDiscr = -1
                tau.q = 0
                tau.decayMode = "undefined"
            taus.append(tau)

        retVal = {
            "tauP4": ak.Array([tau.p4 for tau in taus]),
            "tauSigCandP4s": ak.Array([ak.Array([cand.p4 for cand in tau.signalCands]) for tau in taus]),
            "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),
            "tauCharge": ak.Array([tau.q for tau in taus]),
            "tauDmode": ak.Array([get_decayMode(tau) for tau in taus]),
        }
        return retVal
