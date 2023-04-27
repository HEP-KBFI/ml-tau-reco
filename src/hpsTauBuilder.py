import json
import os
import vector

from basicTauBuilder import BasicTauBuilder
from hpsAlgo import HPSAlgo
from hpsCand import readCands
from hpsJet import readJets
from hpsTau import Tau, writeTaus

# CV: to run the HPS tau reconstruction algorithm, execute
#       './scripts/run-env.sh python3 src/runBuilder.py builder=HPS samples_to_process=['ZH_Htautau'] n_files=1 verbosity=1'
#     in the ml-tau-reco directory


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

        jets = readJets(data)
        event_cands = readCands(data)

        taus = []
        for idxJet, jet in enumerate(jets):
            if self.verbosity >= 2:
                print("Processing entry %i" % idxJet)
                jet.print()
            elif idxJet > 0 and (idxJet % 100) == 0:
                print("Processing entry %i" % idxJet)
            # CV: enable the following two lines for faster turn-around time when testing
            # if idxJet > 10:
            #    continue

            event_iso_cands = event_cands[idxJet]
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
                tau.updatePtEtaPhiMass()
                tau.signal_cands = set()
                tau.signal_gammaCands = set()
                tau.iso_cands = set()
                tau.iso_chargedCands = set()
                tau.iso_gammaCands = set()
                tau.iso_neutralHadronCands = set()
                tau.metric_dR_or_angle = None
                tau.metric_dEta_or_dTheta = None
                tau.idDiscr = -1.0
                tau.q = 0.0
                tau.decayMode = "undefined"
                tau.barcode = -1
            if self.verbosity >= 2:
                tau.print()
            if self.verbosity >= 4 and idxJet > 100:
                raise ValueError("STOP.")
            taus.append(tau)

        retVal = writeTaus(taus)
        return retVal
