import math

from hpsCand import isHigherPt
from hpsCombinatoricsGenerator import CombinatoricsGenerator
from hpsGetParameter import getParameter
from hpsStrip import Strip
from hpsStripAlgo import StripAlgo
from hpsTau import Tau


def deltaR(cand1, cand2):
    dEta = cand1.eta - cand2.eta
    dPhi = cand1.phi - cand2.phi
    dR = math.sqrt(dEta * dEta + dPhi * dPhi)
    return dR


def selectCandsByDeltaR(cands, ref, dRmax):
    selectedCands = []
    for cand in cands:
        dR = deltaR(cand, ref)
        if dR < dRmax:
            selectedCands.append(cand)
    return selectedCands


def selectCandsByPdgId(cands, pdgIds=[]):
    selectedCands = []
    for cand in cands:
        if cand.pdgId in pdgIds:
            selectedCands.append(cand)
    return selectedCands


def isHigherRank(tau1, tau2):
    if tau1.numChargedCands > tau2.numChargedCands:
        return True
    if tau1.numChargedCands < tau2.numChargedCands:
        return False
    if tau1.pt > tau2.pt:
        return True
    if tau1.pt < tau2.pt:
        return False
    if tau1.numStrips > tau2.numStrips:
        return True
    if tau1.numStrips < tau2.numStrips:
        return False
    if tau1.combinedIso < tau2.combinedIso:
        return True
    if tau1.combinedIso > tau2.combinedIso:
        return False


def compPtSum(cands):
    pt_sum = 0.0
    for cand in cands:
        pt_sum += cand.pt
    return pt_sum


class HPSAlgo:
    def __init__(self, cfg, verbosity=0):
        print("<HPSAlgo::HPSAlgo>:")

        cfgSignalCands = cfg["signalCands"]
        self.signalCand_minChargedHadronPt = getParameter(cfgSignalCands, "minChargedHadronPt", 0.5)
        self.signalCand_minElectronPt = getParameter(cfgSignalCands, "minElectronPt", 0.5)
        # CV: don't use muons when building the signal constituents of the tau
        self.signalCand_minMuonPt = getParameter(cfgSignalCands, "minChargedHadronPt", 1.0e6)
        print("signalCands:")
        print(" minChargedHadronPt = %1.2f" % self.signalCand_minChargedHadronPt)
        print(" minElectronPt = %1.2f" % self.signalCand_minElectronPt)
        print(" minMuonPt = %1.2f" % self.signalCand_minMuonPt)

        cfgIsolationCands = cfg["isolationCands"]
        self.isolationCand_minChargedHadronPt = getParameter(cfgIsolationCands, "minChargedHadronPt", 0.5)
        self.isolationCand_minElectronPt = getParameter(cfgIsolationCands, "minElectronPt", 0.5)
        self.isolationCand_minGammaPt = getParameter(cfgIsolationCands, "minGammaPt", 0.5)
        self.isolationCand_minMuonPt = getParameter(cfgIsolationCands, "minMuonPt", 0.5)
        self.isolationCand_minNeutralHadronPt = getParameter(cfgIsolationCands, "minNeutralHadronPt", 10.0)
        print("isolationCands:")
        print(" minChargedHadronPt = %1.2f" % self.isolationCand_minChargedHadronPt)
        print(" minElectronPt = %1.2f" % self.isolationCand_minElectronPt)
        print(" minGammaPt = %1.2f" % self.isolationCand_minGammaPt)
        print(" minMuonPt = %1.2f" % self.isolationCand_minMuonPt)
        print(" minNeutralHadronPt = %1.2f" % self.isolationCand_minNeutralHadronPt)

        self.matchingConeSize = getParameter(cfg, "matchingConeSize", 1.0e-1)
        self.isolationConeSize = getParameter(cfg, "isolationConeSize", 5.0e-1)
        print(" matchingConeSize = %1.2f" % self.matchingConeSize)
        print(" isolationConeSize = %1.2f" % self.isolationConeSize)

        self.stripAlgo = StripAlgo(cfg["StripAlgo"])

        self.targetedDecayModes = {
            "1Prong0Pi0": {
                "numChargedCands": 1,
                "numStrips": 0,
                "minTauMass": -1.0e3,
                "maxTauMass": 1.0,
                "maxChargedCands": 6,
                "maxStrips": 0,
            },
            "1Prong1Pi0": {
                "numChargedCands": 1,
                "numStrips": 1,
                "minTauMass": 0.3,
                "maxTauMass": 1.3,
                "minStripMass": -1.0e3,
                "maxStripMass": +1.0e3,
                "maxChargedCands": 6,
                "maxStrips": 6,
            },
            "1Prong2Pi0": {
                "numChargedCands": 1,
                "numStrips": 2,
                "minTauMass": 0.4,
                "maxTauMass": 1.2,
                "minStripMass": 0.05,
                "maxStripMass": 0.20,
                "maxChargedCands": 6,
                "maxStrips": 5,
            },
            "3Prong0Pi0": {
                "numChargedCands": 3,
                "numStrips": 0,
                "minTauMass": 0.8,
                "maxTauMass": 1.5,
                "maxChargedCands": 6,
                "maxStrips": 0,
            },
            "3Prong1Pi0": {
                "numChargedCands": 3,
                "numStrips": 1,
                "minTauMass": 0.9,
                "maxTauMass": 1.6,
                "minStripMass": -1.0e3,
                "maxStripMass": +1.0e3,
                "maxChargedCands": 6,
                "maxStrips": 3,
            },
        }

        self.combinatorics = CombinatoricsGenerator(verbosity)

        self.verbosity = verbosity

    def selectSignalChargedCands(self, cands):
        signalCands = []
        for cand in cands:
            if (
                (cand.pdgId == 11 and cand.pt > self.signalCand_minElectronPt)
                or (cand.pdgId == 13 and cand.pt > self.signalCand_minMuonPt)
                or (cand.pdgId == 111 and cand.pt > self.signalCand_minChargedHadronPt)
            ):
                signalCands.append(cand)
        return signalCands

    def selectIsolationCands(self, cands):
        isolationCands = []
        for cand in cands:
            if (
                (cand.pdgId == 11 and cand.pt > self.isolationCand_minElectronPt)
                or (cand.pdgId == 13 and cand.pt > self.isolationCand_minMuonPt)
                or (cand.pdgId == 22 and cand.pt > self.isolationCand_minGammaPt)
                or (cand.pdgId == 111 and cand.pt > self.isolationCand_minChargedHadronPt)
                or (cand.pdgId == 130 and cand.pt > self.isolationCand_minNeutralHadronPt)
            ):
                isolationCands.append(cand)
        return isolationCands

    def cleanCands(self, cands_to_clean, cands, dRmatch=1.0e-3):
        cleanedCands = []
        for cand_to_clean in cands_to_clean:
            isOverlap = False
            for cand in cands:
                if cand.pdgId == cand_to_clean.pdgId and cand.q == cand_to_clean.q:
                    dR = deltaR(cand, cand_to_clean)
                    if dR < dRmatch:
                        isOverlap = True
                        break
            if not isOverlap:
                cleanedCands.append(cand_to_clean)
        return cleanedCands

    def cleanStrips(self, strips, chargedCands):
        cleanedStrips = []
        for strip in strips:
            cleanedCands = self.cleanCands(strip.cands, chargedCands)
            cleanedStrip = Strip(cleanedCands, strip.barcode)
            if len(cleanedCands) > 0 and cleanedStrip.pt > self.stripAlgo.self.minStripPt:
                cleanedStrips.append(cleanedStrip)
        return cleanedStrips

    def preselectTaus(self, taus):
        preselectedTaus = []
        for tau in taus:
            if not math.abs(tau.charge) == 1:
                continue
            if not deltaR(tau, tau.jet) < self.matchingConeSize:
                continue
            if not (
                tau.mass > self.targetedDecayModes[tau.decayMode]["minMass"]
                and tau.mass < self.targetedDecayModes[tau.decayMode]["maxMass"]
            ):
                continue
            preselectedTaus.append(tau)
        return preselectedTaus

    def buildTau(self, jet, iso_cands):
        signal_chargedCands = self.selectSignalChargedCands(jet.constituents)

        signal_strips = self.stripAlgo.buildStrips(jet.constituents)
        signal_strips.sort(key=isHigherPt)

        tau_candidates = []
        barcode = 0
        for decayMode, cfgDecayMode in self.targetedDecayModes.items():
            decayMode_numChargedCands = cfgDecayMode["numChargedCands"]
            if len(signal_chargedCands) < decayMode_numChargedCands:
                continue

            selectedStrips = None
            if len(signal_strips) > 0:
                minStripMass = cfgDecayMode["minStripMass"]
                maxStripMass = cfgDecayMode["maxStripMass"]
                selectedStrips = []
                for strip in signal_strips:
                    if strip.mass > minStripMass and strip.mass < maxStripMass:
                        selectedStrips.append(strip)

            decayMode_numStrips = cfgDecayMode["numStrips"]
            if len(selectedStrips) < decayMode_numStrips:
                continue

            chargedCandCombos = self.combinatorics.generate(
                decayMode_numChargedCands, min(len(signal_chargedCands), cfgDecayMode["maxChargedCands"])
            )
            stripCombos = self.combinatorics.generate(
                decayMode_numStrips, min(len(selectedStrips)), cfgDecayMode["maxStrips"]
            )

            numChargedCandCombos = len(chargedCandCombos)
            for idxChargedCandCombo in range(numChargedCandCombos):
                chargedCandCombo = chargedCandCombos[idxChargedCandCombo]
                assert len(chargedCandCombo) == decayMode_numChargedCands
                chargedCands = [signal_chargedCands[chargedCandCombo[idx]] for idx in range(decayMode_numChargedCands)]

                numStripCombos = len(stripCombos)
                for idxStripCombo in range(max(1, numStripCombos)):
                    strips = []
                    if idxStripCombo < numStripCombos:
                        stripCombo = stripCombos[idxStripCombo]
                        assert len(stripCombo) == decayMode_numStrips
                        strips = [selectedStrips[stripCombo[idx]] for idx in range(decayMode_numStrips)]

                    cleanedStrips = self.cleanStrips(strips, chargedCands)
                    if len(cleanedStrips) < decayMode_numStrips:
                        continue
                    cleanedStrips.sort(key=isHigherPt)

                    tau_candidate = Tau(chargedCands, cleanedStrips, barcode)
                    tau_candidate.decayMode = decayMode
                    tau_candidates.append(tau_candidate)
                    barcode += 1

        tau_candidates = self.preselectTaus(tau_candidates)
        tau_candidates.sort(key=isHigherRank)

        tau = None
        if len(tau_candidates) > 0:
            tau = tau_candidates[0]

            iso_cands = self.selectCandsByDeltaR(iso_cands, tau, self.isolationConeSize)
            iso_cands = self.cleanCands(tau.cands, iso_cands)

            tau.isoCands = iso_cands
            tau.isoChargedCands = selectCandsByPdgId(iso_cands, [11, 13, 111])
            tau.isoGammaCands = selectCandsByPdgId(iso_cands, [22])
            tau.isoNeutralHadronCands = selectCandsByPdgId(iso_cands, [130])
            tau.chargedIso = compPtSum(tau.isoChargedCands)
            tau.gammaIso = compPtSum(tau.isoGammaCands)
            tau.neutralHadronIso = compPtSum(tau.isoNeutralHadronCands)
            # CV: don't use neutral hadrons when computing the isolation of the tau
            tau.combinedIso = tau.chargedIso + tau.gammaIso

            # CV: constant alpha choosen such that idDiscr varies smoothly between 0 and 1
            #     for typical values of the combined isolation pT-sum
            alpha = 0.2
            tau.idDiscr = math.exp(-alpha * tau.combinedIso)

        return tau
