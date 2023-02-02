from functools import cmp_to_key
import math

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
        if cand.abs_pdgId in pdgIds:
            selectedCands.append(cand)
    return selectedCands


def rank_tau_candidates(tau1, tau2):
    if tau1.numSignalChargedCands > tau2.numSignalChargedCands:
        return +1
    if tau1.numSignalChargedCands < tau2.numSignalChargedCands:
        return -1
    if tau1.pt > tau2.pt:
        return +1
    if tau1.pt < tau2.pt:
        return -1
    if tau1.numSignalStrips > tau2.numSignalStrips:
        return +1
    if tau1.numSignalStrips < tau2.numSignalStrips:
        return -1
    if tau1.combinedIso < tau2.combinedIso:
        return +1
    if tau1.combinedIso > tau2.combinedIso:
        return -1
    return 0


def compPtSum(cands):
    pt_sum = 0.0
    for cand in cands:
        pt_sum += cand.pt
    return pt_sum


class HPSAlgo:
    def __init__(self, cfg, verbosity=0):
        if verbosity >= 1:
            print("<HPSAlgo::HPSAlgo>:")

        cfgSignalCands = cfg["signalCands"]
        self.signalCand_minChargedHadronPt = getParameter(cfgSignalCands, "minChargedHadronPt", 0.5)
        self.signalCand_minElectronPt = getParameter(cfgSignalCands, "minElectronPt", 0.5)
        # CV: don't use muons when building the signal constituents of the tau
        self.signalCand_minMuonPt = getParameter(cfgSignalCands, "minChargedHadronPt", 1.0e6)
        if verbosity >= 1:
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
        if verbosity >= 1:
            print("isolationCands:")
            print(" minChargedHadronPt = %1.2f" % self.isolationCand_minChargedHadronPt)
            print(" minElectronPt = %1.2f" % self.isolationCand_minElectronPt)
            print(" minGammaPt = %1.2f" % self.isolationCand_minGammaPt)
            print(" minMuonPt = %1.2f" % self.isolationCand_minMuonPt)
            print(" minNeutralHadronPt = %1.2f" % self.isolationCand_minNeutralHadronPt)

        self.matchingConeSize = getParameter(cfg, "matchingConeSize", 1.0e-1)
        self.isolationConeSize = getParameter(cfg, "isolationConeSize", 5.0e-1)
        if verbosity >= 1:
            print("matchingConeSize = %1.2f" % self.matchingConeSize)
            print("isolationConeSize = %1.2f" % self.isolationConeSize)

        self.stripAlgo = StripAlgo(cfg["StripAlgo"], verbosity)

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
                (cand.abs_pdgId == 11 and cand.pt > self.signalCand_minElectronPt)
                or (cand.abs_pdgId == 13 and cand.pt > self.signalCand_minMuonPt)
                or (cand.abs_pdgId == 211 and cand.pt > self.signalCand_minChargedHadronPt)
            ):
                signalCands.append(cand)
        return signalCands

    def selectIsolationCands(self, cands):
        isolationCands = []
        for cand in cands:
            if (
                (cand.abs_pdgId == 11 and cand.pt > self.isolationCand_minElectronPt)
                or (cand.abs_pdgId == 13 and cand.pt > self.isolationCand_minMuonPt)
                or (cand.abs_pdgId == 22 and cand.pt > self.isolationCand_minGammaPt)
                or (cand.abs_pdgId == 211 and cand.pt > self.isolationCand_minChargedHadronPt)
                or (cand.abs_pdgId in [130, 2112] and cand.pt > self.isolationCand_minNeutralHadronPt)
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
            if len(cleanedCands) > 0 and cleanedStrip.pt > self.stripAlgo.minStripPt:
                cleanedStrips.append(cleanedStrip)
        return cleanedStrips

    def preselectTaus(self, taus):
        preselectedTaus = []
        for tau in taus:
            if not abs(tau.q) == 1:
                continue
            if not deltaR(tau, tau.jet) < self.matchingConeSize:
                continue
            if not (
                tau.mass > self.targetedDecayModes[tau.decayMode]["minTauMass"]
                and tau.mass < self.targetedDecayModes[tau.decayMode]["maxTauMass"]
            ):
                continue
            preselectedTaus.append(tau)
        return preselectedTaus

    def buildTau(self, jet, iso_cands):
        if self.verbosity >= 2:
            print("<hpsAlgo::buildTau>:")

        signal_chargedCands = self.selectSignalChargedCands(jet.constituents)

        signal_strips = self.stripAlgo.buildStrips(jet.constituents)
        # CV: reverse=True argument needed in order to sort strips in order of decreasing (and NOT increasing) pT
        signal_strips.sort(key=lambda cand: cand.pt, reverse=True)

        if self.verbosity >= 2:
            print("#signal_chargedCands = %i" % len(signal_chargedCands))
            print("#signal_strips = %i" % len(signal_strips))
            print("#iso_cands = %i" % len(iso_cands))

        tau_candidates = []
        barcode = 0
        for decayMode, cfgDecayMode in self.targetedDecayModes.items():
            if self.verbosity >= 4:
                print(
                    "decayMode = %s: numChargedCands = %i, numStrips = %i"
                    % (decayMode, cfgDecayMode["numChargedCands"], cfgDecayMode["numStrips"])
                )

            decayMode_numChargedCands = cfgDecayMode["numChargedCands"]
            if len(signal_chargedCands) < decayMode_numChargedCands:
                continue

            decayMode_numStrips = cfgDecayMode["numStrips"]
            selectedStrips = []
            if decayMode_numStrips > 0 and len(signal_strips) > 0:
                minStripMass = cfgDecayMode["minStripMass"]
                maxStripMass = cfgDecayMode["maxStripMass"]
                for strip in signal_strips:
                    if strip.mass > minStripMass and strip.mass < maxStripMass:
                        selectedStrips.append(strip)
            if self.verbosity >= 4:
                print("selectedStrips = %i" % len(selectedStrips))
            if len(selectedStrips) < decayMode_numStrips:
                continue

            chargedCandCombos = self.combinatorics.generate(
                decayMode_numChargedCands, min(len(signal_chargedCands), cfgDecayMode["maxChargedCands"])
            )
            if self.verbosity >= 4:
                print("chargedCandCombos = %s" % chargedCandCombos)
            stripCombos = self.combinatorics.generate(
                decayMode_numStrips, min(len(selectedStrips), cfgDecayMode["maxStrips"])
            )
            if self.verbosity >= 4:
                print("stripCombos = %s" % stripCombos)

            numChargedCandCombos = len(chargedCandCombos)
            for idxChargedCandCombo in range(numChargedCandCombos):
                chargedCandCombo = chargedCandCombos[idxChargedCandCombo]
                assert len(chargedCandCombo) == decayMode_numChargedCands
                chargedCands = [signal_chargedCands[chargedCandCombo[idx]] for idx in range(decayMode_numChargedCands)]

                numStripCombos = len(stripCombos)
                for idxStripCombo in range(max(1, numStripCombos)):
                    stripCombo = []
                    strips = []
                    if idxStripCombo < numStripCombos:
                        stripCombo = stripCombos[idxStripCombo]
                        assert len(stripCombo) == decayMode_numStrips
                        strips = [selectedStrips[stripCombo[idx]] for idx in range(decayMode_numStrips)]
                    if self.verbosity >= 4:
                        print("Processing combination of chargedCands = %s & strips = %s" % (chargedCandCombo, stripCombo))

                    cleanedStrips = self.cleanStrips(strips, chargedCands)
                    if self.verbosity >= 4:
                        print("#cleanedStrips = %i" % len(cleanedStrips))
                    if len(cleanedStrips) < decayMode_numStrips:
                        continue
                    # CV: reverse=True argument needed in order to sort strips in order of decreasing (and NOT increasing) pT
                    cleanedStrips.sort(key=lambda strip: strip.pt, reverse=True)

                    tau_candidate = Tau(chargedCands, cleanedStrips, barcode)
                    tau_candidate.jet = jet
                    tau_candidate.decayMode = decayMode
                    if self.verbosity >= 4:
                        print("tau_candidate:")
                        tau_candidate.print()
                    tau_candidates.append(tau_candidate)
                    barcode += 1

        if self.verbosity >= 4:
            print("#tau_candidates (before preselection) = %i" % len(tau_candidates))
        tau_candidates = self.preselectTaus(tau_candidates)
        if self.verbosity >= 4:
            print("#tau_candidates (after preselection) = %i" % len(tau_candidates))
        for tau_candidate in tau_candidates:
            tau_iso_cands = selectCandsByDeltaR(iso_cands, tau_candidate, self.isolationConeSize)
            print("#tau_iso_cands@1 = %i" % len(tau_iso_cands))
            tau_iso_cands = self.selectIsolationCands(tau_iso_cands)
            print("#tau_iso_cands@2 = %i" % len(tau_iso_cands))
            tau_iso_cands = self.cleanCands(tau_iso_cands, tau_candidate.signalCands)
            print("#tau_iso_cands@3 = %i" % len(tau_iso_cands))

            tau_candidate.isoCands = tau_iso_cands
            tau_candidate.isoChargedCands = selectCandsByPdgId(tau_iso_cands, [11, 13, 211])
            tau_candidate.isoGammaCands = selectCandsByPdgId(tau_iso_cands, [22])
            tau_candidate.isoNeutralHadronCands = selectCandsByPdgId(tau_iso_cands, [130, 2112])
            tau_candidate.chargedIso = compPtSum(tau_candidate.isoChargedCands)
            tau_candidate.gammaIso = compPtSum(tau_candidate.isoGammaCands)
            tau_candidate.neutralHadronIso = compPtSum(tau_candidate.isoNeutralHadronCands)
            # CV: don't use neutral hadrons when computing the isolation of the tau
            tau_candidate.combinedIso = tau_candidate.chargedIso + tau_candidate.gammaIso

            # CV: constant alpha choosen such that idDiscr varies smoothly between 0 and 1
            #     for typical values of the combined isolation pT-sum
            alpha = 0.2
            tau_candidate.idDiscr = math.exp(-alpha * tau_candidate.combinedIso)
            if self.verbosity >= 3:
                tau_candidate.print()
        # CV: sort tau candidates by multiplicity of charged signal candidates,
        #     pT, multiplicity of strips, and combined isolation (in that order);
        #     reverse=True argument needed in order to sort tau candidates in order of decreasing (and NOT increasing) rank
        tau_candidates.sort(key=cmp_to_key(rank_tau_candidates), reverse=True)

        tau = None
        if len(tau_candidates) > 0:
            tau = tau_candidates[0]

        return tau
