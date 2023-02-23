from functools import cmp_to_key
import math

from hpsAlgoTools import (
    comp_angle,
    comp_deltaEta,
    comp_deltaTheta,
    comp_deltaR,
    comp_pt_sum,
    selectCandsByDeltaR,
    selectCandsByPdgId,
)
from hpsCombinatoricsGenerator import CombinatoricsGenerator
from hpsGetParameter import getParameter
from hpsStrip import Strip
from hpsStripAlgo import StripAlgo
from hpsTau import Tau


def rank_tau_candidates(tau1, tau2):
    if tau1.num_signal_chargedCands > tau2.num_signal_chargedCands:
        return +1
    if tau1.num_signal_chargedCands < tau2.num_signal_chargedCands:
        return -1
    if tau1.pt > tau2.pt:
        return +1
    if tau1.pt < tau2.pt:
        return -1
    if tau1.num_signal_strips > tau2.num_signal_strips:
        return +1
    if tau1.num_signal_strips < tau2.num_signal_strips:
        return -1
    if tau1.combinedIso_dR0p5 < tau2.combinedIso_dR0p5:
        return +1
    if tau1.combinedIso_dR0p5 > tau2.combinedIso_dR0p5:
        return -1
    return 0


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
        metric = getParameter(cfg, "metric", "theta-phi")
        self.metric_dR_or_angle = None
        self.metric_dEta_or_dTheta = None
        if metric == "eta-phi":
            self.metric_dR_or_angle = comp_deltaR
            self.metric_dEta_or_dTheta = comp_deltaEta
        elif metric == "theta-phi":
            self.metric_dR_or_angle = comp_angle
            self.metric_dEta_or_dTheta = comp_deltaTheta
        else:
            raise RuntimeError("Invalid configuration parameter 'metric' = '%s' !!" % metric)
        if verbosity >= 1:
            print("matchingConeSize = %1.2f" % self.matchingConeSize)
            print("isolationConeSize = %1.2f" % self.isolationConeSize)
            print("metric = '%s'" % metric)

        self.stripAlgo = StripAlgo(cfg["StripAlgo"], verbosity)

        self.targetedDecayModes = cfg["decayModes"]
        if verbosity >= 1:
            print("targetedDecayModes:")
            for decayMode in ["1Prong0Pi0", "1Prong1Pi0", "1Prong2Pi0", "3Prong0Pi0", "3Prong1Pi0"]:
                if decayMode in self.targetedDecayModes.keys():
                    print(" %s:" % decayMode)
                    targetedDecayMode = self.targetedDecayModes[decayMode]
                    print("  numChargedCands = %i" % targetedDecayMode["numChargedCands"])
                    print("  numStrips = %i" % targetedDecayMode["numStrips"])
                    print(
                        "  tauMass: min = %1.2f, max = %1.2f"
                        % (targetedDecayMode["minTauMass"], targetedDecayMode["maxTauMass"])
                    )
                    if targetedDecayMode["numStrips"] > 0:
                        print(
                            "  stripMass: min = %1.2f, max = %1.2f"
                            % (targetedDecayMode["minStripMass"], targetedDecayMode["maxStripMass"])
                        )
                    print("  maxChargedCands = %i" % targetedDecayMode["maxChargedCands"])
                    print("  maxStrips = %i" % targetedDecayMode["maxStrips"])

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
                    dR = self.metric_dR_or_angle(cand, cand_to_clean)
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

    def buildTau(self, jet, event_iso_cands):
        if self.verbosity >= 2:
            print("<hpsAlgo::buildTau>:")

        signal_chargedCands = self.selectSignalChargedCands(jet.constituents)
        if self.verbosity >= 2:
            print("#signal_chargedCands = %i" % len(signal_chargedCands))
            if self.verbosity >= 3:
                print("signal_chargedCands")
                for cand in signal_chargedCands:
                    cand.print()

        signal_strips = self.stripAlgo.buildStrips(jet.constituents)
        # CV: reverse=True argument needed in order to sort strips in order of decreasing (and NOT increasing) pT
        signal_strips.sort(key=lambda cand: cand.pt, reverse=True)
        if self.verbosity >= 2:
            print("#signal_strips = %i" % len(signal_strips))
            if self.verbosity >= 3:
                print("signal_strips")
                for strip in signal_strips:
                    strip.print()

        jet_iso_cands = self.selectIsolationCands(jet.constituents)
        if self.verbosity >= 2:
            print("#jet_iso_cands = %i" % len(jet_iso_cands))
            if self.verbosity >= 3:
                print("jet_iso_cands:")
                for cand in jet_iso_cands:
                    cand.print()

        event_iso_cands = self.selectIsolationCands(event_iso_cands)
        event_iso_cands = selectCandsByDeltaR(
            event_iso_cands, jet, self.isolationConeSize + self.matchingConeSize, self.metric_dR_or_angle
        )
        event_iso_cands = self.cleanCands(event_iso_cands, jet.constituents)
        if self.verbosity >= 2:
            print("#event_iso_cands = %i" % len(event_iso_cands))
            print("event_iso_cands:")
            for cand in event_iso_cands:
                cand.print()

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
                    tau_candidate.signalConeSize = max(min(0.10, 3.0 / tau_candidate.pt), 0.05)
                    tau_candidate.metric_dR_or_angle = self.metric_dR_or_angle
                    tau_candidate.metric_dEta_or_dTheta = self.metric_dEta_or_dTheta
                    passesSignalCone = True
                    for cand in tau_candidate.signal_chargedCands:
                        if tau_candidate.metric_dR_or_angle(tau_candidate, cand) > tau_candidate.signalConeSize:
                            passesSignalCone = False
                            break
                    for strip in tau_candidate.signal_strips:
                        if tau_candidate.metric_dR_or_angle(tau_candidate, strip) > tau_candidate.signalConeSize:
                            passesSignalCone = False
                            break
                    if (
                        abs(round(tau_candidate.q)) == 1
                        and tau_candidate.metric_dR_or_angle(tau_candidate, tau_candidate.jet) < self.matchingConeSize
                        and passesSignalCone
                        and tau_candidate.mass > self.targetedDecayModes[decayMode]["minTauMass"]
                        and tau_candidate.mass < self.targetedDecayModes[decayMode]["maxTauMass"]
                    ):
                        tau_iso_cands = selectCandsByDeltaR(
                            jet_iso_cands, tau_candidate, self.isolationConeSize, tau_candidate.metric_dR_or_angle
                        )
                        tau_iso_cands = self.cleanCands(tau_iso_cands, tau_candidate.signal_cands)
                        tau_iso_cands.extend(event_iso_cands)

                        tau_candidate.iso_cands = tau_iso_cands
                        tau_candidate.iso_chargedCands = selectCandsByPdgId(tau_iso_cands, [11, 13, 211])
                        tau_candidate.iso_gammaCands = selectCandsByPdgId(tau_iso_cands, [22])
                        tau_candidate.iso_neutralHadronCands = selectCandsByPdgId(tau_iso_cands, [130, 2112])
                        tau_candidate.chargedIso_dR0p5 = comp_pt_sum(tau_candidate.iso_chargedCands)
                        tau_candidate.gammaIso_dR0p5 = comp_pt_sum(tau_candidate.iso_gammaCands)
                        tau_candidate.neutralHadronIso_dR0p5 = comp_pt_sum(tau_candidate.iso_neutralHadronCands)
                        # CV: don't use neutral hadrons when computing the isolation of the tau
                        tau_candidate.combinedIso_dR0p5 = tau_candidate.chargedIso_dR0p5 + tau_candidate.gammaIso_dR0p5

                        # CV: constant alpha choosen such that idDiscr varies smoothly between 0 and 1
                        #     for typical values of the combined isolation pT-sum
                        alpha = 0.2
                        tau_candidate.idDiscr = math.exp(-alpha * tau_candidate.combinedIso_dR0p5)

                        if self.verbosity >= 3:
                            tau_candidate.print()
                        tau_candidates.append(tau_candidate)
                        barcode += 1
                    else:
                        if self.verbosity >= 4:
                            print("fails preselection:")
                            print(" q = %i" % round(tau_candidate.q))
                            print(
                                " dR(tau,jet) = %1.2f" % tau_candidate.metric_dR_or_angle(tau_candidate, tau_candidate.jet)
                            )
                            print(" signalConeSize = %1.2f" % tau_candidate.signalConeSize)
                            for idx, cand in enumerate(tau_candidate.signal_chargedCands):
                                print(
                                    " dR(tau,signal_chargedCand #%i) = %1.2f"
                                    % (idx, tau_candidate.metric_dR_or_angle(tau_candidate, cand))
                                )
                            for idx, strip in enumerate(tau_candidate.signal_chargedCands):
                                print(
                                    " dR(tau,signal_strip #%i) = %1.2f"
                                    % (idx, tau_candidate.metric_dR_or_angle(tau_candidate, strip))
                                )
                            print(" mass = %1.2f" % tau_candidate.mass)

        # CV: sort tau candidates by multiplicity of charged signal candidates,
        #     pT, multiplicity of strips, and combined isolation (in that order);
        #     reverse=True argument needed in order to sort tau candidates in order of decreasing (and NOT increasing) rank
        tau_candidates.sort(key=cmp_to_key(rank_tau_candidates), reverse=True)
        if self.verbosity >= 2:
            print("#tau_candidates = %i" % len(tau_candidates))

        tau = None
        if len(tau_candidates) > 0:
            tau = tau_candidates[0]

        return tau
