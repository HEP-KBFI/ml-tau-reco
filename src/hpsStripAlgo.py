import vector

from hpsGetParameter import getParameter
from hpsStrip import Strip
from hpsAlgoTools import comp_deltaPhi


class StripAlgo:
    def __init__(self, cfg, metric_dEta_or_dTheta, verbosity=0):
        if verbosity >= 1:
            print("<StripAlgo::StripAlgo>:")
        self.useGammas = getParameter(cfg, "useGammas", True)
        self.minGammaPtSeed = getParameter(cfg, "minGammaPtSeed", 1.0)
        self.minGammaPtAdd = getParameter(cfg, "minGammaPtAdd", 1.0)
        self.useElectrons = getParameter(cfg, "useElectrons", True)
        self.minElectronPtSeed = getParameter(cfg, "minElectronPtSeed", 0.5)
        self.minElectronPtAdd = getParameter(cfg, "minElectronPtAdd", 0.5)
        self.minStripPt = getParameter(cfg, "minStripPt", 1.0)
        self.updateStripAfterEachCand = getParameter(cfg, "updateStripAfterEachCand", False)
        self.maxStripBuildIterations = getParameter(cfg, "maxStripBuildIterations", -1)
        self.maxStripSizeEta = getParameter(cfg, "maxStripSizeEta", 0.05)
        self.maxStripSizePhi = getParameter(cfg, "maxStripSizePhi", 0.20)
        if verbosity >= 1:
            print(" useGammas = %s" % self.useGammas)
            print(" minGammaPtSeed = %1.2f" % self.minGammaPtSeed)
            print(" minGammaPtAdd = %1.2f" % self.minGammaPtAdd)
            print(" useElectrons = %s" % self.useElectrons)
            print(" minElectronPtSeed = %1.2f" % self.minElectronPtSeed)
            print(" minElectronPtAdd = %1.2f" % self.minElectronPtAdd)
            print(" minStripPt = %1.2f" % self.minStripPt)
            print(" updateStripAfterEachCand = %s" % self.updateStripAfterEachCand)
            print(" maxStripBuildIterations = %i" % self.maxStripBuildIterations)
            print(" maxStripSizeEta = %1.2f" % self.maxStripSizeEta)
            print(" maxStripSizePhi = %1.2f" % self.maxStripSizePhi)

        self.metric_dEta_or_dTheta = metric_dEta_or_dTheta

        self.verbosity = verbosity

    def updateStripP4(self, strip):
        strip.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for cand in strip.cands:
            strip.p4 = strip.p4 + cand.p4
        strip.updatePtEtaPhiMass()

    def addCandsToStrip(self, strip, cands, candBarcodesPreviousStrips, candBarcodesCurrentStrip):
        isCandAdded = False
        for cand in cands:
            if not (cand.barcode in candBarcodesPreviousStrips or cand.barcode in candBarcodesCurrentStrip):
                dEta = self.metric_dEta_or_dTheta(cand, strip)
                dPhi = comp_deltaPhi(cand, strip)
                if dEta < self.maxStripSizeEta and dPhi < self.maxStripSizePhi:
                    strip.cands.add(cand)
                    if self.updateStripAfterEachCand:
                        self.updateStripP4(strip)
                    candBarcodesCurrentStrip.add(cand.barcode)
                    isCandAdded = True
        return isCandAdded

    def markCandsInStrip(self, candBarcodesPreviousStrips, candBarcodesCurrentStrip):
        candBarcodesPreviousStrips.update(candBarcodesCurrentStrip)

    def buildStrips(self, cands):
        if self.verbosity >= 3:
            print("<StripAlgo::buildStrips>:")
        seedCands = []
        addCands = []
        for cand in cands:
            if (cand.abs_pdgId == 22 and self.useGammas) or (cand.abs_pdgId == 11 and self.useElectrons):
                minPtSeed = None
                minPtAdd = None
                if cand.abs_pdgId == 22:
                    minPtSeed = self.minGammaPtSeed
                    minPtAdd = self.minGammaPtAdd
                elif cand.abs_pdgId == 11:
                    minPtSeed = self.minElectronPtSeed
                    minPtAdd = self.minElectronPtAdd
                else:
                    assert 0
                if cand.pt > minPtSeed:
                    seedCands.append(cand)
                elif cand.pt > minPtAdd:
                    addCands.append(cand)
        if self.verbosity >= 3:
            print("seedCands:")
            for cand in seedCands:
                cand.print()
            print("#seedCands = %i" % len(seedCands))
            print("addCands:")
            for cand in addCands:
                cand.print()
            print("#addCands = %i" % len(addCands))

        output_strips = []

        seedCandBarcodesPreviousStrips = set()
        addCandBarcodesPreviousStrips = set()

        idxStrip = 0
        for seedCand in seedCands:
            if self.verbosity >= 4:
                print("Processing seedCand #%i" % seedCand.barcode)
            if seedCand.barcode not in seedCandBarcodesPreviousStrips:
                currentStrip = Strip([seedCand], idxStrip)

                seedCandBarcodesCurrentStrip = set([seedCand.barcode])
                addCandBarcodesCurrentStrip = set()

                stripBuildIterations = 0
                while stripBuildIterations < self.maxStripBuildIterations or self.maxStripBuildIterations == -1:
                    isCandAdded = False
                    isCandAdded |= self.addCandsToStrip(
                        currentStrip, seedCands, seedCandBarcodesPreviousStrips, seedCandBarcodesCurrentStrip
                    )
                    isCandAdded |= self.addCandsToStrip(
                        currentStrip, addCands, addCandBarcodesPreviousStrips, addCandBarcodesCurrentStrip
                    )
                    if not self.updateStripAfterEachCand:
                        self.updateStripP4(currentStrip)
                    if not isCandAdded:
                        break
                    ++stripBuildIterations

                if self.verbosity >= 4:
                    print("currentStrip:")
                    currentStrip.print()

                if currentStrip.pt > self.minStripPt:
                    currentStrip.barcode = idxStrip
                    ++idxStrip
                    output_strips.append(currentStrip)
                    self.markCandsInStrip(seedCandBarcodesPreviousStrips, seedCandBarcodesCurrentStrip)
                    self.markCandsInStrip(addCandBarcodesPreviousStrips, addCandBarcodesCurrentStrip)

        if self.verbosity >= 4:
            print("output_strips:")
            for strip in output_strips:
                strip.print()

        return output_strips
