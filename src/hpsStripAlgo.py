import math
import vector

from hpsGetParameter import getParameter


class StripAlgo:
    def __init__(self, cfg, verbosity=0):
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

        self.verbosity = verbosity

    def updateStripP4(strip):
        strip.p4 = vector(pt=0.0, phi=0.0, theta=0.0, mass=0.0)
        for cand in strip.cands:
            strip.p4 += cand.p4
        strip.updatePtEtaPhiMass()

    def addCandsToStrip(strip, cands, candBarcodesPreviousStrips, candBarcodesCurrentStrip):
        isCandAdded = False
        for cand in cands:
            if not (cand.barcode in candBarcodesPreviousStrips or cand.barcode in candBarcodesCurrentStrip):
                dEta = math.fabs(cand.eta - strip.eta)
                dPhi = math.fabs(cand.phi - strip.phi)
                if dEta < self.maxStripSizeEta and dPhi < self.maxStripSizePhi:
                    strip.cands.add(cand)
                    if self.updateStripAfterEachCand:
                        updateStripP4(strip)
                    candBarcodesCurrentStrip.add(cand.barcode)
                    isCandAdded = True
        return isCandAdded

    def markCandsInStrip(candBarcodesPreviousStrips, candBarcodesCurrentStrip):

        candBarcodesPreviousStrips.update(candBarcodesCurrentStrip)

    def buildStrips(cands):
        seedCands = []
        addCands = []
        for cand in cands:
            if (cand.pdgId == 22 and self.useGammas) or (cand.pdgId == 11 and self.useElectrons):
                if cand.pt > self.minGammaPtSeed:
                    seedCands.append(cand)
                elif cand.pt > self.minGammaPtAdd:
                    addCands.append(cand)

        output_strips = []

        seedCandBarcodesPreviousStrips = set()
        addCandBarcodesPreviousStrips = set()

        idxStrip = 0
        for seedCand in seedCands:
            if not seedCand.barcode in seedCandFlags:
                currentStrip = Strip()

                seedCandBarcodesCurrentStrip = set()
                addCandBarcodesCurrentStrip = set()

                stripBuildIterations = 0
                while stripBuildIterations < self.maxStripBuildIterations or self.maxStripBuildIterations == -1:
                    isCandAdded = False
                    isCandAdded |= addCandsToStrip(
                        currentStrip, seedCands, seedCandBarcodesPreviousStrips, seedCandBarcodesCurrentStrip
                    )
                    isCandAdded |= addCandsToStrip(
                        currentStrip, addCands, addCandBarcodesPreviousStrips, addCandBarcodesCurrentStrip
                    )
                    if not self.updateStripAfterEachCand:
                        updateStripP4(currentStrip)
                    if not isCandAdded:
                        break
                    ++stripBuildIterations

                if currentStrip.pt > self.minStripPt:
                    currentStrip.barcode = idxStrip
                    ++idxStrip
                    output_strips.append(currentStrip)
                    markCandsInStrip(seedCandBarcodesPreviousStrips, seedCandBarcodesCurrentStrip)
                    markCandsInStrip(addCandBarcodesPreviousStrips, addCandBarcodesCurrentStrip)

        return output_strips
