import vector

class Tau:
    def __init__(self, chargedCands=[], strips=[], barcode=-1):
        self.p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for chargedCand in chargedCands:
            self.p4 = self.p4 + chargedCand.p4
        for strip in strips:
            self.p4 = self.p4 + strip.p4
        self.updatePtEtaPhiMass()
        self.q = 0.0
        for chargedCand in chargedCands:
            if chargedCand.q > 0.0:
                self.q += 1.0
            elif chargedCand.q < 0.0:
                self.q -= 1.0
            else:
                assert 0
        self.signalChargedCands = chargedCands
        self.signalStrips = strips
        self.updateSignalCands()
        self.idDiscr = -1.
        self.decayMode = "undefined"
        self.isoCands = set()
        self.chargedIso = -1.
        self.gammaIso = -1.
        self.neutralHadronIso = -1.
        self.combinedIso = -1.
        self.barcode = barcode

    def updateSignalCands(self):
        self.numSignalChargedCands = len(self.signalChargedCands)
        self.numSignalStrips = len(self.signalStrips)
        self.signalCands = set()
        self.signalCands.update(self.signalChargedCands)
        for strip in self.signalStrips:
            for cand in strip.cands:
                self.signalCands.add(cand)

    def updatePtEtaPhiMass(self):
        self.pt = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

    def print(self):
        print("tau #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, idDiscr = %1.3f, decayMode = %s" % \
          (self.barcode, self.pt, self.eta, self.phi, self.mass, self.idDiscr, self.decayMode))
        #for cand in self.signalChargedCands:
        #    print(" signalChargedCand #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, pdgId = %i, charge = %1.1f" % \
        #      (cand.barcode, cand.pt, cand.eta, cand.phi, cand.pdgId, cand.q))
        #for strip in self.signalStrips:
        #    print(" signalStrip #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f" % \
        #      (strip.barcode, strip.pt, strip.eta, strip.phi, strip.mass))
        for cand in self.signalCands:
            print(" signalCand #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, pdgId = %i, charge = %1.1f" % \
              (cand.barcode, cand.pt, cand.eta, cand.phi, cand.pdgId, cand.q))
        for cand in self.isoCands:
            print(" isoCand #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, pdgId = %i, charge = %1.1f" % \
              (cand.barcode, cand.pt, cand.eta, cand.phi, cand.pdgId, cand.q))
        print(" isolation: charged = %1.2f, gamma = %1.2f, neutralHadron = %1.2f, combined = %1.2f" % \
          (self.chargedIso, self.gammaIso, self.neutralHadronIso, self.combinedIso))
