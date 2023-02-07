import vector
import numpy as np


class Tau:
    def __init__(self, chargedCands=[], strips=[], barcode=-1):
        cands_and_strips = [c.p4 for c in chargedCands] + [s.p4 for s in strips]
        cands_and_strips = np.array([[v.px, v.py, v.pz, v.E] for v in cands_and_strips])
        if len(cands_and_strips) == 0:
            self.p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(cands_and_strips, axis=0)
            self.p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])

        self.updatePtEtaPhiMass()
        self.q = 0.0
        for chargedCand in chargedCands:
            if chargedCand.q > 0.0:
                self.q += 1.0
            elif chargedCand.q < 0.0:
                self.q -= 1.0
            else:
                assert 0
        self.signal_chargedCands = chargedCands
        self.signal_strips = strips
        self.updateSignalCands()
        self.idDiscr = -1.0
        self.decayMode = "undefined"
        self.iso_cands = set()
        self.chargedIso = -1.0
        self.gammaIso = -1.0
        self.neutralHadronIso = -1.0
        self.combinedIso = -1.0
        self.barcode = barcode

    def updateSignalCands(self):
        self.num_signal_chargedCands = len(self.signal_chargedCands)
        self.num_signal_strips = len(self.signal_strips)
        self.signal_cands = set()
        self.signal_cands.update(self.signal_chargedCands)
        for strip in self.signal_strips:
            for cand in strip.cands:
                self.signal_cands.add(cand)

    def updatePtEtaPhiMass(self):
        self.pt = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

    def print(self):
        print(
            "tau #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, idDiscr = %1.3f, decayMode = %s"
            % (self.barcode, self.pt, self.eta, self.phi, self.mass, self.idDiscr, self.decayMode)
        )
        # print("signal_chargedCands:")
        # for cand in self.signal_chargedCands:
        #    cand.print()
        # print("signal_strips:")
        # for strip in self.signal_strips:
        #    strip.print()
        print("signal_cands:")
        for cand in self.signal_cands:
            cand.print()
        print("iso_cands:")
        for cand in self.iso_cands:
            cand.print()
        print(
            " isolation: charged = %1.2f, gamma = %1.2f, neutralHadron = %1.2f, combined = %1.2f"
            % (self.chargedIso, self.gammaIso, self.neutralHadronIso, self.combinedIso)
        )
