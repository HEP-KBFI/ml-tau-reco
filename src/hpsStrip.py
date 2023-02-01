import vector


class Strip:
    def __init__(self, cands=[], barcode=-1):
        self.p4 = vector(pt=0.0, phi=0.0, theta=0.0, mass=0.0)
        for cand in cands:
            self.p4 += cand.p4
        self.updatePtEtaPhiMass()
        self.cands = set(cands)
        self.barcode = barcode

    def updatePtEtaPhiMass(self):
        self.pt = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass
