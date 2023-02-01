import vector


class Strip:
    def __init__(self, barcode=-1):
        self.p4 = vector(pt=0.0, phi=0.0, theta=0.0, mass=0.0)
        updatePtEtaPhiMass()
        self.cands = set()
        self.barcode = barcode

    def __init__(self, cands, barcode=-1):
        self.p4 = vector(pt=0.0, phi=0.0, theta=0.0, mass=0.0)
        for cand in cands:
            self.p4 += cand.p4
        updatePtEtaPhiMass()
        self.cands = set(cands)
        self.barcode = barcode

    def updatePtEtaPhiMass():
        self.pt = p4.pt
        self.eta = p4.eta
        self.phi = p4.phi
        self.mass = p4.mass
