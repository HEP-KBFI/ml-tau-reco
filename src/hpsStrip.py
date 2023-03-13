import math
import vector
import numpy as np

m_pi0 = 0.135


class Strip:
    def __init__(self, cands=[], barcode=-1):
        cands_p4 = [c.p4 for c in cands]
        cands_p4 = np.array([[v.px, v.py, v.pz, v.E] for v in cands_p4])
        if len(cands_p4) == 0:
            sum_p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(cands_p4, axis=0)
            sum_p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])
        strip_px = sum_p4.px
        strip_py = sum_p4.py
        strip_pz = sum_p4.pz
        strip_E = math.sqrt(strip_px * strip_px + strip_py * strip_py + strip_pz * strip_pz + m_pi0 * m_pi0)
        self.p4 = vector.obj(px=strip_px, py=strip_py, pz=strip_pz, E=strip_E)
        self.updatePtEtaPhiMass()
        self.cands = set(cands)
        self.barcode = barcode

    def updatePtEtaPhiMass(self):
        self.energy = self.p4.energy
        self.pt = self.p4.pt
        self.theta = self.p4.theta
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

    def print(self):
        print(
            "strip #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f"
            % (self.barcode, self.energy, self.pt, self.eta, self.phi, self.mass)
        )
        for cand in self.cands:
            cand.print()
