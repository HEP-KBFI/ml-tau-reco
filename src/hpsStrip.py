import math
import vector
import numpy as np

from hpsParticleBase import hpsParticleBase

m_pi0 = 0.135


class Strip(hpsParticleBase):
    def __init__(self, cands=[], barcode=-1):
        cand_p4s = [cand.p4 for cand in cands]
        cand_p4s = np.array([[p4.px, p4.py, p4.pz, p4.E] for p4 in cand_p4s])
        sum_p4 = None
        if len(cand_p4s) == 0:
            sum_p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(cand_p4s, axis=0)
            sum_p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])
        strip_px = sum_p4.px
        strip_py = sum_p4.py
        strip_pz = sum_p4.pz
        strip_E = math.sqrt(strip_px * strip_px + strip_py * strip_py + strip_pz * strip_pz + m_pi0 * m_pi0)
        strip_p4 = vector.obj(px=strip_px, py=strip_py, pz=strip_pz, E=strip_E)
        super().__init__(p4=strip_p4, barcode=barcode)
        self.cands = set(cands)

    def print(self):
        print(
            "strip #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f"
            % (self.barcode, self.energy, self.pt, self.eta, self.phi, self.mass)
        )
        for cand in self.cands:
            cand.print()
