import math
import vector

m_pi0 = 0.135


class Strip:
    def __init__(self, cands=[], barcode=-1):
        sum_p4 = vector.obj(px=0.0, py=0.0, pz=0.0, E=0.0)
        for cand in cands:
            sum_p4 = sum_p4 + cand.p4
        strip_px = sum_p4.px
        strip_py = sum_p4.py
        strip_pz = sum_p4.pz
        strip_E = math.sqrt(strip_px * strip_px + strip_py * strip_py + strip_pz * strip_pz + m_pi0 * m_pi0)
        self.p4 = vector.obj(px=strip_px, py=strip_py, pz=strip_pz, E=strip_E)
        self.updatePtEtaPhiMass()
        self.cands = set(cands)
        self.barcode = barcode

    def updatePtEtaPhiMass(self):
        self.pt = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

    def print(self):
        print(
            "strip #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f"
            % (self.barcode, self.p4.energy, self.pt, self.eta, self.phi, self.p4.mass)
        )
        for cand in self.cands:
            print(
                " cand #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, pdgId = %i, charge = %1.1f"
                % (cand.barcode, cand.pt, cand.eta, cand.phi, cand.pdgId, cand.q)
            )
