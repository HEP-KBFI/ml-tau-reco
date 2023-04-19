import awkward as ak
import math
import vector


class hpsParticleBase:
    def __init__(self, p4, barcode=-1):
        self.p4 = p4
        self.updatePtEtaPhiMass()
        self.barcode = barcode

    def updatePtEtaPhiMass(self):
        self.energy = self.p4.energy
        self.p = self.p4.p
        self.pt = self.p4.pt
        self.theta = self.p4.theta
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

        self.u_x = math.cos(self.phi) * math.sin(self.theta)
        self.u_y = math.sin(self.phi) * math.sin(self.theta)
        self.u_z = math.cos(self.theta)
