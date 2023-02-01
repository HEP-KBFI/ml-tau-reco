import math
import vector

m_pi0 = 0.135

class Tau:
  def __init__(self, barcode = -1):
    self.p4 = vector(pt = 0., phi = 0., theta = 0., mass = 0.)
    updatePtEtaPhiMass()
    self.signalChargedCands = []
    self.signalStrips = []
    updateSignalCands()
    self.barcode = barcode

  def __init__(self, chargedCands, strips, barcode = -1):
    self.p4 = vector(pt = 0., phi = 0., theta = 0., mass = 0.)
    for chargedCand in chargedCands:
      self.p4 += chargedCand.p4
    for strip in strips:
      strip_px = strip.p4.px
      strip_py = strip.p4.py
      strip_pz = strip.p4.pz
      strip_E  = math.sqrt(strip_px*strip_px + strip_py*strip_py + strip_pz*strip_pz + m_pi0*m_pi0)
      self.p4 += vector(px = strip_px, py = strip_py, pz = strip_pz, E = strip_E)
    updatePtEtaPhiMass()
    self.q = 0.
    for chargedCand in chargedCands:
      if chargedCand.q > 0.:
        self.q += 1.
      elif chargedCand.q < 0.:
        self.q -= 1.
      else:
        assert(0)
    self.signalChargedCands = chargedCands
    self.signalStrips = strips
    updateCands()
    self.barcode = barcode
 
  def updateCands(self):
    self.numSignalChargedCands = len(self.signalChargedCands)
    self.numSignalStrips = len(self.signalStrips)
    self.signalCands = set()
    self.signalCands.update(self.signalChargedCands)
    for strip in self.signalSrips:
      for cand in strip.cands:
        self.signalCands.add(cand)

  def updatePtEtaPhiMass(self):
    self.pt = self.p4.pt
    self.eta = self.p4.eta
    self.phi = self.p4.phi
    self.mass = self.p4.mass
