import awkward as ak
import vector

class Cand:
    def __init__(self, p4, pdgId, q, barcode):
        self.p4 = p4
        self.pt = p4.pt
        self.eta = p4.eta
        self.phi = p4.phi
        self.pdgId = pdgId
        self.abs_pdgId = abs(pdgId)
        self.q = q
        self.barcode = barcode

    def print(self):
        print(
            "cand #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, pdgId = %i, charge = %1.1f"
            % (self.barcode, self.p4.energy, self.pt, self.eta, self.phi, self.p4.mass, self.pdgId, self.q)
        )

def read_event_cand_p4s(data):
    retVal = vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "mass": data.tau}))
    return retVal

def buildCands(cand_p4s, cand_pdgIds, cand_qs):
    if not (len(cand_p4s) == len(cand_pdgIds) and len(cand_pdgIds) == len(cand_qs)):
        raise ValueError("Length of arrays for candidate for p4, pdgId, and q don't match !!")
    cands = []
    num_cands = len(cand_p4s)
    for idx in range(num_cands):
        cand = Cand(cand_p4s[idx], cand_pdgIds[idx], cand_qs[idx], barcode=idx)
        cands.append(cand)
    return cands

def readCands(data):
    event_cand_p4s = read_event_cand_p4s(data["event_reco_cand_p4s"])
    event_cand_pdgIds = data["event_reco_cand_pdg"]
    event_cand_qs = data["event_reco_cand_charge"]
    if not (len(event_cand_p4s) == len(event_cand_pdgIds) and len(event_cand_pdgIds) == len(event_cand_qs)):
        raise ValueError("Length of arrays for candidate p4, pdgId, and q don't match !!")
    event_cands = []
    num_jets = len(event_cand_p4s)
    for idx in range(num_jets):
        jet_cands = buildCands(event_cand_p4s[idx], event_cand_pdgIds[idx], event_cand_qs[idx])
        event_cands.append(jet_cands)
    return event_cands
