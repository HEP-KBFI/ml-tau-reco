import awkward as ak
import vector

from hpsCand import buildCands


class Jet:
    def __init__(self, jet_p4, jet_constituents_p4, jet_constituents_pdgId, jet_constituents_q, barcode=-1):
        self.p4 = jet_p4
        self.pt = jet_p4.pt
        self.eta = jet_p4.eta
        self.phi = jet_p4.phi
        self.mass = jet_p4.mass
        self.constituents = buildCands(jet_constituents_p4, jet_constituents_pdgId, jet_constituents_q)
        # CV: reverse=True argument needed in order to sort jet constituents in order of decreasing (and NOT increasing) pT
        self.constituents.sort(key=lambda cand: cand.pt, reverse=True)
        self.num_constituents = len(self.constituents)
        self.q = 0.0
        for cand in self.constituents:
            self.q += cand.q
        self.barcode = barcode

    def print(self):
        print(
            "jet #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.2f, #constituents = %i"
            % (self.barcode, self.pt, self.eta, self.phi, self.mass, len(self.constituents))
        )
        print("constituents:")
        for cand in self.constituents:
            cand.print()


def read_jet_p4s(data):
    retVal = vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "mass": data.tau}))
    return retVal


def read_jet_constituent_p4s(data):
    retVal = vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "mass": data.tau}))
    return retVal


def buildJets(jet_p4s, jet_constituent_p4s, jet_constituent_pdgIds, jet_constituent_qs):
    if not (
        len(jet_p4s) == len(jet_constituent_p4s)
        and len(jet_constituent_p4s) == len(jet_constituent_pdgIds)
        and len(jet_constituent_pdgIds) == len(jet_constituent_qs)
    ):
        raise ValueError("Length of arrays for jet p4, constituent_p4, constituent_pdgId, and constituent_q don't match !!")
    jets = []
    num_jets = len(jet_p4s)
    for idx in range(num_jets):
        jet = Jet(jet_p4s[idx], jet_constituent_p4s[idx], jet_constituent_pdgIds[idx], jet_constituent_qs[idx], idx)
        jets.append(jet)
    return jets


def readJets(data):
    jet_p4s = read_jet_p4s(data["reco_jet_p4s"])
    jet_constituent_p4s = read_jet_constituent_p4s(data["reco_cand_p4s"])
    jet_constituent_pdgIds = data["reco_cand_pdg"]
    jet_constituent_qs = data["reco_cand_charge"]
    jets = buildJets(jet_p4s, jet_constituent_p4s, jet_constituent_pdgIds, jet_constituent_qs)
    return jets
