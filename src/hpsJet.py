from hpsCand import buildCands


class Jet:
    def __init__(self, jet_p4, jet_constituents_p4, jet_constituents_pdgId, jet_constituents_q, barcode=-1):
        self.p4 = jet_p4
        print(type(jet_p4))
        self.pt = jet_p4.pt
        self.eta = jet_p4.eta
        self.phi = jet_p4.phi
        self.mass = jet_p4.mass
        self.constituents = buildCands(jet_constituents_p4, jet_constituents_pdgId, jet_constituents_q)
        self.q = 0.0
        for cand in self.constituents:
            self.q += cand.q
        self.barcode = barcode


def buildJets(jets_p4, jets_constituents_p4, jets_constituents_pdgId, jets_constituents_q):
    if not (
        len(jets_p4) == len(jets_constituents_p4)
        and len(jets_constituents_p4) == len(jets_constituents_pdgId)
        and len(jets_constituents_pdgId) == len(jets_constituents_q)
    ):
        raise ValueError("Length of lists for p4, constituents_p4, constituents_pdgId, and constituents_q don't match !!")
    jets = []
    numJets = len(jets_p4)
    for idx in range(numJets):
        jet = Jet(jets_p4[idx], jets_constituents_p4[idx], jets_constituents_pdgId[idx], jets_constituents_q[idx])
        jets.append(jet)
    return jets
