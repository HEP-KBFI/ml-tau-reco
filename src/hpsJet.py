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
        jet = Jet(jets_p4[idx], jets_constituents_p4[idx], jets_constituents_pdgId[idx], jets_constituents_q[idx], idx)
        jets.append(jet)
    return jets
