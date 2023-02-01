class Cand:
    def __init__(self, p4, pdgId, q, barcode):
        self.p4 = p4
        self.pt = p4.pt
        self.eta = p4.eta
        self.phi = p4.phi
        self.pdgId = pdgId
        self.q = q
        self.barcode = barcode


def buildCands(cands_p4, cands_pdgId, cands_q):
    if not (len(cands_p4) == len(cands_pdgId) and len(cands_pdgId) == len(cands_q)):
        raise ValueError("Length of lists for p4, pdgId, and q don't match !!")
    cands = []
    numCands = len(cands_p4)
    for idx in range(numCands):
        cand = Cand(cands_p4[idx], cands_pdgId[idx], cands_q[idx], barcode=idx)
        cands.append(cand)
    return cands


def isHigherPt(cand1, cand2):
    return cand1.pt > cand2.pt
