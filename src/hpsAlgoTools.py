import math


def comp_angle(cand1, cand2):
    dTheta = cand1.p4.theta - cand2.p4.theta
    dPhi = cand1.phi - cand2.phi
    angle = math.sqrt(dTheta * dTheta + dPhi * dPhi)
    return angle


def comp_deltaTheta(cand1, cand2):
    dTheta = cand1.p4.theta - cand2.p4.theta
    return dTheta


def comp_deltaR(cand1, cand2):
    dEta = cand1.eta - cand2.eta
    dPhi = cand1.phi - cand2.phi
    dR = math.sqrt(dEta * dEta + dPhi * dPhi)
    return dR


def comp_deltaEta(cand1, cand2):
    dEta = cand1.eta - cand2.eta
    return dEta


def comp_deltaPhi(cand1, cand2):
    dPhi = cand1.phi - cand2.phi
    return dPhi


def comp_pt_sum(cands):
    pt_sum = 0.0
    for cand in cands:
        pt_sum += cand.pt
    return pt_sum


def selectCandsByDeltaR(cands, ref, dRmax, coneMetric):
    selectedCands = []
    for cand in cands:
        dR = coneMetric(cand, ref)
        if dR < dRmax:
            selectedCands.append(cand)
    return selectedCands


def selectCandsByPdgId(cands, pdgIds=[]):
    selectedCands = []
    for cand in cands:
        if cand.abs_pdgId in pdgIds:
            selectedCands.append(cand)
    return selectedCands
