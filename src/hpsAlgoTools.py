import math


twopi = 2.0 * math.pi
one_over_twopi = 1.0 / twopi


def comp_angle(cand1, cand2):
    dTheta = math.fabs(cand1.p4.theta - cand2.p4.theta)
    dPhi = math.fabs(cand1.phi - cand2.phi)
    if dPhi > math.pi:
        n = round(dPhi * one_over_twopi)
        dPhi = math.fabs(dPhi - n * twopi)
    angle = math.sqrt(dTheta * dTheta + dPhi * dPhi)
    return angle


def comp_deltaTheta(cand1, cand2):
    dTheta = math.fabs(cand1.p4.theta - cand2.p4.theta)
    return dTheta


def comp_deltaR(cand1, cand2):
    dEta = math.fabs(cand1.eta - cand2.eta)
    dPhi = math.fabs(cand1.phi - cand2.phi)
    if dPhi > math.pi:
        n = round(dPhi * one_over_twopi)
        dPhi = math.fabs(dPhi - n * twopi)
    dR = math.sqrt(dEta * dEta + dPhi * dPhi)
    return dR


def comp_deltaEta(cand1, cand2):
    dEta = math.fabs(cand1.eta - cand2.eta)
    return dEta


def comp_deltaPhi(cand1, cand2):
    dPhi = math.fabs(cand1.phi - cand2.phi)
    if dPhi > math.pi:
        n = round(dPhi * one_over_twopi)
        dPhi = math.fabs(dPhi - n * twopi)
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
