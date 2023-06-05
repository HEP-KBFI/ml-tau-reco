import math


twopi = 2.0 * math.pi
one_over_twopi = 1.0 / twopi


def comp_angle3d(cand1, cand2):
    cos_angle = cand1.u_x * cand2.u_x + cand1.u_y * cand2.u_y + cand1.u_z * cand2.u_z
    assert cos_angle >= -1.0 and cos_angle <= +1.0
    angle = math.acos(cos_angle)
    return angle


def comp_deltaR_thetaphi(cand1, cand2):
    dTheta = math.fabs(cand1.theta - cand2.theta)
    dPhi = math.fabs(cand1.phi - cand2.phi)
    if dPhi > math.pi:
        n = round(dPhi * one_over_twopi)
        dPhi = math.fabs(dPhi - n * twopi)
    dR = math.sqrt(dTheta * dTheta + dPhi * dPhi)
    return dR


def comp_deltaTheta(cand1, cand2):
    dTheta = math.fabs(cand1.theta - cand2.theta)
    return dTheta


def comp_deltaR_etaphi(cand1, cand2):
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


def comp_p_sum(cands):
    p_sum = 0.0
    for cand in cands:
        p_sum += cand.p
    return p_sum


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
