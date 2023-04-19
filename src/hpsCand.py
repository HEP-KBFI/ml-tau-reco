import awkward as ak
import math
import vector

from hpsParticleBase import hpsParticleBase


class Cand(hpsParticleBase):
    def __init__(self, p4, pdgId, q, d0, d0err, dz, dzerr, barcode=-1):
        super().__init__(p4=p4, barcode=barcode)
        self.pdgId = pdgId
        self.abs_pdgId = abs(pdgId)
        self.q = q
        self.d0 = d0
        self.d0err = d0err
        self.dz = math.fabs(dz)
        self.dzerr = dzerr

    def print(self):
        output = (
            "cand #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, pdgId = %i, charge = %1.1f"
            % (self.barcode, self.energy, self.pt, self.eta, self.phi, self.mass, self.pdgId, self.q)
        )
        if abs(self.q) > 0.5:
            output += ", d0 = %1.3f +/- %1.3f, dz = %1.3f +/- %1.3f" % (self.d0*1.e+4, self.d0err*1.e+4, self.dz*1.e+4, self.dzerr*1.e+4)
        print(output)


def read_event_cand_p4s(data):
    retVal = vector.awk(ak.zip({"px": data.x, "py": data.y, "pz": data.z, "mass": data.tau}))
    return retVal


def buildCands(cand_p4s, cand_pdgIds, cand_qs, cand_d0s, cand_d0errs, cand_dzs, cand_dzerrs):
    if not (
        len(cand_p4s) == len(cand_pdgIds)
        and len(cand_pdgIds) == len(cand_qs)
        and len(cand_qs) == len(cand_d0s)
        and len(cand_d0s) == len(cand_d0errs)
        and len(cand_d0errs) == len(cand_dzs)
        and len(cand_dzs) == len(cand_dzerrs)
    ):
        raise ValueError("Length of arrays for candidate for p4 and other features don't match !!")
    cands = []
    num_cands = len(cand_p4s)
    for idx in range(num_cands):
        cand = Cand(
            cand_p4s[idx],
            cand_pdgIds[idx],
            cand_qs[idx],
            cand_d0s[idx],
            cand_d0errs[idx],
            cand_dzs[idx],
            cand_dzerrs[idx],
            barcode=idx,
        )
        cands.append(cand)
    return cands


def readCands(data):
    event_cand_p4s = read_event_cand_p4s(data["event_reco_cand_p4s"])
    event_cand_pdgIds = data["event_reco_cand_pdg"]
    event_cand_qs = data["event_reco_cand_charge"]
    event_cand_d0s = data["event_reco_cand_dxy"]
    event_cand_d0errs = data["event_reco_cand_dxy_err"]
    event_cand_dzs = data["event_reco_cand_dz"]
    event_cand_dzerrs = data["event_reco_cand_dz_err"]
    if not (
        len(event_cand_p4s) == len(event_cand_pdgIds)
        and len(event_cand_pdgIds) == len(event_cand_qs)
        and len(event_cand_qs) == len(event_cand_d0s)
        and len(event_cand_d0s) == len(event_cand_d0errs)
        and len(event_cand_d0errs) == len(event_cand_dzs)
        and len(event_cand_dzs) == len(event_cand_dzerrs)
    ):
        raise ValueError("Length of arrays for candidate p4 and other features don't match !!")
    event_cands = []
    num_jets = len(event_cand_p4s)
    for idx in range(num_jets):
        jet_cands = buildCands(
            event_cand_p4s[idx],
            event_cand_pdgIds[idx],
            event_cand_qs[idx],
            event_cand_d0s[idx],
            event_cand_d0errs[idx],
            event_cand_dzs[idx],
            event_cand_dzerrs[idx],
        )
        event_cands.append(jet_cands)
    return event_cands
