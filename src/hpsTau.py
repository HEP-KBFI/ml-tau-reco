import awkward as ak
import numpy as np
import vector

from hpsAlgoTools import comp_deltaPhi, comp_pt_sum, selectCandsByDeltaR, selectCandsByPdgId
from hpsParticleBase import hpsParticleBase


class Tau(hpsParticleBase):
    def __init__(self, chargedCands=[], strips=[], barcode=-1):
        chargedCand_and_strip_p4s = [chargedCand.p4 for chargedCand in chargedCands] + [strip.p4 for strip in strips]
        chargedCand_and_strip_p4s = np.array([[p4.px, p4.py, p4.pz, p4.E] for p4 in chargedCand_and_strip_p4s])
        sum_p4 = None
        if len(chargedCand_and_strip_p4s) == 0:
            sum_p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(chargedCand_and_strip_p4s, axis=0)
            sum_p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])
        super().__init__(p4=sum_p4, barcode=barcode)
        self.q = 0.0
        self.leadChargedCand_pt = 0.0
        for chargedCand in chargedCands:
            if chargedCand.q > 0.0:
                self.q += 1.0
            elif chargedCand.q < 0.0:
                self.q -= 1.0
            else:
                assert 0
            if chargedCand.pt > self.leadChargedCand_pt:
                self.leadChargedCand_pt = chargedCand.pt
        self.signal_chargedCands = chargedCands
        self.signal_strips = strips
        self.updateSignalCands()
        self.idDiscr = -1.0
        self.decayMode = "undefined"
        self.iso_cands = set()
        self.chargedIso_dR0p5 = -1.0
        self.gammaIso_dR0p5 = -1.0
        self.neutralHadronIso_dR0p5 = -1.0
        self.combinedIso_dR0p5 = -1.0

    def updateSignalCands(self):
        self.num_signal_chargedCands = len(self.signal_chargedCands)
        self.num_signal_strips = len(self.signal_strips)
        self.signal_cands = set()
        self.signal_cands.update(self.signal_chargedCands)
        for strip in self.signal_strips:
            for cand in strip.cands:
                self.signal_cands.add(cand)
        self.signal_gammaCands = selectCandsByPdgId(self.signal_cands, [22])

    def print(self):
        print(
            "tau #%i: energy = %1.1f, pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, idDiscr = %1.3f, decayMode = %s"
            % (self.barcode, self.energy, self.pt, self.eta, self.phi, self.mass, self.idDiscr, self.decayMode)
        )
        # print("signal_chargedCands:")
        # for cand in self.signal_chargedCands:
        #    cand.print()
        # print("#signal_gammaCands = %i" % len(self.signal_gammaCands))
        # print("#signal_chargedCands = %i" % len(self.signal_chargedCands))
        # print("leadChargedCand: pT = %1.1f" % self.leadChargedCand_pt)
        # print("signal_strips:")
        # for strip in self.signal_strips:
        #    strip.print()
        print("signal_cands:")
        for cand in self.signal_cands:
            cand.print()
        print("iso_cands:")
        for cand in self.iso_cands:
            cand.print()
        print(
            " isolation: charged = %1.2f, gamma = %1.2f, neutralHadron = %1.2f, combined = %1.2f"
            % (self.chargedIso_dR0p5, self.gammaIso_dR0p5, self.neutralHadronIso_dR0p5, self.combinedIso_dR0p5)
        )


def write_tau_p4s(taus):
    retVal = vector.awk(
        ak.zip(
            {
                "px": [tau.p4.px for tau in taus],
                "py": [tau.p4.py for tau in taus],
                "pz": [tau.p4.pz for tau in taus],
                "mass": [tau.p4.mass for tau in taus],
            }
        )
    )
    return retVal


def build_dummy_array(dtype=np.float):
    num = 0
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            ak.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )


def write_tau_cand_p4s(taus, collection):
    retVal = ak.Array(
        [
            vector.awk(
                ak.zip(
                    {
                        "px": [cand.p4.px for cand in getattr(tau, collection)],
                        "py": [cand.p4.py for cand in getattr(tau, collection)],
                        "pz": [cand.p4.pz for cand in getattr(tau, collection)],
                        "mass": [cand.p4.mass for cand in getattr(tau, collection)],
                    }
                )
            )
            if len(getattr(tau, collection)) >= 1
            else build_dummy_array()
            for tau in taus
        ]
    )
    return retVal


def write_tau_cand_attrs(taus, collection, attr, dtype):
    retVal = ak.Array(
        [
            ak.Array([getattr(cand, attr) for cand in getattr(tau, collection)])
            if len(getattr(tau, collection)) >= 1
            else build_dummy_array(dtype)
            for tau in taus
        ]
    )
    return retVal


def get_decayMode(tau):
    retVal = None
    if tau.decayMode == "undefined":
        retVal = -1
    elif tau.decayMode == "1Prong0Pi0":
        retVal = 0
    elif tau.decayMode == "1Prong1Pi0":
        retVal = 1
    elif tau.decayMode == "1Prong2Pi0":
        retVal = 2
    elif tau.decayMode == "3Prong0Pi0":
        retVal = 10
    elif tau.decayMode == "3Prong1Pi0":
        retVal = 11
    else:
        raise RuntimeError("Invalid decayMode = '%s'" % tau.decayMode)
    return retVal


def comp_photonPtSumOutsideSignalCone(tau):
    retVal = 0.0
    if tau.metric_dR_or_angle is not None:
        for cand in tau.signal_gammaCands:
            dR = tau.metric_dR_or_angle(tau, cand)
            if dR > tau.signalConeSize:
                retVal += cand.pt
    return retVal


def comp_pt_weighted_dX(tau, cands, metric):
    pt_weighted_dX_sum = 0.0
    pt_sum = 0.0
    if metric is not None:
        for cand in cands:
            dX = abs(metric(tau, cand))
            pt_weighted_dX_sum += cand.pt * dX
            pt_sum += cand.pt
    if pt_sum > 0.0:
        return pt_weighted_dX_sum / pt_sum
    else:
        return 0.0


def writeTaus(taus):
    retVal = {
        "tau_p4s": write_tau_p4s(taus),
        "tauSigCand_p4s": write_tau_cand_p4s(taus, "signal_cands"),
        "tauSigCand_pdgIds": write_tau_cand_attrs(taus, "signal_cands", "pdgId", np.int),
        "tauSigCand_q": write_tau_cand_attrs(taus, "signal_cands", "q", np.float),
        "tauSigCand_d0": write_tau_cand_attrs(taus, "signal_cands", "d0", np.float),
        "tauSigCand_d0err": write_tau_cand_attrs(taus, "signal_cands", "d0err", np.float),
        "tauSigCand_dz": write_tau_cand_attrs(taus, "signal_cands", "dz", np.float),
        "tauSigCand_dzerr": write_tau_cand_attrs(taus, "signal_cands", "dzerr", np.float),
        "tauIsoCand_p4s": write_tau_cand_p4s(taus, "iso_cands"),
        "tauIsoCand_pdgIds": write_tau_cand_attrs(taus, "iso_cands", "pdgId", np.int),
        "tauIsoCand_q": write_tau_cand_attrs(taus, "iso_cands", "q", np.float),
        "tauIsoCand_d0": write_tau_cand_attrs(taus, "iso_cands", "d0", np.float),
        "tauIsoCand_d0err": write_tau_cand_attrs(taus, "iso_cands", "d0err", np.float),
        "tauIsoCand_dz": write_tau_cand_attrs(taus, "iso_cands", "dz", np.float),
        "tauIsoCand_dzerr": write_tau_cand_attrs(taus, "iso_cands", "dzerr", np.float),
        "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),
        "tauChargedIso_dR0p5": ak.Array([tau.chargedIso_dR0p5 for tau in taus]),
        "tauGammaIso_dR0p5": ak.Array([tau.gammaIso_dR0p5 for tau in taus]),
        "tauNeutralHadronIso_dR0p5": ak.Array([tau.neutralHadronIso_dR0p5 for tau in taus]),
        "tauChargedIso_dR0p3": ak.Array(
            [comp_pt_sum(selectCandsByDeltaR(tau.iso_chargedCands, tau, 0.3, tau.metric_dR_or_angle)) for tau in taus]
        ),
        "tauGammaIso_dR0p3": ak.Array(
            [comp_pt_sum(selectCandsByDeltaR(tau.iso_gammaCands, tau, 0.3, tau.metric_dR_or_angle)) for tau in taus]
        ),
        "tauNeutralHadronIso_dR0p3": ak.Array(
            [comp_pt_sum(selectCandsByDeltaR(tau.iso_neutralHadronCands, tau, 0.3, tau.metric_dR_or_angle)) for tau in taus]
        ),
        "tauPhotonPtSumOutsideSignalCone": ak.Array([comp_photonPtSumOutsideSignalCone(tau) for tau in taus]),
        "tau_charge": ak.Array([tau.q for tau in taus]),
        "tau_decaymode": ak.Array([get_decayMode(tau) for tau in taus]),
        "tau_nGammas": ak.Array([len(tau.signal_gammaCands) for tau in taus]),
        "tau_nCharged": ak.Array([len(tau.signal_chargedCands) for tau in taus]),
        "tau_leadChargedCand_pt": ak.Array([tau.leadChargedCand_pt for tau in taus]),
        "tau_emEnergyFrac": ak.Array(
            [(comp_pt_sum(tau.signal_gammaCands) / tau.pt) if tau.pt > 0.0 else 0.0 for tau in taus]
        ),
        "tau_dEta_strip": ak.Array(
            [comp_pt_weighted_dX(tau, tau.signal_gammaCands, tau.metric_dEta_or_dTheta) for tau in taus]
        ),
        "tau_dPhi_strip": ak.Array([comp_pt_weighted_dX(tau, tau.signal_gammaCands, comp_deltaPhi) for tau in taus]),
        "tau_dR_signal": ak.Array([comp_pt_weighted_dX(tau, tau.signal_gammaCands, tau.metric_dR_or_angle) for tau in taus]),
        "tau_dR_iso": ak.Array([comp_pt_weighted_dX(tau, tau.iso_gammaCands, tau.metric_dR_or_angle) for tau in taus]),
    }
    return retVal
