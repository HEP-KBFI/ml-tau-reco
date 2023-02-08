import awkward as ak
import numpy as np
import vector


class Tau:
    def __init__(self, chargedCands=[], strips=[], barcode=-1):
        cands_and_strips = [c.p4 for c in chargedCands] + [s.p4 for s in strips]
        cands_and_strips = np.array([[v.px, v.py, v.pz, v.E] for v in cands_and_strips])
        if len(cands_and_strips) == 0:
            self.p4 = vector.obj(px=0, py=0, pz=0, E=0)
        else:
            sum_p4 = np.sum(cands_and_strips, axis=0)
            self.p4 = vector.obj(px=sum_p4[0], py=sum_p4[1], pz=sum_p4[2], E=sum_p4[3])

        self.updatePtEtaPhiMass()
        self.q = 0.0
        for chargedCand in chargedCands:
            if chargedCand.q > 0.0:
                self.q += 1.0
            elif chargedCand.q < 0.0:
                self.q -= 1.0
            else:
                assert 0
        self.signal_chargedCands = chargedCands
        self.signal_strips = strips
        self.updateSignalCands()
        self.idDiscr = -1.0
        self.decayMode = "undefined"
        self.iso_cands = set()
        self.chargedIso = -1.0
        self.gammaIso = -1.0
        self.neutralHadronIso = -1.0
        self.combinedIso = -1.0
        self.barcode = barcode

    def updateSignalCands(self):
        self.num_signal_chargedCands = len(self.signal_chargedCands)
        self.num_signal_strips = len(self.signal_strips)
        self.signal_cands = set()
        self.signal_cands.update(self.signal_chargedCands)
        for strip in self.signal_strips:
            for cand in strip.cands:
                self.signal_cands.add(cand)

    def updatePtEtaPhiMass(self):
        self.pt = self.p4.pt
        self.eta = self.p4.eta
        self.phi = self.p4.phi
        self.mass = self.p4.mass

    def print(self):
        print(
            "tau #%i: pT = %1.1f, eta = %1.3f, phi = %1.3f, mass = %1.3f, idDiscr = %1.3f, decayMode = %s"
            % (self.barcode, self.pt, self.eta, self.phi, self.mass, self.idDiscr, self.decayMode)
        )
        # print("signal_chargedCands:")
        # for cand in self.signal_chargedCands:
        #    cand.print()
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
            % (self.chargedIso, self.gammaIso, self.neutralHadronIso, self.combinedIso)
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
        raise ValueError("Invalid decayMode = '%s'" % tau.decayMode)
    return retVal


def writeTaus(taus):
    retVal = {
        "tau_p4s": write_tau_p4s(taus),
        "tauSigCand_p4s": write_tau_cand_p4s(taus, "signal_cands"),
        "tauSigCand_pdgIds": write_tau_cand_attrs(taus, "signal_cands", "pdgId", np.int),
        "tauSigCand_q": write_tau_cand_attrs(taus, "signal_cands", "q", np.float),
        "tauIsoCand_p4s": write_tau_cand_p4s(taus, "iso_cands"),
        "tauIsoCand_pdgIds": write_tau_cand_attrs(taus, "iso_cands", "pdgId", np.int),
        "tauIsoCand_q": write_tau_cand_attrs(taus, "iso_cands", "q", np.float),
        "tauClassifier": ak.Array([tau.idDiscr for tau in taus]),
        "tauChargedIso": ak.Array([tau.chargedIso for tau in taus]),
        "tauGammaIso": ak.Array([tau.gammaIso for tau in taus]),
        "tauNeutralHadronIso": ak.Array([tau.neutralHadronIso for tau in taus]),
        "tau_charge": ak.Array([tau.q for tau in taus]),
        "tau_decaymode": ak.Array([get_decayMode(tau) for tau in taus]),
    }
    return retVal
