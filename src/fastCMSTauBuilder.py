import vector
import numpy as np
import awkward as ak
import uproot
from basicTauBuilder import BasicTauBuilder

"""
author(s): Torben Lange, Christian Veelken
date: 09.02.2023
Class that builds taus from gen information, mimicing CMS tau reconstruction quality,
i.e. energy response, signalEff, missID rate, decayMode reconstruction, and charge flip rate.
"""


class FastCMSTauBuilder(BasicTauBuilder):
    def __init__(
        self,
        config={
            "energyResponsePDF": "data/tauEnergyResponse_all.root",  # PRF-14-001 Fig 19
            "missIDLooseMap": {
                "pT_thr": [20, 30, 40, 50, 70, 90, 120, 150, 200],  # CMS Tau 20-001 sig: Fig 5 (hepdata)
                "prob": [0.00075, 0.0075, 0.0085, 0.00625, 0.0041, 0.00275, 0.002, 0.0015, 0.001],
            },
            "missIDTightMap": {
                "pT_thr": [20, 30, 40, 50, 70, 90, 120, 150, 200],  # CMS Tau 20-001 sig: Fig 5 (hepdata)
                "prob": [0.0003, 0.00265, 0.00205, 0.00205, 0.0013, 0.0009, 0.00075, 0.00045, 0.00035],
            },
            "effIDLooseMap": {
                "pT_thr": [0, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200],  # CMS Tau 20-001 sig: Fig 13
                "prob": [
                    0.297,
                    0.427,
                    0.479,
                    0.542,
                    0.571,
                    0.598,
                    0.615,
                    0.624,
                    0.635,
                    0.643,
                    0.653,
                    0.663,
                    0.666,
                    0.677,
                    0.679,
                    0.69,
                ],
            },
            "effIDTightMap": {
                "pT_thr": [0, 22, 24, 26, 28, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200],  # CMS Tau 20-001 sig: Fig 13
                "prob": [
                    0.207,
                    0.299,
                    0.349,
                    0.382,
                    0.404,
                    0.426,
                    0.442,
                    0.452,
                    0.47,
                    0.486,
                    0.502,
                    0.515,
                    0.524,
                    0.545,
                    0.563,
                    0.611,
                ],
            },
            "dModeConfMat": [  # DP 2020-041 Figure 6 right
                [0.829, 0.158, 0.01, 0.001, 0.0, 0.002],  # 1p->x (didnt add up to one other_orig->0.003)
                [0.102, 0.794, 0.070, 0.004, 0.003, 0.027],  # 1p1p0->x
                [0.023, 0.526, 0.387, 0.005, 0.006, 0.053],  # 1p2p0->x
                [0.022, 0.020, 0.003, 0.866, 0.065, 0.024],  # 3p->x
                [0.002, 0.041, 0.012, 0.227, 0.652, 0.066],  # 3p1p0 ->x (didnt end up to one, other_orig->0.065)
                [0.143, 0.337, 0.205, 0.111, 0.116, 0.088],  # Other -> rare
            ]
            # [ # CMS Tau 20-001 Fig 1
            #   [0.8,0.09,0.,0.,0.,0.11], #1p->X
            #   [0.14,0.57,0.,0.01,0.01,0.27], #1p1p0->x
            #   [0.14,0.57,0.,0.01,0.01,0.27], #1p2p0->x
            #   [0.03,0.02,0.,0.61,0.05,0.29], # 3p ->x
            #   [0.01,0.06,0.,0.27,0.36,0.49], #3p1p0->x
            #   [0.04,0.36,0.,0.07,0.11,0.41] #Other->Rare
            # ]
        },
    ):
        self._builderConfig = dict()
        for key in config:
            self._builderConfig[key] = config[key]
        self._pdfFile = uproot.open(self._builderConfig["energyResponsePDF"])
        self._pdf_sig_20_100 = self._getPdf(
            "signal_20_100",
        )
        self._pdf_sig_100_200 = self._getPdf("signal_100_200")
        self._pdf_sig_200_inf = self._getPdf("signal_200_inf")
        self._pdf_bkg_20_100 = self._getPdf("bkg_20_100")
        self._pdf_bkg_100_200 = self._getPdf("bkg_100_200")
        self._pdf_bkg_200_inf = self._getPdf("bkg_200_inf")
        self._missIDLoose = [self._builderConfig["missIDLooseMap"]["pT_thr"], self._builderConfig["missIDLooseMap"]["prob"]]
        self._missIDTight = [self._builderConfig["missIDTightMap"]["pT_thr"], self._builderConfig["missIDTightMap"]["prob"]]
        self._effIDLoose = [self._builderConfig["effIDLooseMap"]["pT_thr"], self._builderConfig["effIDLooseMap"]["prob"]]
        self._effIDTight = [self._builderConfig["effIDTightMap"]["pT_thr"], self._builderConfig["effIDTightMap"]["prob"]]
        self._sigConfMat = self._builderConfig["dModeConfMat"]

    """
    Helper to read the pdfs for the tauEnergy response
    """

    def _getPdf(self, name):
        hist = self._pdfFile[name]
        values = hist.axis().centers()
        prob = hist.values()
        return [values, prob]

    """
    Quick helper to reinitialize akward 4-Vectors
    """

    def _asP4(self, p4):
        P4 = vector.awk(
            ak.zip(
                {
                    "mass": p4.tau,
                    "x": p4.x,
                    "y": p4.y,
                    "z": p4.z,
                }
            )
        )
        return P4

    """
    Fakes the deepJet tau classifier against jets, using the missID rates for jets and sigEff for real taus,
    the classifier is set to either 0 (not reconstructed), 0.49 (passes loose ID) or 0.99 (passes tight ID),
    to mimic the ROC of CMS taus.
    """

    def _calcClassifier(self, tauVisPt, jetPt, isBG):
        isSig = isBG == 0
        survivalProb_SigLoose = np.zeros(len(tauVisPt))
        survivalProb_SigTight = np.zeros(len(tauVisPt))
        for idx in range(len(self._effIDLoose[0])):
            mask = None
            if idx < len(self._effIDLoose[0]) - 1:
                mask = np.logical_and(
                    np.asarray(tauVisPt > self._effIDLoose[0][idx]), np.asarray(tauVisPt <= self._effIDLoose[0][idx + 1])
                )
            else:
                mask = np.asarray(tauVisPt > self._effIDLoose[0][idx])
            dLoose = self._effIDLoose[1][idx] * np.ones(mask.shape)
            dLoose *= mask
            survivalProb_SigLoose += dLoose
            dTight = self._effIDTight[1][idx] * np.ones(mask.shape)
            dTight *= mask
            survivalProb_SigTight += dTight
        survivalProb_BkgLoose = np.zeros(len(tauVisPt))
        survivalProb_BkgTight = np.zeros(len(tauVisPt))
        for idx in range(len(self._missIDLoose[0])):
            mask = None
            if idx < len(self._missIDLoose[0]) - 1:
                mask = np.logical_and(
                    np.asarray(jetPt > self._missIDLoose[0][idx]), np.asarray(jetPt <= self._missIDLoose[0][idx + 1])
                )
            else:
                mask = np.asarray(jetPt > self._missIDLoose[0][idx])
            dLoose = self._missIDLoose[1][idx] * np.ones(mask.shape)
            dLoose *= mask
            survivalProb_BkgLoose += dLoose
            dTight = self._missIDLoose[1][idx] * np.ones(mask.shape)
            dTight *= mask
            survivalProb_BkgTight += dTight
        probExp = np.random.rand(len(survivalProb_SigLoose))
        survivalSigTight = np.logical_and(probExp < survivalProb_SigTight, isSig)
        survivalSigLoose = np.logical_and(
            np.logical_and(probExp < survivalProb_SigLoose, isSig), np.logical_not(survivalSigTight)
        )
        survivalBkgTight = np.logical_and(probExp < survivalProb_BkgTight, isBG)
        survivalBkgLoose = np.logical_and(
            np.logical_and(probExp < survivalProb_BkgLoose, isBG), np.logical_not(survivalBkgTight)
        )
        survivalLoose = np.asarray(np.logical_or(survivalSigLoose, survivalBkgLoose))
        survivalTight = np.asarray(np.logical_or(survivalSigTight, survivalBkgTight))
        scoreLoose = 0.49 * np.ones(len(tauVisPt))
        scoreLoose *= survivalLoose
        scoreTight = 0.98 * np.ones(len(tauVisPt))
        scoreTight *= survivalTight
        score = scoreLoose + scoreTight + 0.01
        dclass = ak.Array(score)
        return dclass

    """
    Smears the energy for real tau_had and non-tau jets.
    For tau_had, the HPS energy response of tau_genjets is used,
    for non-tau jets, the ratio of reconstructed tau energy and the genJet pt
    """

    def _smearEnergy(self, P4s_sig, P4s_bkg, bkgMask):
        properP4_sig = self._asP4(P4s_sig)
        px_sig = np.asarray(properP4_sig.px.layout)
        py_sig = np.asarray(properP4_sig.py.layout)
        pz_sig = np.asarray(properP4_sig.pz.layout)
        energy_sig = np.asarray(properP4_sig.energy.layout)
        pt_sig = np.asarray(properP4_sig.pt.layout)
        mask_20_100_sig = pt_sig <= 100
        mask_100_200_sig = np.logical_and(pt_sig > 100, pt_sig <= 200)
        mask_200_inf_sig = pt_sig > 200
        smear_20_100_sig = np.random.choice(self._pdf_sig_20_100[0], px_sig.shape, p=self._pdf_sig_20_100[1])
        smear_100_200_sig = np.random.choice(self._pdf_sig_100_200[0], px_sig.shape, p=self._pdf_sig_100_200[1])
        smear_200_inf_sig = np.random.choice(self._pdf_sig_200_inf[0], px_sig.shape, p=self._pdf_sig_200_inf[1])
        smear_sig = (
            smear_20_100_sig * mask_20_100_sig + smear_100_200_sig * mask_100_200_sig + smear_200_inf_sig * mask_200_inf_sig
        )
        properP4_bkg = self._asP4(P4s_bkg)
        px_bkg = np.asarray(properP4_bkg.px.layout)
        py_bkg = np.asarray(properP4_bkg.py.layout)
        pz_bkg = np.asarray(properP4_bkg.pz.layout)
        energy_bkg = np.asarray(properP4_bkg.energy.layout)
        pt_bkg = np.asarray(properP4_bkg.pt.layout)
        mask_20_100_bkg = pt_bkg <= 100
        mask_100_200_bkg = np.logical_and(pt_bkg > 100, pt_bkg <= 200)
        mask_200_inf_bkg = pt_bkg > 200
        smear_20_100_bkg = np.random.choice(self._pdf_bkg_20_100[0], px_bkg.shape, p=self._pdf_bkg_20_100[1])
        smear_100_200_bkg = np.random.choice(self._pdf_bkg_100_200[0], px_bkg.shape, p=self._pdf_bkg_100_200[1])
        smear_200_inf_bkg = np.random.choice(self._pdf_bkg_200_inf[0], px_bkg.shape, p=self._pdf_bkg_200_inf[1])
        smear_bkg = (
            smear_20_100_bkg * mask_20_100_bkg + smear_100_200_bkg * mask_100_200_bkg + smear_200_inf_bkg * mask_200_inf_bkg
        )
        outP4 = vector.awk(
            ak.zip(
                {
                    "px": px_bkg * smear_bkg * bkgMask + px_sig * smear_sig * (bkgMask == 0),
                    "py": py_bkg * smear_bkg * bkgMask + py_sig * smear_sig * (bkgMask == 0),
                    "pz": pz_bkg * smear_bkg * bkgMask + pz_sig * smear_sig * (bkgMask == 0),
                    "E": energy_bkg * smear_bkg * bkgMask + energy_sig * smear_sig * (bkgMask == 0),
                }
            )
        )
        return self._asP4(outP4)

    """
    The generator decay mode is given in a more granular classification then we need:
    1p, 1p1p0, 1p2p0, 3p, 3p1p0, other.
    This helper, converts between the two conventions.
    -> less granular for easier handling.
    """

    def _conv_dMode(self, dmode):
        mask_1p = dmode == 0
        mask_1p1p0 = dmode == 1
        mask_1p2p0 = dmode == 2
        mask_3p = dmode == 10
        mask_3p1p0 = dmode == 11
        mask_notTau = dmode == -1
        mask_other = np.logical_or(mask_1p, mask_1p1p0)
        mask_other = np.logical_or(mask_other, mask_1p2p0)
        mask_other = np.logical_or(mask_other, mask_3p)
        mask_other = np.logical_or(mask_other, mask_3p1p0)
        mask_other = np.logical_or(mask_other, mask_notTau)
        mask_other = np.logical_not(mask_other)
        dmode_conv = (
            np.asarray(mask_notTau) * (-1.0)
            + np.asarray(mask_1p) * 1.0
            + np.asarray(mask_1p1p0) * 2.0
            + np.asarray(mask_1p2p0) * 3.0
            + np.asarray(mask_3p) * 4.0
            + np.asarray(mask_3p1p0) * 5.0
            + np.asarray(mask_other) * 6.0
        )
        return ak.Array(dmode_conv)

    """
    The generator decay mode is given in a more granular classification then we need:
    1p, 1p1p0, 1p2p0, 3p, 3p1p0, other.
    This helper, converts between the two conventions.
    -> more granular for compatability
    """

    def _conv_dModeToGranular(self, dmode):
        mask_notTau = dmode == -1
        mask_1p = dmode == 1
        mask_1p1p0 = dmode == 2
        mask_1p2p0 = dmode == 3
        mask_3p = dmode == 4
        mask_3p1p0 = dmode == 5
        mask_other = dmode == 6
        dmode_conv = (
            np.asarray(mask_notTau) * (-1.0)
            + np.asarray(mask_1p) * 0.0
            + np.asarray(mask_1p1p0) * 1.0
            + np.asarray(mask_1p2p0) * 2.0
            + np.asarray(mask_3p) * 10.0
            + np.asarray(mask_3p1p0) * 11.0
            + np.asarray(mask_other) * 15.0
        )
        return ak.Array(dmode_conv)

    """
    'Calculates' the tau decay mode. For real tau_had, the gen decay mode is smeared
     according to the confusion matrix from the decay mode finding in CMS tau reconstruction.
    For non-taus, the decay mode is randomly drawn from 1p, 1p1p0, 1p2p0, 3p, 3p1p0, other.
    """

    def _calcDmode(self, genMode):
        mask_1p = np.asarray(genMode == 1)
        smear_1p = np.random.choice(np.arange(1, 7), mask_1p.shape, p=self._sigConfMat[0])
        mask_1p1p0 = np.asarray(genMode == 2)
        smear_1p1p0 = np.random.choice(np.arange(1, 7), mask_1p1p0.shape, p=self._sigConfMat[1])
        mask_1p2p0 = np.asarray(genMode == 3)
        smear_1p2p0 = np.random.choice(np.arange(1, 7), mask_1p2p0.shape, p=self._sigConfMat[2])
        mask_3p = np.asarray(genMode == 4)
        smear_3p = np.random.choice(np.arange(1, 7), mask_3p.shape, p=self._sigConfMat[3])
        mask_3p1p0 = np.asarray(genMode == 5)
        smear_3p1p0 = np.random.choice(np.arange(1, 7), mask_3p1p0.shape, p=self._sigConfMat[4])
        mask_other = np.asarray(genMode == 6)
        smear_other = np.random.choice(np.arange(1, 7), mask_other.shape, p=self._sigConfMat[5])
        mask_notTau = np.asarray(genMode == -1)
        smear_notTau = np.random.choice(np.arange(1, 7), mask_1p.shape)
        genMode_smeared = (
            mask_1p * smear_1p
            + mask_1p1p0 * smear_1p1p0
            + mask_1p2p0 * smear_1p2p0
            + mask_3p * smear_3p
            + mask_3p1p0 * smear_3p1p0
            + mask_other * smear_other
            + mask_notTau * smear_notTau
        )
        return ak.Array(genMode_smeared)

    """
    Smears the generator tau_had charge according to the charge flip rate given in the CMS deepTau paper:
    'For the decay modes without missing charged hadrons, the charge assignment is 99% correct
     for an average Z → ττ event sample, 98% for τh with pT ≈ 200GeV, and 92%forτh with pT ≈ 1TeV.'
    As we dont have this rate given in more granularity, we chose thresholds of 200 and 350 GeV for the
    three given flip probabilities. For non-tau jets, the charge is randomly drawn from [1,-1.].
    For both signal and background we ignore decay modes with an absolute charge different from 1.
    """

    def _smearCharge(self, charge, isBG, pt):
        mask_pt_0_200 = np.asarray(pt < 200)
        smear_pt_0_200 = np.random.choice([1, -1], mask_pt_0_200.shape, p=[0.99, 0.01])
        mask_pt_200_350 = np.logical_and(np.asarray(pt >= 200), np.asarray(pt < 350))
        smear_pt_200_350 = np.random.choice([1, -1], mask_pt_200_350.shape, p=[0.98, 0.02])
        mask_pt_350_inf = np.asarray(pt > 200)
        smear_pt_350_inf = np.random.choice([1, -1], mask_pt_350_inf.shape, p=[0.92, 0.08])
        smear_sig = mask_pt_0_200 * smear_pt_0_200 + mask_pt_200_350 * smear_pt_200_350 + mask_pt_350_inf * smear_pt_350_inf
        charge_sig = (smear_sig * charge) * np.logical_not(np.asarray(isBG))
        charge_bkg = np.asarray(isBG) * np.random.choice([1, -1], charge_sig.shape)
        charge_smeared = charge_sig + charge_bkg
        return ak.Array(charge_smeared)

    """
    Processes the input jets to construct tau candidates.
    """

    def processJets(self, jets):
        isBG = jets["gen_jet_tau_decaymode"] <= -1
        genJetP4s = jets["gen_jet_p4s"]
        genVisTauP4s = jets["gen_jet_tau_p4s"]
        tauP4s = self._smearEnergy(genVisTauP4s, genJetP4s, np.asarray(isBG))
        dclass = self._calcClassifier(
            self._asP4(jets["gen_jet_tau_p4s"]).pt, self._asP4(jets["reco_jet_p4s"]).pt, np.asarray(isBG)
        )
        dmode_orig = self._conv_dMode(jets["gen_jet_tau_decaymode"])
        dmode_smeared = self._calcDmode(dmode_orig)
        charge_smeared = self._smearCharge(np.asarray(jets["gen_jet_tau_charge"]), isBG, self._asP4(genVisTauP4s).pt)
        return {
            "tau_p4s": tauP4s,
            "tauSigCand_p4s": jets["reco_cand_p4s"],
            "tauClassifier": dclass,
            "tau_charge": charge_smeared,
            "tau_decaymode": self._conv_dModeToGranular(dmode_smeared),
        }
