import vector
import numpy as np
import awkward as ak

from basicTauBuilder import BasicTauBuilder


class OracleTauBuilder(BasicTauBuilder):
    def __init__(
        self,
        config={
            "truthCandThreshold": 0.5,
            "sigProb": 0.8,
            "bkgProb": 0.05,
            "dclassCenterIso": 4.5,
            "goodPFEnergythrshold": 0.5,
        },
    ):
        self._builderConfig = dict()
        for key in config:
            self._builderConfig[key] = config[key]

    def _randomDraw(self, mask, goodMask, prob):
        mask_numpy = np.asarray(goodMask.layout.content)
        goodMask_numpy = np.asarray(mask.layout.content)
        mask_numpy = np.logical_and(mask_numpy, goodMask_numpy)
        mask_rand = np.random.rand(*mask_numpy.shape) < prob
        mask_new_numpy = np.logical_and(mask_numpy, mask_rand)
        mask_new = ak.Array(ak.contents.ListOffsetArray(mask.layout.offsets, ak.Array(mask_new_numpy).layout))
        return mask_new

    def _sig(self, x):
        return 1 / (1 + np.exp(-x))

    def _calcClassifier(self, tauP4, isoP4, chargedIsoP4, neutralIsoP4):
        tauEnergy_numpy = tauP4.energy
        isoEnergy_numpy = isoP4.energy
        chargedIsoEnergy_numpy = chargedIsoP4.energy
        neutralIsoEnergy_numpy = neutralIsoP4.energy
        isoTerm = (tauEnergy_numpy > 5) * (isoEnergy_numpy > 0.5) * isoEnergy_numpy / (tauEnergy_numpy + 0.001)
        chargedIsoTerm = (
            (tauEnergy_numpy > 5) * (chargedIsoEnergy_numpy > 0.1) * chargedIsoEnergy_numpy / (tauEnergy_numpy + 0.001)
        )
        neutralIsoTerm = (
            (tauEnergy_numpy > 5) * (neutralIsoEnergy_numpy > 0.5) * neutralIsoEnergy_numpy / (tauEnergy_numpy + 0.001)
        )
        maxIsoTerm = np.max((np.max((isoTerm, chargedIsoTerm), 0), neutralIsoTerm), 0)
        dclass_numpy = self._sig(-maxIsoTerm + self._builderConfig["dclassCenterIso"])
        dclass = ak.Array(dclass_numpy)
        return dclass

    def processJets(self, jets):
        candP4s = vector.awk(
            ak.zip(
                {
                    "mass": jets["reco_cand_p4s"].tau,
                    "x": jets["reco_cand_p4s"].x,
                    "y": jets["reco_cand_p4s"].y,
                    "z": jets["reco_cand_p4s"].z,
                }
            )
        )
        # quality citeria
        goodCands = candP4s.energy > self._builderConfig["goodPFEnergythrshold"]
        candEfrac = jets["reco_cand_matched_gen_energy"] / candP4s.energy
        sigMask = candEfrac > self._builderConfig["truthCandThreshold"]
        sigRandMask = self._randomDraw(sigMask, goodCands, self._builderConfig["sigProb"])
        bkgMask = candEfrac <= self._builderConfig["truthCandThreshold"]
        bkgRandMask = self._randomDraw(bkgMask, goodCands, self._builderConfig["bkgProb"])
        chargedMask = jets["reco_cand_charge"] != 0
        neutralMask = jets["reco_cand_charge"] == 0
        tauSelMask = ak.sum((sigRandMask, bkgRandMask), 0) == 1
        tauCandP4s = ak.mask(candP4s, tauSelMask)
        tauCandCharges = ak.mask(jets["reco_cand_charge"], tauSelMask)
        tauCharges = ak.sum(tauCandCharges, 1)
        isoCandP4s = ak.mask(candP4s, tauSelMask == 0)
        chargedIsoCandP4s = ak.mask(candP4s, ak.sum((tauSelMask == 0, chargedMask), 0) == 1)
        neutralIsoCandP4s = ak.mask(candP4s, ak.sum((tauSelMask == 0, neutralMask), 0) == 1)
        tauP4 = vector.awk(
            ak.zip(
                {
                    "px": ak.sum(tauCandP4s.px, axis=1),
                    "py": ak.sum(tauCandP4s.py, axis=1),
                    "pz": ak.sum(tauCandP4s.pz, axis=1),
                    "mass": ak.sum(tauCandP4s.mass, axis=1),
                }
            )
        )
        isoP4 = vector.awk(
            ak.zip(
                {
                    "px": ak.sum(isoCandP4s.px, axis=1),
                    "py": ak.sum(isoCandP4s.py, axis=1),
                    "pz": ak.sum(isoCandP4s.pz, axis=1),
                    "mass": ak.sum(isoCandP4s.mass, axis=1),
                }
            )
        )
        chargedIsoP4 = vector.awk(
            ak.zip(
                {
                    "px": ak.sum(chargedIsoCandP4s.px, axis=1),
                    "py": ak.sum(chargedIsoCandP4s.py, axis=1),
                    "pz": ak.sum(chargedIsoCandP4s.pz, axis=1),
                    "mass": ak.sum(chargedIsoCandP4s.mass, axis=1),
                }
            )
        )
        neutralIsoP4 = vector.awk(
            ak.zip(
                {
                    "px": ak.sum(neutralIsoCandP4s.px, axis=1),
                    "py": ak.sum(neutralIsoCandP4s.py, axis=1),
                    "pz": ak.sum(neutralIsoCandP4s.pz, axis=1),
                    "mass": ak.sum(neutralIsoCandP4s.mass, axis=1),
                }
            )
        )
        dclass = self._calcClassifier(tauP4, isoP4, chargedIsoP4, neutralIsoP4)
        dmode = self._get_decayModes(ak.mask(jets["reco_cand_pdg"], tauSelMask))
        return {
            "tau_p4": tauP4,
            "tauSigCand_p4s": tauCandP4s,
            "tauClassifier": dclass,
            "tau_charge": tauCharges,
            "tau_decaymode": dmode,
        }
