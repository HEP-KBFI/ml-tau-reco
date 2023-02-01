import numpy as np
import awkward as ak
from abc import ABC, abstractmethod

class BasicTauBuilder(ABC):
    def __init__(self, config=dict()):
        self._builderConfig = config

    def _get_decayMode(self, pdg_ids):
        """Tau decaymodes are the following:
        decay_mode_mapping = {
            0: 'OneProng0PiZero',
            1: 'OneProng1PiZero',
            2: 'OneProng2PiZero',
            3: 'OneProng3PiZero',
            4: 'OneProngNPiZero',
            5: 'TwoProng0PiZero',
            6: 'TwoProng1PiZero',
            7: 'TwoProng2PiZero',
            8: 'TwoProng3PiZero',
            9: 'TwoProngNPiZero',
            10: 'ThreeProng0PiZero',
            11: 'ThreeProng1PiZero',
            12: 'ThreeProng2PiZero',
            13: 'ThreeProng3PiZero',
            14: 'ThreeProngNPiZero',
            15: 'RareDecayMode'
        }
        0: [0, 5, 10]
        1: [1, 6, 11]
        2: [2, 3, 4, 7, 8, 9, 12, 13, 14, 15]
        """
        pdg_ids = np.abs(np.asarray(pdg_ids.layout.content))
        unique, counts = np.unique(pdg_ids, return_counts=True)
        p_counts = {i: 0 for i in [16, 111, 211, 13, 14, 12, 11, 22]}
        p_counts.update(dict(zip(unique, counts)))
        if np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 0:
            return 0
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 1:
            return 1
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 2:
            return 2
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] == 3:
            return 3
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 1 and p_counts[111] > 3:
            return 4
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 0:
            return 5
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 1:
            return 6
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 2:
            return 7
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] == 3:
            return 8
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 2 and p_counts[111] > 3:
            return 9
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 0:
            return 10
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 1:
            return 11
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 2:
            return 12
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] == 3:
            return 13
        elif np.sum(p_counts[211] + p_counts[11] + p_counts[13]) == 3 and p_counts[111] > 3:
            return 14
        else:
            return 15

    def _get_decayModes(self, taus_pdg_ids):
        return ak.Array([self._get_decayMode(cand_pdg_ids) for cand_pdg_ids in taus_pdg_ids])

    def printConfig(self):
        print("Running %s with config:" % self.__class__.__name__)
        print(self._builderConfig)

    @property
    def builderConfig(self):
        return self._builderConfig

    @abstractmethod
    def processJets(self, jets):
        pass
