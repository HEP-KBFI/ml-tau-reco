import os
import glob
import numpy as np
import awkward as ak


def load_all_data(input_dir: str, n_files: int = None) -> ak.Array:
    """Loads all .parquet files from a given directory

    Args:
        input_dir : str
            The directory where the .parquet files are located
        n_files : int
            Number of files to load from the given input directory.
            By default all will be loaded [default: None]

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))[:n_files]
    input_data = []
    for file_path in input_files:
        input_data.append(ak.Array((ak.from_parquet(file_path).tolist())))
    input_data = ak.concatenate(input_data)
    return input_data


def get_decaymode(pdg_ids):
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
    pdg_ids = np.abs(np.array(pdg_ids))
    unique, counts = np.unique(pdg_ids, return_counts=True)
    p_counts = {i: 0 for i in [16, 111, 211, 13, 14, 12, 11, 22]}
    p_counts.update(dict(zip(unique, counts)))
    if check_rare_decaymode(pdg_ids):
        return 15
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


def check_rare_decaymode(pdg_ids):
    """The common particles in order are tau-neutrino, pi0, pi+, mu,
    mu-neutrino, electron-neutrino, electron, photon"""
    common_particles = [16, 111, 211, 13, 14, 12, 11, 22]
    return sum(np.in1d(pdg_ids, common_particles)) != len(pdg_ids)
