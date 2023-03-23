import os
import glob
import vector
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
    if n_files == -1:
        n_files = None
    input_dir = os.path.expandvars(input_dir)
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))[:n_files]
    input_data = []
    for file_path in input_files:
        print(f"Loading from {file_path}")
        input_data.append(ak.Array((ak.from_parquet(file_path).tolist())))
    input_data = ak.concatenate(input_data)
    print("Input data loaded")
    return input_data


def load_data_from_paths(input_paths: list, n_files: int = None) -> ak.Array:
    """Loads all .parquet files from a given directory

    Args:
        input_dir : list
            The .parquet files where the data is loaded
        n_files : int
            Number of files to load from the given input directory.
            By default all will be loaded [default: None]

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    input_data = []
    for file_path in input_paths[:n_files]:
        print(f"Loading from {file_path}")
        try:
            input_data.append(ak.Array((ak.from_parquet(file_path).tolist())))
        except ValueError:
            print(f"{file_path} does not exist")
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
        16: 'LeptonicDecay'
    }
    0: [0, 5, 10]
    1: [1, 6, 11]
    2: [2, 3, 4, 7, 8, 9, 12, 13, 14, 15]
    """
    pdg_ids = np.abs(np.array(pdg_ids))
    unique, counts = np.unique(pdg_ids, return_counts=True)
    p_counts = {i: 0 for i in [16, 111, 211, 13, 14, 12, 11, 22]}
    p_counts.update(dict(zip(unique, counts)))
    if np.sum(p_counts[211]) == 1 and p_counts[111] == 0:
        return 0
    elif np.sum(p_counts[211]) == 1 and p_counts[111] == 1:
        return 1
    elif np.sum(p_counts[211]) == 1 and p_counts[111] == 2:
        return 2
    elif np.sum(p_counts[211]) == 1 and p_counts[111] == 3:
        return 3
    elif np.sum(p_counts[211]) == 1 and p_counts[111] > 3:
        return 4
    elif np.sum(p_counts[211]) == 2 and p_counts[111] == 0:
        return 5
    elif np.sum(p_counts[211]) == 2 and p_counts[111] == 1:
        return 6
    elif np.sum(p_counts[211]) == 2 and p_counts[111] == 2:
        return 7
    elif np.sum(p_counts[211]) == 2 and p_counts[111] == 3:
        return 8
    elif np.sum(p_counts[211]) == 2 and p_counts[111] > 3:
        return 9
    elif np.sum(p_counts[211]) == 3 and p_counts[111] == 0:
        return 10
    elif np.sum(p_counts[211]) == 3 and p_counts[111] == 1:
        return 11
    elif np.sum(p_counts[211]) == 3 and p_counts[111] == 2:
        return 12
    elif np.sum(p_counts[211]) == 3 and p_counts[111] == 3:
        return 13
    elif np.sum(p_counts[211]) == 3 and p_counts[111] > 3:
        return 14
    elif np.sum(p_counts[11] + p_counts[13]) > 0:
        return 16
    else:
        return 15


def get_reduced_decaymodes(decaymodes: np.array):
    """Maps the full set of decay modes into a smaller subset, setting the rarer decaymodes under "Other" (# 15)"""
    target_mapping = {
        -1: -1,
        0: 0,
        1: 1,
        2: 2,
        3: 15,
        4: 15,
        5: 15,
        6: 15,
        7: 15,
        8: 15,
        9: 15,
        10: 10,
        11: 11,
        12: 15,
        13: 15,
        14: 15,
        15: 15,
    }
    return np.vectorize(target_mapping.get)(decaymodes)


def reinitialize_p4(p4_obj):
    if 't' in p4_obj.fields:
        p4 = vector.awk(
            ak.zip(
                {
                    "energy": p4_obj.t,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    else:
        p4 = vector.awk(
            ak.zip(
                {
                    "mass": p4_obj.tau,
                    "x": p4_obj.x,
                    "y": p4_obj.y,
                    "z": p4_obj.z,
                }
            )
        )
    return p4