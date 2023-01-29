import os
import glob
import awkward as ak


def load_all_data(input_dir: str) -> ak.Array:
    """Loads all .parquet files from a given directory

    Args:
        input_dir : str
            The directory where the .parquet files are located

    Returns:
        input_data : ak.Array
            The concatenated data from all the loaded files
    """
    input_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    input_data = []
    for file_path in input_files:
        input_data.append(ak.Array((ak.from_parquet(file_path).tolist())))
    input_data = ak.concatenate(input_data)
    return input_data
