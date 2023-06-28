""" Counts number of signal and background in the datasets """
import os
import json
import hydra
from omegaconf import DictConfig
from general import load_data_from_paths


@hydra.main(config_path="../config", config_name="ml_datasets", version_base=None)
def main(cfg: DictConfig) -> None:
    dataset_info = {
        "train": calculate_bkg_and_sig_size("train", cfg),
        "validation": calculate_bkg_and_sig_size("validation", cfg),
        "test": calculate_bkg_and_sig_size("test", cfg),
    }
    for dataset in ["train", "test", "validation"]:
        print(
            f"For {dataset} dataset there is {dataset_info[dataset]['nSig']} ZH_Htautau signal, \
            {dataset_info[dataset]['ZH_Htautau_nBkg']} ZH_Htautau background and \
            {dataset_info[dataset]['QCD_nBkg']} QCD background"
        )
    dataset_info["total"] = {
        "nSig": sum([dataset_info[dataset]["nSig"] for dataset in dataset_info.keys()]),
        "ZH_Htautau_nBkg": sum([dataset_info[dataset]["ZH_Htautau_nBkg"] for dataset in dataset_info.keys()]),
        "QCD_nBkg": sum([dataset_info[dataset]["QCD_nBkg"] for dataset in dataset_info.keys()]),
    }
    print(json.dumps(dataset_info, indent=4))
    output_path = os.path.join(os.path.expandvars(cfg.list_dir), "dataset_sizes.json")
    with open(output_path, "wt") as out_file:
        json.dump(dataset_info, out_file, indent=4)


def calculate_bkg_and_sig_size(dataset, cfg):
    sig_paths, bkg_paths = get_paths(dataset, cfg)
    sig_data = load_data_from_paths(sig_paths, columns=['gen_jet_tau_decaymode'])
    bkg_data = load_data_from_paths(bkg_paths, columns=['gen_jet_tau_decaymode'])
    n_sig = sum(sig_data.gen_jet_tau_decaymode != -1)
    n_bkg = len(bkg_data.gen_jet_tau_decaymode)
    ZH_Htautau_bkg = sum(sig_data.gen_jet_tau_decaymode == -1)
    info = {"nSig": int(n_sig), "ZH_Htautau_nBkg": ZH_Htautau_bkg, "QCD_nBkg": int(n_bkg)}
    return info


def get_paths(dataset, cfg):
    sig_paths = [path for path in cfg.datasets[dataset]["paths"] if "ZH_Htautau" in path]
    bkg_paths = [path for path in cfg.datasets[dataset]["paths"] if "QCD" in path]
    return sig_paths, bkg_paths


if __name__ == "__main__":
    main()
