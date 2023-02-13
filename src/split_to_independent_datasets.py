import os
import yaml
import glob
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="ml_datasets", version_base=None)
def split_to_datasets(cfg: DictConfig) -> None:
    list_dir = os.path.expandvars(cfg.list_dir)
    total = sum([cfg[dataset] for dataset in ["train", "test", "validation"]])
    fractions = {dataset: cfg[dataset] / total for dataset in ["train", "test", "validation"]}
    datasets = {dataset: [] for dataset in ["train", "test", "validation"]}
    for sample in cfg.samples_to_process:
        sample_paths = glob.glob(os.path.join(cfg.samples[sample].output_dir, "*"))
        n_files_in_sample = len(sample_paths)
        n_train_files = int(n_files_in_sample * fractions["train"])
        n_test_files = int(n_files_in_sample * fractions["test"])
        datasets["train"].extend(sample_paths[:n_train_files])
        datasets["test"].extend(sample_paths[n_train_files : n_train_files + n_test_files])
        datasets["validation"].extend(sample_paths[n_train_files + n_test_files :])
    for dataset in ["train", "test", "validation"]:
        output_path = os.path.join(list_dir, f"{dataset}.yaml")
        output_info = {dataset: {"paths": datasets[dataset]}}
        with open(output_path, "wt") as out_file:
            yaml.dump(output_info, out_file)


if __name__ == "__main__":
    split_to_datasets()
