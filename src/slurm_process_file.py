import hydra
from omegaconf import DictConfig
import edm4hep_to_ntuple as nt


@hydra.main(config_path="../config", config_name="ntupelizer", version_base=None)
def main(cfg: DictConfig) -> None:
    nt.process_single_file(input_path=cfg.input_path, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()