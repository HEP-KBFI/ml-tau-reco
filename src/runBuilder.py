#!/usr/bin/python3

import os
import glob
import hydra
import awkward as ak
import multiprocessing
from omegaconf import DictConfig
from itertools import repeat
from oracleTauBuilder import OracleTauBuilder
from hpsTauBuilder import HPSTauBuilder
from endtoend_simple import SimpleDNNTauBuilder


def process_single_file(input_path: str, builder, output_dir) -> None:
    print("Load jets from", input_path)
    jets = ak.from_parquet(input_path)
    print("Processing jets...")
    pjets = builder.processJets(jets)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    print("done, saving to ", output_path)
    merged_info = {field: jets[field] for field in jets.fields}
    merged_info.update(pjets)
    ak.to_parquet(ak.Record(merged_info), output_path)


@hydra.main(config_path="../config", config_name="tau_builder", version_base=None)
def build_taus(cfg: DictConfig) -> None:
    print("<runBuilder>:")
    if cfg.builder == "Oracle":
        builder = OracleTauBuilder()
        builder.printConfig()
    elif cfg.builder == "HPS":
        builder = HPSTauBuilder(verbosity=cfg.verbosity)
        builder.printConfig()
    algo_output_dir = os.path.join(os.path.expandvars(cfg.output_dir), cfg.builder)
    for sample in cfg.samples_to_process:
        output_dir = os.path.join(algo_output_dir, sample)
        samples_dir = cfg.samples[sample].output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(samples_dir):
            raise OSError("Ntuples do not exist: %s" % (samples_dir))
        input_paths = glob.glob(os.path.join(samples_dir, "*.parquet"))[: cfg.n_files]
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=8)
            pool.starmap(process_single_file, zip(input_paths, repeat(builder), repeat(output_dir)))
        else:
            for input_path in input_paths:
                process_single_file(input_path=input_path, builder=builder, output_dir=output_dir)


if __name__ == '__main__':
    build_taus()
