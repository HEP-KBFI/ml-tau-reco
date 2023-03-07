#!/usr/bin/python3

import torch
import os
import glob
import hydra
import awkward as ak
import multiprocessing
from omegaconf import DictConfig
from itertools import repeat
from oracleTauBuilder import OracleTauBuilder
from hpsTauBuilder import HPSTauBuilder
from build_grid import GridBuilder
from endtoend_simple import SimpleDNNTauBuilder
from endtoend_simple import TauEndToEndSimple, SelfAttentionLayer
from fastCMSTauBuilder import FastCMSTauBuilder
from LorentzNetTauBuilder import LorentzNetTauBuilder
from DeepTauBuilder import DeepTauBuilder
from deeptauTraining import DeepTau

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
    elif cfg.builder == "HPS":
        # builder = HPSTauBuilder(cfgFileName="./config/hpsAlgo_cfg.json", verbosity=cfg.verbosity)
        builder = HPSTauBuilder(cfgFileName="./config/hpsAlgo_woPtCuts_cfg.json", verbosity=cfg.verbosity)
    elif cfg.builder == "Grid":
        builder = GridBuilder(verbosity=cfg.verbosity)
    elif cfg.builder == "FastCMSTau":
        builder = FastCMSTauBuilder()
    elif cfg.builder == "SimpleDNN":
        pytorch_model = torch.load("data/model.pt", map_location=torch.device("cpu"))
        assert pytorch_model.__class__ == TauEndToEndSimple
        assert pytorch_model.nn_pf_mha[0].__class__ == SelfAttentionLayer
        builder = SimpleDNNTauBuilder(pytorch_model)
    elif cfg.builder == "LorentzNet":
        builder = LorentzNetTauBuilder(verbosity=cfg.verbosity)
    elif cfg.builder == "DeepTau":
        model = torch.load("data/model_deeptau_v2.pt", map_location=torch.device("cpu"))
        builder = DeepTauBuilder(model)
    builder.printConfig()
    algo_output_dir = os.path.join(os.path.expandvars(cfg.output_dir), cfg.builder)
    for sample in cfg.samples_to_process:
        output_dir = os.path.join(algo_output_dir, sample)
        samples_dir = cfg.samples[sample].output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(samples_dir):
            raise OSError("Ntuples do not exist: %s" % (samples_dir))
        if cfg.n_files == -1:
            n_files = None
        else:
            n_files = cfg.n_files
        input_paths = glob.glob(os.path.join(samples_dir, f"*{sample}*.parquet"))[:n_files]
        if cfg.use_multiprocessing:
            pool = multiprocessing.Pool(processes=8)
            pool.starmap(process_single_file, zip(input_paths, repeat(builder), repeat(output_dir)))
        else:
            for input_path in input_paths:
                process_single_file(input_path=input_path, builder=builder, output_dir=output_dir)


if __name__ == "__main__":
    build_taus()
