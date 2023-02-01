#!/usr/bin/python3

from oracleTauBuilder import OracleTauBuilder
from hpsTauBuilder import HPSTauBuilder
from endtoend_simple import SimpleDNNTauBuilder, TauEndToEndSimple
import argparse
import os
import glob
import awkward as ak

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", "-b", type=str, choices=["oracle", "hps", "simplednn"], default="oracle")
    parser.add_argument("--input", "-i", type=str, default="/local/laurits/CLIC_data/")
    parser.add_argument("--output", "-o", type=str, default="/local/tolange/CLIC_oracle/")
    parser.add_argument("--nFiles", "-n", type=int, default=1)
    args = parser.parse_args()
    
    builder = None
    
    if args.builder == "oracle":
        builder = OracleTauBuilder()
        builder.printConfig()
    elif args.builder == "hps":
        builder = HPSTauBuilder()
        builder.printConfig()
    elif args.builder == "simplednn":
        import torch
        pytorch_model = torch.load("data/model.pt")
        builder = SimpleDNNTauBuilder(pytorch_model)
        builder.printConfig()
    else:
        raise ValueError("This builder is not implemented: %s" % (args.builder))
    
    if not os.path.exists(args.input):
        raise OSError("Path does not exist: %s" % (args.input))
    
    if not os.path.exists(args.output):
        raise OSError("Path does not exist: %s" % (args.output))
    
    input_paths = glob.glob(os.path.join(args.input, "*.parquet"))[: args.nFiles]
    
    for p in input_paths:
        # load jets
        print("Load jets from", p)
        jets = ak.from_parquet(p)
        # process in tauBuilder
        print("Processing jets...")
        pjets = builder.processJets(jets)
        # saving for metric scripts
        outPath = os.path.join(args.output, os.path.split(p)[1])
        print("done, saving to ", outPath)
        ak.to_parquet(ak.Record(pjets), outPath)
