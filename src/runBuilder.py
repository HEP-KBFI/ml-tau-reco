#!/usr/bin/python3

from oracleTauBuilder import OracleTauBuilder
from hpsTauBuilder import HPSTauBuilder
from endtoend_simple import SimpleDNNTauBuilder
import argparse
import os
import glob
import awkward as ak


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", "-b", type=str, choices=["oracle", "hps", "simplednn"], default="oracle")
    parser.add_argument("--input", "-i", type=str, default="/local/laurits/CLIC_data/ZH_Htautau")
    parser.add_argument("--output", "-o", type=str, default="/local/$USER/CLIC_tau_ntuples/")
    parser.add_argument("--nFiles", "-n", type=int, default=1)
    parser.add_argument("--verbosity", "-v", type=int, default=0)
    args = parser.parse_args()

    print("<runBuilder>:")

    builder = None

    if args.builder == "oracle":
        builder = OracleTauBuilder()
        builder.printConfig()
    elif args.builder == "hps":
        builder = HPSTauBuilder(verbosity=args.verbosity)
        builder.printConfig()
    elif args.builder == "simplednn":
        import torch
        from endtoend_simple import TauEndToEndSimple

        pytorch_model = torch.load("data/model.pt")
        assert pytorch_model.__class__ == TauEndToEndSimple
        builder = SimpleDNNTauBuilder(pytorch_model)
        builder.printConfig()
    else:
        raise ValueError("This builder is not implemented: %s" % (args.builder))

    if not os.path.exists(args.input):
        raise OSError("Path does not exist: %s" % (args.input))

    input_paths = glob.glob(os.path.join(args.input, "*.parquet"))[: args.nFiles]

    output_dir = os.path.join(os.path.expandvars(args.output), args.builder)
    os.makedirs(output_dir, exist_ok=True)

    for path in input_paths:
        # load jets
        print("Load jets from", path)
        jets = ak.from_parquet(path)
        # process in tauBuilder
        print("Processing jets...")
        pjets = builder.processJets(jets)
        # saving for metric scripts
        output_path = os.path.join(output_dir, os.path.basename(path))
        print("done, saving to ", output_path)
        merged_info = {field: jets[field] for field in jets.fields}
        merged_info.update(pjets)
        ak.to_parquet(ak.Record(merged_info), output_path)
