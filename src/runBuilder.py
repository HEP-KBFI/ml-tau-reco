#!/usr/bin/python3

from oracleTauBuilder import OracleTauBuilder
from hpsTauBuilder import HPSTauBuilder
import argparse
import os
import glob
import awkward as ak
import getpass

parser = argparse.ArgumentParser()
parser.add_argument("--builder", "-b", type=str, choices=["oracle", "hps"], default="oracle")
parser.add_argument("--input", "-i", type=str, default="/local/laurits/CLIC_data/")
parser.add_argument("--output", "-o", type=str, default="")
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
else:
    raise ValueError("This builder is not implemented: %s" % (args.builder))

if not os.path.exists(args.input):
    raise OSError("Path does not exist: %s" % (args.input))
input_paths = glob.glob(os.path.join(args.input, "*.parquet"))[: args.nFiles]
print(" input_paths = %s" % input_paths)

output_path = args.output
if output_path == "":
    output_path = "/local/%s/CLIC_oracle/" % getpass.getuser()
print(" output_path = %s" % output_path)
if not os.path.exists(output_path):
    raise OSError("Path does not exist: %s" % (output_path))

for p in input_paths:
    # load jets
    print("Load jets from", p)
    jets = ak.from_parquet(p)
    # process in tauBuilder
    print("Processing jets...")
    pjets = builder.processJets(jets)
    # saving for metric scripts
    outPath = os.path.join(output_path, os.path.split(p)[1])
    print("done, saving to ", outPath)
    ak.to_parquet(ak.Record(pjets), outPath)
