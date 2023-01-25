import awkward as ak
import numpy as np
import sys
import uproot

if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("running:", sys.argv[0], infile, outfile)

    fi = uproot.open(infile)
    events = fi["events"]

    # ...

    reco_jet_p4s = np.zeros((100000, 4), dtype=np.float32)

    ak.to_parquet(ak.Record({"reco_jet_p4s": reco_jet_p4s}), outfile)
