import sys
import awkward

if __name__ == "__main__":
    fi = awkward.from_parquet(sys.argv[1])
