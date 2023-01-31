# Installation

git clone git@github.com:HEP-KBFI/ml-tau-reco.git

(as we run everything using a common singularity image no further software installation is needed)

# Housekeeping

The development environment (all system and python libraries) is defined in `scripts/run-env.sh`.

Run a short test of the full pipeline using
```
./scripts/run-env.sh ./scripts/test_pipeline.sh
```
To update the libraries in the development environment, see https://github.com/HEP-KBFI/singularity.

To ensure no formatting issues from people using different editors, run the following every time before committing
```
./scripts/run-env.sh pre-commit run --all
```

The repository is set up to test the basic functionality of the code in a [Github action](https://github.com/HEP-KBFI/ml-tau-reco/actions/workflows/test.yml), configured in [test.yml](.github/workflows/test.yml) which launches [scripts/test_pipeline.sh](scripts/test_pipeline.sh).

# Jupyter notebook

Launch the notebook server on manivald once
```
[manivald] ./scripts/run-env.sh jupyter notebook
```
Note the port XXXX and the token. You may close the SSH session, since `screen` keeps your notebook server running.

Open an SSH tunnel for the notebook from your laptop to manivald, replacing XXXX and MYUSER:
```
[laptop] ssh -N -f -L XXXX:localhost:XXXX MYUSER@manivald.hep.kbfi.ee
```
Navigate from your laptop browser to the notebook address that begins with `https://localhost:XXXX`.

# Producing the general Ntuples

TODO

# Running your tauBuilder

To run your tauBuilder code on the general Ntuples and produce tauBuilder tuples for the metric evaluation adapt ```src/runTauBuilder.py``` and run:
```
[manivald] ./scripts/run-env.sh src/runTauBuilder.py -n NFILES -i NTUPLEINPUTDIR -o OUTPUTTUPLEDIR -b YOURBUILDERCLASS
```
to run on all available ntuples run with ``` -n -1 ```

# Running the metric checks on the tauBuilder tuples

TODO
