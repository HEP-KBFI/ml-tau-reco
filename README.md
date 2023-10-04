

[![DOI](https://zenodo.org/badge/505773250.svg)](https://zenodo.org/badge/latestdoi/505773250)


# Installation

```
git clone git@github.com:HEP-KBFI/ml-tau-reco.git
```

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

# Contributing

In order to push code, put your new code in a new branch and push it:
```
git checkout main
git pull origin
git checkout -b my_new_feature_branch
git commit ...
git push origin my_new_feature_branch
```
Then open a PR on github for your new branch. Basic tests should pass and your code should run in the tests to ensure it's usable by others.

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

The general command to produce the ntuples is
```bash
./scripts/run-env.sh python3 src/edm4hep_to_ntuple.py
```
It takes all the configuration settings from ```config/ntupelizer```. All the parameters can be replaced on commandline.

The same configuration file is used to check the validity of the ntuple. Validation script is run as follows:

```bash
./scripts/run-env.sh python ./src/validation.py
```

Feel free to implement/suggest any other tests for validation as there are currently only the most basic ones.

# Running your tauBuilder

To run your tauBuilder code on the general Ntuples and produce tauBuilder tuples for the metric evaluation adapt ```src/runTauBuilder.py``` and run:
```
[manivald] ./scripts/run-env.sh python3 src/runBuilder.py builder=HPS n_files=1 verbosity=1 output_dir=/local/veelken/CLIC_tau_ntuples/$VERSION
```
it will run both datasets by default, so if you only want ZH_Htautau, then add also ```samples_to_process=['ZH_Htautau']``` to the end of the command


# Running the metric checks on the tauBuilder tuples

After updating in ```config/metrics``` the paths where the tauBuilder for a specific algorithm has written it's output one simply runs:

```bash
./scripts/run-env.sh python3 src/calculate_metrics.py
```

# Refferences and additional documentation

The code in this repo was used in the context of the paper "Identification of hadronic tau decays with neural-network architectures developed for jet-flavour tagging" containing more on the idividual algorithms implemented in this repo. The paper can be found in as a preprin [here](https://arxiv.org/abs/2307.07747).

Our implementaton of tau lepton impact parameters using the Key4HEP format is documented in [impactparameters/impact.pdf](impactparameters/impact.pdf)

