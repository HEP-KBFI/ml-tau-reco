
# Housekeeping

Activate the development environment
```
singularity shell -B /local/ -B /scratch-persistent /home/software/singularity/pytorch.simg:2023-01-25
```
To update the libraries in the development environment, see https://github.com/HEP-KBFI/singularity.

To ensure no formatting issues from people using different editors, run the following every time before committing
```
pre-commit run --all
```

The repository is set up to test the basic functionality of the code in a [Github action](https://github.com/HEP-KBFI/ml-tau-reco/actions/workflows/test.yml), configured in [test.yml](.github/workflows/test.yml) which launches [scripts/test_pipeline.sh](scripts/test_pipeline.sh).

# Jupyter notebook

Launch the notebook server on manivald once
```
[manivald] jupyter notebook
```
Note the port XXXX and the token. You may close the SSH session, since `screen` keeps your notebook server running.

Open an SSH tunnel for the notebook from your laptop to manivald, replacing XXXX and MYUSER:
```
[laptop] ssh -N -f -L XXXX:localhost:XXXX MYUSER@manivald.hep.kbfi.ee
```

Navigate from your laptop browser to the notebook address that begins with `https://localhost:XXXX`.

# Software dependencies

All the necessary packages are installed in `pytorch.simg:2023-01-24` available at `/home/software/singularity`.
