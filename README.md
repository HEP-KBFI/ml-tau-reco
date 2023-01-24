# Jupyter notebook

Launch the notebook server on manivald once
```
[manivald] screen
[manivald] singularity exec -B /scratch -B /local -B /scratch-persistent /home/software/singularity/pytorch.simg:2023-01-24 jupyter notebook
```
Note the port and the token. You may close the SSH session, since `screen` keeps your notebook server running.

Open an SSH tunnel for the notebook from your laptop to manivald:
```
[laptop] ssh -N -f -L XXXX:localhost:XXXX manivald.hep.kbfi.ee
```

Navigate from your laptop browser to the notebook `https://localhost:XXXX`.


# Software dependencies

All the necessary packages are installed in `pytorch.simg:2023-01-24` available at `/home/software/singularity`.
To update this file, see: https://github.com/HEP-KBFI/singularity.


# Code formatting

To ensure no issues with people using different editors, run the following
```
singularity exec /home/software/singularity/pytorch.simg:2023-01-24 pre-commit run --all
```
