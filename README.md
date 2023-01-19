# Jupyter notebook

Launch the notebook server on manivald once
```
[manivald] screen
[manivald] singularity exec -B /scratch -B /local -B /scratch-persistent /home/software/singularity/tf-2.11.0.simg jupyter notebook
```
Note the port and the token. You may close the SSH session, since `screen` keeps your notebook server running.

Open an SSH tunnel for the notebook from your laptop to manivald:
```
[laptop] ssh -N -f -L localhost:XXXX:localhost:XXXX manivald.hep.kbfi.ee
```

Navigate from your laptop browser to the notebook `https://localhost:XXXX`.


# Software dependencies

All the necessary packages are installed in `tf-2.11.0.simg` available at `/home/software/singularity`.
To update this file, see: https://github.com/HEP-KBFI/singularity.
