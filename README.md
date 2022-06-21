# Setup of the repository
```
#for git-lfs support
export PATH=/home/software/bin:$PATH

git clone https://github.com/HEP-KBFI/ml-tau-reco.git
cd ml-tau-reco

git lfs install
git lfs pull 
```
# Jupyter notebook

Launch the notebook server on manivald once
```
[manivald] screen
[manivald] singularity exec -B /scratch -B /hdfs -B /scratch-persistent /home/software/singularity/tf-2.9.0.simg jupyter notebook
```
Note the port and the token. You may close the SSH session, since `screen` keeps your notebook server running.

Open an SSH tunnel for the notebook from your laptop to manivald:
```
[laptop] ssh -N -f -L localhost:XXXX:localhost:XXXX manivald.hep.kbfi.ee
```

Navigate from your laptop browser to the notebook `https://localhost:XXXX`.
