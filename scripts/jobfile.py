#python3 jobfile.py -i /scratch-persistent/veelken/CLIC_tau_ntuples/2023Mar09_woPtCuts/HPS/ZH_Htautau/ -o /local/snandan/CLIC_tau_ntuples/
import glob
import getpass
import os
import sys
import argparse
from textwrap import dedent

parser = argparse.ArgumentParser()
parser.add_argument("-i", dest="inputpath", help="inputpath")
parser.add_argument("-o", dest="outputdir", help="outputdir")
parser.add_argument("-a", dest="algo", help="algo", choices=['DeepTau', 'HPS', 'Grid'])
parser.add_argument("-n", type=int, dest="nfiles", help="number of files to be processed in each job")

options = parser.parse_args()
inputpath = options.inputpath
outputdir = options.outputdir
algo = options.algo
nfiles = options.nfiles
sample = inputpath.split('/')[-2]
outputdir = os.path.join(outputdir, algo, sample)
os.makedirs(outputdir, exist_ok=True)
user = getpass.getuser()

def create_batchfile(cmd, idx):
    output_dir = os.path.join('/home/snandan', algo)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'error'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'out'), exist_ok=True)
    scratch_dir = os.path.join('/scratch-persistent', 'snandan', algo)
    os.makedirs(scratch_dir, exist_ok=True)
    job_file = os.path.join(scratch_dir, str(idx) + '.sh')
    error_file = os.path.join(output_dir, 'error', str(idx))
    output_file = os.path.join(output_dir, 'out', str(idx))
    mkdir = 'mkdir dir_$SLURM_JOBID'
    rmdir = 'rm -r dir_$SLURM_JOBID'
    with open(job_file, 'wt') as filehandle:
        filehandle.writelines(dedent(
            """
            #!/bin/bash
            #SBATCH --job-name=%s_%s
            #SBATCH --ntasks=1
            #SBATCH --cpus-per-task=1
            #SBATCH --partition=short
            #SBATCH -e %s
            #SBATCH -o %s
            %s
            %s
            %s
            """ % (
                algo, idx, error_file, output_file,
                mkdir, cmd, rmdir
            )
        ).strip('\n'))
        return job_file

files = glob.glob(f'{inputpath}/*')
for idx, f in enumerate(range(0, len(files)+1, nfiles)):
    n_files = f + nfiles
    cmd = f"/home/{user}/mltaureco/ml-tau-reco/scripts/run-env.sh python3 /home/{user}/mltaureco/ml-tau-reco/src/runBuilder.py builder={algo} samples_to_process=['{sample}'] n_files={n_files} start={f} output_dir=dir_$SLURM_JOBID verbosity=1 samples.{sample}.output_dir={inputpath} use_multiprocessing=False"
    cmd += f" && mv dir_$SLURM_JOBID/{algo}/{sample}/* {outputdir}"
    jobfile = create_batchfile(cmd, idx)
    os.system(f'sbatch {jobfile}')
