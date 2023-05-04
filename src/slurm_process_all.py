import os
import glob
import hydra
import subprocess
from textwrap import dedent
from omegaconf import DictConfig


def prepare_job_file(input_path, output_dir):
    """Writes the job file that will be executed by slurm

    Parameters:
    ----------
    input_path : str
        Path to the input file
    output_dir : str
        directory where output will be written

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    """
    err_dir = os.path.join(output_dir, "err")
    out_dir = os.path.join(output_dir, "out")
    job_dir = os.path.join(output_dir, "submission_scripts")
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)
    job_file = os.path.join(job_dir, f"{os.path.basename(input_path).split('.')[0]}.sh")
    run_script = os.path.join(os.path.expandvars("$PWD"), "src", "slurm_process_file.py")
    error_file = os.path.join(err_dir, f"{os.path.basename(input_path).split('.')[0]}.err")
    output_file = os.path.join(out_dir, f"{os.path.basename(input_path).split('.')[0]}.out")
    run_env_path = os.path.join(os.path.expandvars("$PWD"), "scripts", "run-env.sh")
    with open(job_file, "wt") as filehandle:
        filehandle.writelines(
            dedent(
                f"""
                #!/bin/bash
                #SBATCH --job-name=ntupelizer
                #SBATCH -p short
                #SBATCH --ntasks=1
                #SBATCH --time=0:05:00
                #SBATCH --cpus-per-task=1
                #SBATCH -e {error_file}
                #SBATCH -o {output_file}
                env
                date
                {run_env_path} python3 {run_script} +input_path={input_path} +output_dir={output_dir}
            """
            ).strip("\n")
        )
    return job_file


def execute_file_processing(input_path, output_dir):
    job_file = prepare_job_file(input_path, output_dir)
    subprocess.call(["sbatch", job_file])


@hydra.main(config_path="../config", config_name="ntupelizer")
def process_all_input_files(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    for sample in cfg.samples_to_process:
        output_dir = cfg.samples[sample].output_dir
        input_dir = cfg.samples[sample].input_dir
        os.makedirs(output_dir, exist_ok=True)
        input_wcp = os.path.join(input_dir, "*.root")
        if cfg.test_run:
            n_files = 3
        else:
            n_files = None
        input_paths = glob.glob(input_wcp)[:n_files]
        for path in input_paths:
            execute_file_processing(path, output_dir)


if __name__ == "__main__":
    process_all_input_files()
