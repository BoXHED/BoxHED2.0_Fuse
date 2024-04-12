import argparse
import subprocess
import time
import os
import json
import shlex
import tkinter as tk
from BoXHED_Fuse.src.helpers import find_next_dir_index

""" 
-----------------------------------------------------------------------------
------------------------------ DEFINE PATHS ---------------------------------
-----------------------------------------------------------------------------
"""
ROOT_DIR = os.getenv("BHF_ROOT")
SRC_DIR = f"{ROOT_DIR}/src"
MIMIC_EXTRACT_DIR = f"{ROOT_DIR}/JSS_SUBMISSION"
NOTE_DIR = os.getenv("NOTE_DIR")
RADIOLOGY_PATH = os.path.join(NOTE_DIR, "radiology.csv")
DISCHARGE_PATH = os.path.join(NOTE_DIR, "discharge.csv")
LOGS_DIR = f"{SRC_DIR}/logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
run_cntr = str(find_next_dir_index(LOGS_DIR))
LOGS_DIR = os.path.join(LOGS_DIR, run_cntr)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)


for p in [
    ROOT_DIR,
    SRC_DIR,
    MIMIC_EXTRACT_DIR,
    NOTE_DIR,
    RADIOLOGY_PATH,
    DISCHARGE_PATH,
    LOGS_DIR,
]:
    assert os.path.exists(p)


def notes_pipeline(bash_files: list[str], run_script_list: list[bool]):
    logs_dir = f"{LOGS_DIR}/"
    print("LOGS DIR:", logs_dir)

    """
    -----------------------------------------------------------------------------
    ----------------------------- READ IN ARGS ----------------------------------
    -----------------------------------------------------------------------------
    """
    all_scripts = []
    all_args = []

    # Read the bash command from the file
    for bsh_f in bash_files:
        filepath = f"{os.getenv('BHF_ROOT')}/scripts/{bsh_f}"
        with open(filepath, "r") as file:
            bash_commands = file.readlines()
        for bash_command in bash_commands:
            if bash_command.strip().startswith("python"):
                split_command = shlex.split(bash_command)
                script_name = split_command[1]
                args_list = split_command[2:]
                all_scripts.append(script_name)
                all_args.append(args_list)

    """ 
    -----------------------------------------------------------------------------
    ----------------------------- RUN SCRIPTS -----------------------------------
    -----------------------------------------------------------------------------
    """

    pipeline_args = zip(all_scripts, all_args, run_script_list)
    # Define a dict to store the runtimes
    pipeline_runtimes = dict(zip(all_scripts, [-1 for _ in range(len(all_scripts))]))

    # Loop through each script and time its execution
    for script, args, do_run in pipeline_args:
        # Construct the command-line arguments
        cmd_args = ["python", script] + args

        if not do_run:
            print(f"skipping {script}")
            continue

        start_time = time.time()
        # Call the script using subprocess and redirect output to the log file
        print(f"calling {cmd_args}")
        basename = os.path.basename(script)
        log_path = os.path.join(logs_dir, f"{os.path.splitext(basename)[0]}.log")
        log_file = open(log_path, "w")
        retcode = subprocess.call(cmd_args, stdout=log_file, stderr=log_file)
        if retcode != 0:
            print(
                f"error on script {script}. See {log_path} for details. Terminating pipeline"
            )
            break
        print(f"\033[94mfinished {script}\033[0m")

        log_file.close()

        # Measure the elapsed time
        elapsed_time = time.time() - start_time
        # Add the runtime to the list
        pipeline_runtimes[script] = elapsed_time
        print(f"\033[92mscript {script} runtime: {elapsed_time}\033[0m")

    print(
        f"\033[93mPIPELINE FINISHED WITH {len([v for v in pipeline_runtimes.values() if v == -1])} ERRORS\033[0m"
    )
    print(f"\033[92mPipeline runtimes: {pipeline_runtimes})\033[0m")
    log_file = open(os.path.join(logs_dir, "notes_pipeline.log"), "w")
    log_file.write(str(pipeline_runtimes))
    log_file.close()


if __name__ == "__main__":

    # # This is a workaround to allow the subprocess to run (most likely not necessary)
    # os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

    # Bash scripts must be run in this order.
    # This example uses a default pipeline configuration
    # You may specify your own bash files here.
    DEFAULT_BASH_FILES = [
        "add_notes.sh",
        "generate_targets.sh",
        "finetune1.sh",
        "extract_embeddings1.sh",
    ]
    DEFAULT_RUN_SCRIPT_LIST = [True, True, True, True]

    parser = argparse.ArgumentParser(description="Run provided bash scripts.")
    parser.add_argument(
        "--bash-files",
        nargs="+",
        default=DEFAULT_BASH_FILES,
        help="List of bash file names. Ex: add_notes.sh generate_targets.sh finetune1.sh extract_embeddings1.sh",
    )
    parser.add_argument(
        "--run-scripts",
        nargs="+",
        type=bool,
        default=DEFAULT_RUN_SCRIPT_LIST,
        help="List of boolean values indicating whether to run or skip each script",
    )

    args = parser.parse_args()

    print("Bash files:", args.bash_files)
    print("Run scripts:", args.run_scripts)

    notes_pipeline(args.bash_files, run_script_list=args.run_scripts)
