import argparse
import subprocess
import time
import os
import json
import shlex


import shlex

BASH_FILES = "../scripts/test_generate_targets_all.sh"


script_args_dict = {}

# Read the bash command from the file
for bsh_f in BASH_FILES:
    with open(bsh_f, "r") as file:
        bash_commands = file.readlines()
    for bash_command in bash_commands:
        if bash_command.strip().startswith("python"):
            split_command = shlex.split(bash_command)
            script_name = split_command[1]
            args_list = split_command[2:]
            script_args_dict[script_name] = args_list

print(script_args_dict)


parser = argparse.ArgumentParser()
parser.add_argument("--dryrun", action="store_true", help="enable dryrun")
args = parser.parse_args()
dryrun = args.dryrun


""" 
-----------------------------------------------------------------------------
------------------------------ DEFINE PATHS ---------------------------------
-----------------------------------------------------------------------------
"""
ROOT_DIR = os.getenv("BHF_ROOT")
SRC_DIR = f"{ROOT_DIR}/src"
MIMIC_EXTRACT_DIR = f"{ROOT_DIR}/JSS_SUBMISSION_NEW"
NOTE_DIR = os.getenv("NOTE_DIR")
RADIOLOGY_PATH = os.path.join(NOTE_DIR, "radiology.csv")
DISCHARGE_PATH = os.path.join(NOTE_DIR, "discharge.csv")
LOGS_DIR = f'{ROOT_DIR}/logs/{"dryrun" if dryrun else ""}'


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

""" 
-----------------------------------------------------------------------------
----------------------------- SUPPLY BASH SCRIPTS ---------------------------
-----------------------------------------------------------------------------
"""

BASH_SCRIPTS = [
    "add_notes.sh",
    "generate_targets.sh",
    "finetune1.sh",
    "extract_embeddings1.sh",
]

# Define an empty list to store the runtimes
pipeline_runtimes = []

""" 
-----------------------------------------------------------------------------
----------------------------- RUN SCRIPTS -----------------------------------
-----------------------------------------------------------------------------
"""
# Loop through each script and time its execution
for script, args, do_run in scripts_to_run:
    # Construct the command-line arguments
    cmd_args = ["python", script] + args

    if not do_run:
        print(f"skipping {script}")
        continue
    elif dryrun:
        print(f"dry run: not calling {cmd_args}")
        continue

    start_time = time.time()
    # Open a log file for the script output
    log_file = open(os.path.join(LOGS_DIR, f"{os.path.basename(script)}.log"), "w")
    # Call the script using subprocess and redirect output to the log file
    print(f"calling {cmd_args}")
    retcode = subprocess.call(cmd_args, stdout=log_file, stderr=log_file)
    if retcode != 0:
        print(f"error on script {script}. Terminating pipeline")
        break
    print(f"finished {script}")
    # Close the log file
    log_file.close()
    # Measure the elapsed time
    elapsed_time = time.time() - start_time
    # Add the runtime to the list
    pipeline_runtimes.append(elapsed_time)
    print(f"script {script} runtime: {elapsed_time}")

print("Pipeline runtimes:", pipeline_runtimes)
