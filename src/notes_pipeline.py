# PIPELINE:
#     add_notes.py
#     generate_targets.ipynb
#     tokenize_notes.py
#     finetune1.py
#     extract_embeddings1.py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dryrun', action='store_true', help='enable dryrun')
args = parser.parse_args()
dryrun = args.dryrun


import subprocess
import time
import os




# Define the list of scripts to run
mimic_extract_dir = '../JSS_SUBMISSION_NEW'
note_dir = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/'
clinical_dir = '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0/'
radiology_path = os.path.join(note_dir, 'radiology.csv')
discharge_path = os.path.join(note_dir, 'discharge.csv')
log_dir = f'./logs/{"dryrun" if dryrun else ""}'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
'''
add_notes: inserts note data for each 
'''
script_list = [
    (os.path.join(mimic_extract_dir, 'add_notes.py'), [], True), # -- test --note-type 
    (os.path.join(clinical_dir, 'generate_targets.py'), [], True),
    (os.path.join(clinical_dir, 'tokenize_notes.py'), [], True),
    (os.path.join(clinical_dir, 'finetune1.py'), ['--test','--gpu-no', '1'], True), # --test --gpu-no --model-ckpt-path # FIXME add model ckpt path if necessary
    (os.path.join(clinical_dir, 'extract_embeddings1.py'), ['--test', '--gpu-no', '1', '--ft-model-path',
        '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0/model_rad/meta_ft_classify.pt'], True), # --test --gpu-no --ft-model-path FIXME: select correct finetuned model
]
# Define an empty list to store the runtimes
pipeline_runtimes = []

# Loop through each script and time its execution
for script, args, do_run in script_list:
    # Construct the command-line arguments
    cmd_args = ['python', script] + args

    if (not do_run):
        print(f"skipping {script}")
        continue
    elif (dryrun):
        print(f"dry run: not calling {cmd_args}")
        continue

    start_time = time.time()
    # Open a log file for the script output
    log_file = open(os.path.join(log_dir, f'{os.path.basename(script)}.log'), 'w')
    # Call the script using subprocess and redirect output to the log file
    print(f"calling {cmd_args}")
    retcode = subprocess.call(cmd_args, stdout=log_file, stderr=log_file)
    if (retcode != 0):
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

print('Pipeline runtimes:', pipeline_runtimes)