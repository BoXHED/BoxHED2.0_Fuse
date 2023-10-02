def generate_stay_and_note_timing_table(relevant_subjects, stays, notes):
    
    # filter for useful notes and stays data
    def _filter_notes_and_stays(relevant_subjects, stays, notes):
        relevant_notes = notes[notes['SUBJECT_ID'].isin(relevant_subjects)]
        relevant_stays = stays[stays['SUBJECT_ID'].isin(relevant_subjects)]

        relevant_notes = relevant_notes[['SUBJECT_ID', 'NOTE_ID', 'charttime']]
        relevant_stays = relevant_stays[['SUBJECT_ID', 'ICUSTAY_ID', 'INTIME']]
        return (relevant_notes, relevant_stays)

    # concatenate notes and stays, grouping by subject and sorting by DATETIME
    def _concat_and_sort_table(relevant_notes, relevant_stays):
        stay_note_timing = pd.concat([relevant_notes, relevant_stays]).sort_values(by='SUBJECT_ID')
        stay_note_timing['DATETIME'] = stay_note_timing['INTIME'].combine_first(stay_note_timing['charttime'])
        stay_note_timing.drop(['charttime'], axis=1, inplace=True)
        stay_note_timing = stay_note_timing.groupby('SUBJECT_ID', group_keys=True).apply(lambda x: x.sort_values('DATETIME'))
        stay_note_timing.reset_index(drop=True, inplace=True)
        stay_note_timing.infer_objects()
        return stay_note_timing
    
    def _fillna_icustay_by_subject(stay_note_timing_by_subject):
        stay_note_timing_by_subject['ICUSTAY_ID'].fillna(method='ffill', inplace=True) # for all notes, fill icustay_id with the most recent icustay.
        stay_note_timing_by_subject['ICUSTAY_ID'].fillna(method='bfill', inplace=True) # for all notes, fill icustay_id with the most recent icustay.        
        return stay_note_timing_by_subject

    
    def _fillna_intime_by_stay(stay_note_timing_by_stay):
        stay_note_timing_by_stay['INTIME'].fillna(method='ffill', inplace=True)
        stay_note_timing_by_stay['INTIME'].fillna(method='bfill', inplace=True)
        return stay_note_timing_by_stay
    

    def _populate_accum_timediff(stay_note_timing_by_subject):
        first_intime = stay_note_timing_by_subject.iloc[0]['INTIME'] # first intime for this subject
        stay_note_timing_by_subject['accum_timediff'] = stay_note_timing['DATETIME'] - first_intime
        stay_note_timing_by_subject['accum_timediff'] = stay_note_timing_by_subject['accum_timediff'].apply(lambda x: x.total_seconds()/3600)
        return stay_note_timing_by_subject

        
    relevant_notes, relevant_stays = _filter_notes_and_stays(relevant_subjects = relevant_subjects, stays = stays, notes = notes)
    stay_note_timing = _concat_and_sort_table(relevant_notes, relevant_stays)
    # stay_note_timing.rename(columns={'note_id':'NOTE_ID'}, inplace=True)

    stay_note_timing = stay_note_timing.groupby('SUBJECT_ID', group_keys=True).apply(_fillna_icustay_by_subject)
    stay_note_timing = stay_note_timing.groupby('ICUSTAY_ID', group_keys=True).apply(_fillna_intime_by_stay)
    
    stay_note_timing['DATETIME'] = pd.to_datetime(stay_note_timing['DATETIME'])
    stay_note_timing['INTIME'] = pd.to_datetime(stay_note_timing['INTIME'])
    # stay_note_timing['rel_timediff'] = (stay_note_timing['DATETIME'] - stay_note_timing['INTIME'])
    # stay_note_timing['rel_timediff'] = stay_note_timing['rel_timediff'].apply(lambda x: x.total_seconds()/3600)

    # stay_note_timing = stay_note_timing.groupby('SUBJECT_ID').apply(_populate_accum_timediff)
    stay_note_timing = stay_note_timing.convert_dtypes(infer_objects=True)

    return stay_note_timing

def _insert_note_data_by_stay(mimic_iv_train_per_stay):
    '''
    for time series data for a single icustay, populate most recent note id, returning None if there are no previous notes

    args:
        mimic_iv_train_per_stay, a dataframe containing time series data for a single icu stay
    
    returns:
        time series data with most recent NOTE_ID in new column 'NOTE_ID'
    '''
    def _populate_recent_note_id(row):
        stay_note_timing_for_subject = stay_note_timing[stay_note_timing['SUBJECT_ID'] == row['SUBJECT_ID']]
        stay_note_timing_for_subject.dropna(inplace=True) # remove 'stay' rows which don't contain notes
        prev_notes = stay_note_timing_for_subject[stay_note_timing_for_subject['DATETIME'] < row['t_start_DT']]['NOTE_ID']
        
        if prev_notes.empty:
            row['NOTE_ID'] = None
            return row
        row['NOTE_ID'] = prev_notes.iloc[-1]
        return row
        
    mimic_iv_train_per_stay = mimic_iv_train_per_stay.apply(_populate_recent_note_id, axis=1)
    return mimic_iv_train_per_stay

def _populate_time_since_note(row):
    if (pd.isnull(row['NOTE_ID'])):
        row['time_since_note'] = None
        return row
    charttime = stay_note_timing[stay_note_timing['NOTE_ID'] == row['NOTE_ID']]['DATETIME']
    time_since_note = (row['t_start_DT'] - charttime.iloc[0]).total_seconds()/3600
    row['time_since_note'] = time_since_note
    return row


import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
args = parser.parse_args()

testing = args.test
print(f"testing = {testing}")
assert(args.note_type == 'radiology' or args.note_type == 'discharge')


trainpath = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train.csv' #mimic_iv_train.csv'
testpath = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test.csv' #mimic_iv_test.csv'

out_trainpath = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type}.csv'# CHANGEME (rad for radiology, dis for discharge)
out_testpath =  f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type}.csv'

mimic_iv_train = pd.read_csv(trainpath)
mimic_iv_test = pd.read_csv(testpath)
print(f"read from {trainpath}")
print(f"read from {testpath}")

mimic_iv_train.rename(columns={'Icustay':'ICUSTAY_ID', 'subject':'SUBJECT_ID'}, inplace=True)
mimic_iv_test.rename(columns={'Icustay':'ICUSTAY_ID', 'subject':'SUBJECT_ID'}, inplace=True)

print(f"{trainpath} has {len(mimic_iv_train['ICUSTAY_ID'].unique())} unique Icustays")
print(f"{testpath} has {len(mimic_iv_test['ICUSTAY_ID'].unique())} unique Icustays")

if testing:
    mimic_iv_train = mimic_iv_train.iloc[:2000]
    mimic_iv_test = mimic_iv_test.iloc[:2000]

all_stays = pd.read_csv('/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/tmp/all_stays.csv')

discharge = pd.read_csv('/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv')
discharge.rename(columns={'subject_id':'SUBJECT_ID'}, inplace = True)
discharge.rename(columns={'note_id':'NOTE_ID'}, inplace = True)

radiology = pd.read_csv('/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv')
radiology.rename(columns={'subject_id':'SUBJECT_ID'}, inplace = True)
radiology.rename(columns={'note_id':'NOTE_ID'}, inplace = True)

notes_to_use = radiology if args.note_type == 'radiology' else discharge




from tqdm.auto import tqdm
tqdm.pandas()
from time import time

# add subject_id to mimic_iv_train
# mimic_iv_train = mimic_iv_train_raw.merge(all_stays[['ICUSTAY_ID', 'SUBJECT_ID']], how='left', on='ICUSTAY_ID')
df_NOTES = []

tstart = time()

for df in [mimic_iv_train, mimic_iv_test]:
    relevant_subjects = set(df["SUBJECT_ID"])
    print(f"time {time() - tstart}: generating stay_note_timing_table")
    stay_note_timing = generate_stay_and_note_timing_table(relevant_subjects, all_stays, notes_to_use)

    # df_tmp is populated with useful columns for our note merge.
    df_tmp = df.copy()
    df_tmp.rename(columns={'Icustay': 'ICUSTAY_ID'}, inplace = True) 
    print(f"time {time() - tstart}: merging INTIME on ICUSTAY_ID")
    df_tmp = df_tmp.merge(all_stays[['ICUSTAY_ID', 'INTIME']], how = 'left', on ='ICUSTAY_ID') # merge with all_stays['intime'] on Icustay

    df_tmp['INTIME'] = pd.to_datetime(df_tmp['INTIME'])
    df_tmp['t_start_DT'] = df_tmp['INTIME'] + pd.to_timedelta(df_tmp['t_start'], unit='h')

    print(f"time {time() - tstart}: inserting note data by stay")
    df_NOTE = df_tmp.groupby('ICUSTAY_ID', group_keys=True).progress_apply(_insert_note_data_by_stay)

    print(f"time {time() - tstart}: populating time since note")
    df_NOTE = df_NOTE.progress_apply(_populate_time_since_note, axis=1)

    print(f"time {time() - tstart}: merging text on NOTE_ID")
    df_NOTE = df_NOTE.merge(notes_to_use[['NOTE_ID', 'text']], how = 'left', on = 'NOTE_ID')

    df_NOTES.append(df_NOTE)

mimic_iv_train_NOTE, mimic_iv_test_NOTE = df_NOTES

mimic_iv_train_NOTE.to_csv(out_trainpath, index=False)
mimic_iv_test_NOTE.to_csv(out_testpath, index=False)

print(f"wrote to {out_trainpath}")
print(f"wrote to {out_testpath}")

print(f"{out_trainpath} has {len(mimic_iv_train_NOTE['NOTE_ID'].unique())} unique NOTE_IDs")
print(f"{out_testpath} has {len(mimic_iv_test_NOTE['NOTE_ID'].unique())} unique NOTE_IDs")