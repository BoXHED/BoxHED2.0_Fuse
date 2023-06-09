import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', dest='target', help='define a number which assigns True if event with that number of days. OR, if -1, defines a continuous "time_until_event"')
# parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
args = parser.parse_args()
target = args.target

import pandas as pd
train = pd.read_csv('/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_rad.csv')
test = pd.read_csv('/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_rad.csv')

outpath_train = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_rad.csv'
outpath_test = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET1_rad.csv'
outpath_ft_train = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT_rad.csv'
outpath_ft_test = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET1_FT_rad.csv'

for i in [outpath_train, outpath_test, outpath_ft_train, outpath_ft_test]:
    print(i)

for df in [train, test]:
    df['t_start_DT'] = pd.to_datetime(df['t_start_DT']) 

from datetime import timedelta
def _generate_target(subject_df, window): # 2 day window
    target_label = f"delta_in_{target}_days" if target != -1 else "time_until_delta"
    delta_times = subject_df[subject_df['delta'] == 1]['t_start_DT']
    # subject_df['num_deltas_per_subj'] = len(delta_times)

    if (delta_times.empty):
        return subject_df

    masks = []
    for delta_time in delta_times:
        mask1 = (subject_df[target_label] <= delta_time) 
        mask2 = (subject_df[target_label] >= (delta_time - timedelta(days = window)))
        mask = mask1 & mask2
        masks.append(mask)
    
    mask_acc = masks[0]
    for m in masks[1:]:
        mask_acc = mask_acc | m
    
    subject_df.loc[mask_acc, target_label] = 1
    return subject_df

from tqdm.auto import tqdm
from functools import partial
tqdm.pandas()

train['delta_in_2_days'] = 0
train = train.groupby('SUBJECT_ID').progress_apply(partial(_generate_target, target))

test['delta_in_2_days'] = 0
test = test.groupby('SUBJECT_ID').progress_apply(partial(_generate_target, target))

train_note_target = train[['ICUSTAY_ID','text','delta_in_2_days','NOTE_ID']].copy()
train_note_target.dropna(inplace=True)
train_note_target = train_note_target.drop_duplicates(subset='text')

test_note_target = test[['ICUSTAY_ID','text','delta_in_2_days','NOTE_ID']].copy()
test_note_target.dropna(inplace=True)
test_note_target = test_note_target.drop_duplicates(subset='text')

train_note_target.to_csv(outpath_ft_train, index=False)
test_note_target.to_csv(outpath_ft_test, index=False)