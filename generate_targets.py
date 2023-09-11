import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target', dest='target', help='define a number which assigns True if event with that number of days. OR, if -1, defines a continuous "time_until_event" OR, if a list like "1,2,3", defines the number of days until the event occurs')
# parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
args = parser.parse_args()
target = args.target
if ',' in target:
    target = target.split(',')
    target_label = None # to be determined later.
else:
    target = int(target)
    target_label = f"delta_in_{target}_days" if target != -1 else "time_until_delta"
print(target)

import pandas as pd
train = pd.read_csv('/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_rad.csv')
test = pd.read_csv('/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_rad.csv')

# outpath_train = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET{target}rad.csv'
# outpath_test = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET{target}rad.csv'
outpath_ft_train = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET{target}_FT_rad.csv'
outpath_ft_test = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET{target}_FT_rad.csv'

for i in [outpath_ft_train, outpath_ft_test]:
    print(i)

for df in [train, test]:
    df['t_start_DT'] = pd.to_datetime(df['t_start_DT']) 

from datetime import timedelta
def _generate_target(subject_df, target, target_label=None): # 2 day window
    delta_times = subject_df[subject_df['delta'] == 1]['t_start_DT']
    # subject_df['num_deltas_per_subj'] = len(delta_times)

    if delta_times.empty:
        return subject_df


    if target > 0:
        masks = []
        for delta_time in delta_times:
            mask1 = (subject_df['t_start_DT'] <= delta_time) 
            mask2 = (subject_df['t_start_DT'] >= (delta_time - timedelta(days = target)))
            mask = mask1 & mask2
            masks.append(mask)
        
        mask_accum = masks[0]
        for m in masks[1:]:
            mask_accum = mask_accum | m
        subject_df.loc[mask_accum, target_label] = 1

    elif isinstance(target,list):
        [[] for _ in target]
        target = target.sort(reverse=True) # sort descending
        for t in target:
            for delta_time in delta_times:
                mask1 = (subject_df[target_label] <= delta_time) 
                mask2 = (subject_df[target_label] >= (delta_time - timedelta(days = t)))
                mask = mask1 & mask2
                masks.append(mask)
        
        for t in target:
            target_label == f'delta_in_{t}_days'
            mask_accum = masks[0] # ex: 3 day window
            for m in masks[1:]:
                mask_accum = mask_accum | m   
            subject_df.loc[mask_accum, target_label] = t

    return subject_df

from tqdm.auto import tqdm
from functools import partial
tqdm.pandas()
if isinstance(target, list):
    target_labels = [f'delta_in_{t}_days' for t in target]
    for l in target_labels:
        train[l] = 0
else:
    train[target_label] = 0
    train = train.groupby('SUBJECT_ID').progress_apply(partial(_generate_target, target=target, target_label=target_label))
    test[target_label] = 0
    test = test.groupby('SUBJECT_ID').progress_apply(partial(_generate_target, target=target, target_label=target_label))

train_note_target = train[['ICUSTAY_ID','text',target_label,'NOTE_ID']].copy()
train_note_target.dropna(inplace=True)
train_note_target = train_note_target.drop_duplicates(subset='text')

test_note_target = test[['ICUSTAY_ID','text',target_label,'NOTE_ID']].copy()
test_note_target.dropna(inplace=True)
test_note_target = test_note_target.drop_duplicates(subset='text')

train_note_target.to_csv(outpath_ft_train, index=False)
test_note_target.to_csv(outpath_ft_test, index=False)