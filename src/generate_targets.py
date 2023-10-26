import argparse
import pandas as pd
from datetime import timedelta
from tqdm.auto import tqdm
from functools import partial
import os
from helpers import convert_to_list

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

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='if test mode enabled, reads and stores to testing folders')
parser.add_argument('--target', dest='target', help='define a number which assigns True if event within that number of days. OR, if -1, defines a continuous "time_until_event" OR, if a list like "1,2,3", defines the number of days until the event occurs')
parser.add_argument('--note-type', dest = 'note_type', help='kw: radiology or discharge?')
parser.add_argument('--noteid-mode', dest = 'noteid_mode', help = 'kw: all or recent')
args = parser.parse_args()

assert(args.note_type in  ['radiology', 'discharge'])
assert(args.noteid_mode == 'all' or args.noteid_mode == 'recent')

target = args.target
if ',' in target:
    target = target.split(',')
    target_label = None # to be determined later.
else:
    target = int(target)
    target_label = f"delta_in_{target}_days" if target != -1 else "time_until_delta"
print('target:', target)

train_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
test_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
outpath_ft_train = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_{target}_{args.note_type[:3]}_{args.noteid_mode}.csv'
outpath_ft_test = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_{target}_{args.note_type[:3]}_{args.noteid_mode}.csv'

if args.test:
    train_path = os.path.join(os.path.dirname(train_path), 'testing', os.path.basename(train_path))
    test_path = os.path.join(os.path.dirname(test_path), 'testing', os.path.basename(test_path))
    outpath_ft_train = os.path.join(os.path.dirname(outpath_ft_train), 'testing', os.path.basename(outpath_ft_train))
    outpath_ft_test = os.path.join(os.path.dirname(outpath_ft_test), 'testing', os.path.basename(outpath_ft_test))

if args.noteid_mode == 'all':
    train = pd.read_csv(train_path, converters = {'NOTE_ID': convert_to_list})
    test = pd.read_csv(test_path, converters = {'NOTE_ID': convert_to_list})
elif args.noteid_mode == "recent":
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

assert(not os.path.exists(outpath_ft_train))
assert(not os.path.exists(outpath_ft_test))

for df in [train, test]:
    df['t_start_DT'] = pd.to_datetime(df['t_start_DT']) 

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

train_note_target = train[['ICUSTAY_ID','NOTE_ID',target_label]].copy()
train_note_target.dropna(inplace=True)
train_note_target = train_note_target.drop_duplicates(subset='NOTE_ID')

test_note_target = test[['ICUSTAY_ID','NOTE_ID',target_label]].copy()
test_note_target.dropna(inplace=True)
test_note_target = test_note_target.drop_duplicates(subset='NOTE_ID') # FIXME find all note ids from train and test


if args.noteid_mode == 'all':
    # breakpoint()
    train_note_target = train_note_target[train_note_target.NOTE_ID.apply(len) > 0]
    test_note_target = test_note_target[test_note_target.NOTE_ID.apply(len) > 0]


train_note_target.to_csv(outpath_ft_train, index=False)
test_note_target.to_csv(outpath_ft_test, index=False)