import argparse
import pandas as pd
from datetime import timedelta
from tqdm.auto import tqdm
from functools import partial
import os
from BoXHED_Fuse.src.helpers import convert_to_list


def convert_to_category(df, targets):
    """
    Convert binary columns into a single categorical column with string representations.

    Parameters:
    - df: DataFrame, input DataFrame
    - targets: list[int], integer targets which represent columns 'delta_in_X_hours', in order from greatest to least
    Returns:
    - DataFrame with a new categorical column 'delta_in_X_hours'
    """
    targets = sorted(targets, reverse=True)
    df["delta_in_X_hours"] = None

    columns = [f"delta_in_{t}_hours" for t in targets]

    df.fillna(0, inplace=True)
    for i, col in enumerate(columns):
        df.loc[df[col] == 1, "delta_in_X_hours"] = len(columns) - i

    # Drop the original binary columns if needed
    df.drop(columns=columns, inplace=True)
    return df


def _generate_target(stay_df, target):  # 2 day window
    if target == -1:
        delta_times = stay_df[stay_df["delta"] == 1]["t_start"]
        end_time = float(stay_df.iloc[-1]["t_start"])
        if delta_times.empty:
            stay_df[target_label] = end_time - stay_df["t_start"]
        else:
            delta_times = delta_times.iloc[0]
            stay_df.loc[stay_df["t_start"] <= delta_times, target_label] = (
                delta_times - stay_df["t_start"]
            )
            # make sure all data after delta_time uses end_time instead
            stay_df.loc[stay_df["t_start"] > delta_times, target_label] = (
                end_time - stay_df["t_start"]
            )
        return stay_df

    else:
        delta_times = stay_df[stay_df["delta"] == 1]["t_start_DT"]

        if delta_times.empty:
            return stay_df

        if isinstance(target, list):
            target.sort(reverse=True)  # sort descending
            for t in target:
                # for each event, populate each target by referring to time-to-event
                masks = []
                for delta_time in delta_times:
                    mask1 = stay_df["t_start_DT"] <= delta_time
                    mask2 = stay_df["t_start_DT"] >= (delta_time - timedelta(hours=t))
                    mask = mask1 & mask2
                    masks.append(mask)

                    # accumulate masks for this target
                    target_label = f"delta_in_{t}_hours"
                    mask_accum = masks[0]
                    for m in masks[1:]:
                        mask_accum = mask_accum | m
                    stay_df.loc[mask_accum, target_label] = 1

        else:
            target_label = f"delta_in_{args.target}_days"
            masks = []
            for delta_time in delta_times:
                mask1 = stay_df["t_start_DT"] <= delta_time
                mask2 = stay_df["t_start_DT"] >= (delta_time - timedelta(days=target))
                mask = mask1 & mask2
                masks.append(mask)

            mask_accum = masks[0]
            for m in masks[1:]:
                mask_accum = mask_accum | m
            stay_df.loc[mask_accum, target_label] = 1

        return stay_df


parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="if test mode enabled, reads and stores to testing folders",
)
parser.add_argument(
    "--target",
    dest="target",
    required=True,
    help='define a number which assigns True if event within that number of days. OR, if -1, defines a continuous "time_until_event" OR, if a list like "1,3,10,30", defines the number of hours until the event occurs',
)
parser.add_argument(
    "--note-type", dest="note_type", required=True, help="kw: radiology or discharge?"
)
parser.add_argument(
    "--noteid-mode", dest="noteid_mode", required=True, help="kw: all or recent"
)
args = parser.parse_args()

assert args.note_type in ["radiology", "discharge"]
assert args.noteid_mode == "all" or args.noteid_mode == "recent"

TRAIN_PATH = f"{os.getenv('BHF_ROOT')}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv"
TEST_PATH = f"{os.getenv('BHF_ROOT')}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv"
OUTPATH_FT_TRAIN = f"{os.getenv('BHF_ROOT')}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}.csv"
OUTPATH_FT_TEST = f"{os.getenv('BHF_ROOT')}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}.csv"

if "," in args.target:
    args.target = args.target.split(",")
    args.target = [int(t) for t in args.target]
    target_label = [f"delta_in_{t}_hours" for t in args.target]
else:
    args.target = int(args.target)
    target_label = (
        f"delta_in_{args.target}_hours" if args.target != -1 else "time_until_event"
    )
print("target:", args.target)

if args.test:
    TRAIN_PATH = os.path.join(
        os.path.dirname(TRAIN_PATH), "testing", os.path.basename(TRAIN_PATH)
    )
    TEST_PATH = os.path.join(
        os.path.dirname(TEST_PATH), "testing", os.path.basename(TEST_PATH)
    )
    OUTPATH_FT_TRAIN = os.path.join(
        os.path.dirname(OUTPATH_FT_TRAIN), "testing", os.path.basename(OUTPATH_FT_TRAIN)
    )
    OUTPATH_FT_TEST = os.path.join(
        os.path.dirname(OUTPATH_FT_TEST), "testing", os.path.basename(OUTPATH_FT_TEST)
    )

os.makedirs(os.path.dirname(OUTPATH_FT_TRAIN), exist_ok=True)
os.makedirs(os.path.dirname(OUTPATH_FT_TEST), exist_ok=True)

if args.noteid_mode == "all":
    train = pd.read_csv(TRAIN_PATH, converters={"NOTE_ID_SEQ": convert_to_list})
    test = pd.read_csv(TEST_PATH, converters={"NOTE_ID_SEQ": convert_to_list})
elif args.noteid_mode == "recent":
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

for df in [train, test]:
    df["t_start_DT"] = pd.to_datetime(df["t_start_DT"])

tqdm.pandas()
if isinstance(args.target, list):
    for t in target_label:
        train[t] = 0
        test[t] = 0
    train = (
        train[
            ["ICUSTAY_ID", "NOTE_ID", "delta", "t_start", "t_start_DT"] + target_label
        ]
        .groupby("ICUSTAY_ID")
        .progress_apply(
            partial(
                _generate_target,
                target=args.target,
            )
        )
    )
    test = (
        test[["ICUSTAY_ID", "NOTE_ID", "delta", "t_start", "t_start_DT"] + target_label]
        .groupby("ICUSTAY_ID")
        .progress_apply(
            partial(
                _generate_target,
                target=args.target,
            )
        )
    )
else:
    train[target_label] = 0
    test[target_label] = 0
    train = train.groupby("ICUSTAY_ID").progress_apply(
        partial(
            _generate_target,
            target=args.target,
        )
    )
    test = test.groupby("ICUSTAY_ID").progress_apply(
        partial(
            _generate_target,
            target=args.target,
        )
    )


if isinstance(target_label, list):
    columns = ["ICUSTAY_ID", "NOTE_ID", "delta", "t_start_DT"] + target_label
else:
    columns = ["ICUSTAY_ID", "NOTE_ID", "delta", "t_start_DT", target_label]
train_note_target = train[columns].copy()
test_note_target = test[columns].copy()
if args.noteid_mode == "all":
    train_note_target["NOTE_ID_SEQ"] = train["NOTE_ID_SEQ"].copy()
    test_note_target["NOTE_ID_SEQ"] = test["NOTE_ID_SEQ"].copy()


train_note_target = train_note_target.drop_duplicates(subset="NOTE_ID").dropna()
test_note_target = test_note_target.drop_duplicates(subset="NOTE_ID").dropna()

if isinstance(args.target, list):
    train_note_target = convert_to_category(train_note_target, args.target)
    test_note_target = convert_to_category(test_note_target, args.target)

train_note_target.to_csv(OUTPATH_FT_TRAIN, index=False)
test_note_target.to_csv(OUTPATH_FT_TEST, index=False)
