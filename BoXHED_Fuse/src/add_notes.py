import pandas as pd
import argparse
from time import time
import os
import wandb
from BoXHED_Fuse.experiments.log_artifact import log_artifact
from functools import partial
from tqdm.auto import tqdm

tqdm.pandas()


def generate_stay_and_note_timing_table(relevant_subjects, stays, notes):

    # filter for useful notes and stays data
    def _filter_notes_and_stays(relevant_subjects, stays, notes):
        """
        Get all NOTE_ID, SUBJECT_ID, and charttime from overall note data.
        Get all SUBJECT_ID, ICUSTAY_ID, and INTIME from overall stay data.
        Returns both dataframes.

        Args:
            relevant_subjects: all subjects in cohort (train or test),
                drawn from time series data
            stays: icustay data -- from MIMIC IV
            notes: note data -- from MIMIC IV Note (Radiology or Discharge)
        """
        relevant_notes = notes[notes["SUBJECT_ID"].isin(relevant_subjects)]
        relevant_stays = stays[stays["SUBJECT_ID"].isin(relevant_subjects)]

        relevant_notes = relevant_notes[["SUBJECT_ID", "NOTE_ID", "charttime"]]
        relevant_stays = relevant_stays[["SUBJECT_ID", "ICUSTAY_ID", "INTIME"]]
        return (relevant_notes, relevant_stays)

    # concatenate notes and stays, grouping by subject and sorting by DATETIME
    def _concat_and_sort_table(relevant_notes, relevant_stays):
        """
        Time series alignment of notes and stays.

        args:
            relevant notes: df with cols ['SUBJECT_ID', 'NOTE_ID', 'charttime']
            relevant_stays: df with cols ['SUBJECT_ID', 'ICUSTAY_ID', 'INTIME']

        returns:
            stay_note_timing: df with cols ['SUBJECT_ID', 'DATETIME', 'NOTE_ID', 'ICUSTAY_ID']
        """
        stay_note_timing = pd.concat([relevant_notes, relevant_stays]).sort_values(
            by="SUBJECT_ID"
        )
        stay_note_timing["DATETIME"] = stay_note_timing["INTIME"].combine_first(
            stay_note_timing["charttime"]
        )
        stay_note_timing.drop(["charttime"], axis=1, inplace=True)
        stay_note_timing = stay_note_timing.groupby(
            "SUBJECT_ID", group_keys=True
        ).apply(lambda x: x.sort_values("DATETIME"))
        stay_note_timing.reset_index(drop=True, inplace=True)
        stay_note_timing.infer_objects()

        return stay_note_timing

    def _fillna_icustay_by_subject(stay_note_timing_by_subject):
        """
        inputs:
            stay_note_timing_by_stay: a subset of stay_note_timing for a single subject

        for all rows with notes, fill forwards with the most recent ICUSTAY_ID, filling backwards if note predates ICUSTAY.
        """
        stay_note_timing_by_subject["ICUSTAY_ID"].fillna(method="ffill", inplace=True)
        stay_note_timing_by_subject["ICUSTAY_ID"].fillna(method="bfill", inplace=True)
        return stay_note_timing_by_subject

    def _fillna_intime_by_stay(stay_note_timing_by_stay):
        """
        inputs:
            stay_note_timing_by_stay: a subset of stay_note_timing for a single stay

        for all rows with notes, fill forwards with the most recent INTIME, filling backwards if note predates ICUSTAY.
        """
        stay_note_timing_by_stay["INTIME"].fillna(method="ffill", inplace=True)
        stay_note_timing_by_stay["INTIME"].fillna(method="bfill", inplace=True)
        return stay_note_timing_by_stay

    relevant_notes, relevant_stays = _filter_notes_and_stays(
        relevant_subjects=relevant_subjects, stays=stays, notes=notes
    )
    stay_note_timing = _concat_and_sort_table(relevant_notes, relevant_stays)

    stay_note_timing = stay_note_timing.groupby("SUBJECT_ID", group_keys=True).apply(
        _fillna_icustay_by_subject
    )
    stay_note_timing = stay_note_timing.groupby("ICUSTAY_ID", group_keys=True).apply(
        _fillna_intime_by_stay
    )

    stay_note_timing["DATETIME"] = pd.to_datetime(stay_note_timing["DATETIME"])
    stay_note_timing["INTIME"] = pd.to_datetime(stay_note_timing["INTIME"])

    stay_note_timing = stay_note_timing.convert_dtypes(infer_objects=True)

    stay_note_timing.dropna(
        inplace=True
    )  # remove 'stay' rows which don't contain notes
    return stay_note_timing


def _insert_note_data_by_stay(mimic_iv_train_per_stay, how="recent"):
    """
    for time series data for a single icustay, populate most recent note id or a sequence of note ids,
        returning None if there are no previous notes

    args:
        mimic_iv_train_per_stay: a dataframe containing time series data for a single icu stay
        how: from ["recent", "all"] determines how to insert note data.
            if how = "recent", gets most recent note.
            if how = "all", gets all notes and stores them as a list.

    returns:
        time series data with NOTE_ID in new column 'NOTE_ID' and possibly a new column'NOTE_ID_SEQ'
    """

    def _populate_recent_note_id(row):
        stay_note_timing_for_subject = stay_note_timing[
            stay_note_timing["SUBJECT_ID"] == row["SUBJECT_ID"]
        ]
        prev_notes = stay_note_timing_for_subject[
            stay_note_timing_for_subject["DATETIME"] < row["t_start_DT"]
        ]["NOTE_ID"]

        if prev_notes.empty:
            row["NOTE_ID"] = None
            return row
        row["NOTE_ID"] = prev_notes.iloc[-1]
        return row

    def _populate_all_note_ids(row):
        stay_note_timing_for_subject = stay_note_timing[
            stay_note_timing["SUBJECT_ID"] == row["SUBJECT_ID"]
        ]
        prev_notes = stay_note_timing_for_subject[
            stay_note_timing_for_subject["DATETIME"] < row["t_start_DT"]
        ]["NOTE_ID"]
        row["NOTE_ID_SEQ"] = prev_notes.tolist()
        return row

    if how == "recent":
        mimic_iv_train_per_stay = mimic_iv_train_per_stay.apply(
            _populate_recent_note_id, axis=1
        )
    elif how == "all":
        mimic_iv_train_per_stay = mimic_iv_train_per_stay.apply(
            _populate_recent_note_id, axis=1
        )
        mimic_iv_train_per_stay = mimic_iv_train_per_stay.apply(
            _populate_all_note_ids, axis=1
        )

    return mimic_iv_train_per_stay


def _populate_time_since_note(row):
    # if isinstance(row['NOTE_ID'], list):
    #     if row['NOTE_ID'] == []:
    #         row['time_since_note'] = None
    #     else:
    #         charttime = stay_note_timing[stay_note_timing['NOTE_ID'] == row['NOTE_ID'][-1]]['DATETIME']
    #         time_since_note = (row['t_start_DT'] - charttime.iloc[0]).total_seconds()/3600
    #         row['time_since_note'] = time_since_note
    # else:

    if pd.isna(row["NOTE_ID"]):
        row["time_since_note"] = None
    else:
        charttime = stay_note_timing[stay_note_timing["NOTE_ID"] == row["NOTE_ID"]][
            "DATETIME"
        ]
        time_since_note = (row["t_start_DT"] - charttime.iloc[0]).total_seconds() / 3600
        row["time_since_note"] = time_since_note
    return row


parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="enable testing mode")
parser.add_argument("--use-wandb", action="store_true", help="enable wandb")
parser.add_argument("--note-type", dest="note_type", help="kw: radiology or discharge")
parser.add_argument("--noteid-mode", dest="noteid_mode", help="kw: all or recent")
args = parser.parse_args()

assert args.note_type == "radiology" or args.note_type == "discharge"
assert args.noteid_mode == "all" or args.noteid_mode == "recent"

trainpath = f"{os.getenv('BHF_ROOT')}/data/till_end_mimic_iv_extra_features_train.csv"  # mimic_iv_train.csv'
testpath = f"{os.getenv('BHF_ROOT')}/data/till_end_mimic_iv_extra_features_test.csv"  # mimic_iv_test.csv'

out_trainpath = f"{os.getenv('BHF_ROOT')}/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv"
out_testpath = f"{os.getenv('BHF_ROOT')}/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv"
assert os.path.exists(os.path.dirname(out_trainpath))
assert os.path.exists(os.path.dirname(out_testpath))

mimic_iv_train = pd.read_csv(trainpath)
mimic_iv_test = pd.read_csv(testpath)
print(f"read from {trainpath}")
print(f"read from {testpath}")

mimic_iv_train.rename(
    columns={"Icustay": "ICUSTAY_ID", "subject": "SUBJECT_ID"}, inplace=True
)
mimic_iv_test.rename(
    columns={"Icustay": "ICUSTAY_ID", "subject": "SUBJECT_ID"}, inplace=True
)


mimic_iv_train = mimic_iv_train[
    ["ICUSTAY_ID", "SUBJECT_ID", "t_start", "t_end", "delta"]
]
mimic_iv_test = mimic_iv_test[
    ["ICUSTAY_ID", "SUBJECT_ID", "t_start", "t_end", "Y", "delta"]
]

if args.test:
    mimic_iv_train = mimic_iv_train.iloc[:2000]
    mimic_iv_test = mimic_iv_test.iloc[:2000]

    out_trainpath = os.path.join(
        os.path.dirname(out_trainpath), "testing", os.path.basename(out_trainpath)
    )
    out_testpath = os.path.join(
        os.path.dirname(out_testpath), "testing", os.path.basename(out_testpath)
    )
    os.makedirs(os.path.dirname(out_trainpath), exist_ok=True)
    os.makedirs(os.path.dirname(out_testpath), exist_ok=True)

all_stays = pd.read_csv(f"{os.getenv('BHF_ROOT')}/JSS_SUBMISSION/tmp/all_stays.csv")

if args.note_type == "discharge":
    discharge = pd.read_csv(f"{os.getenv('NOTE_DIR')}/discharge.csv")
    discharge.rename(columns={"subject_id": "SUBJECT_ID"}, inplace=True)
    discharge.rename(columns={"note_id": "NOTE_ID"}, inplace=True)
elif args.note_type == "radiology":
    radiology = pd.read_csv(f"{os.getenv('NOTE_DIR')}/radiology.csv")
    radiology.rename(columns={"subject_id": "SUBJECT_ID"}, inplace=True)
    radiology.rename(columns={"note_id": "NOTE_ID"}, inplace=True)
notes = radiology if args.note_type == "radiology" else discharge

df_NOTES = []

tstart = time()

for df in [mimic_iv_train, mimic_iv_test]:
    relevant_subjects = set(df["SUBJECT_ID"])
    print(f"time {time() - tstart}: generating stay_note_timing_table")
    stay_note_timing = generate_stay_and_note_timing_table(
        relevant_subjects, all_stays, notes
    )

    df_tmp = df.copy()
    df_tmp.rename(columns={"Icustay": "ICUSTAY_ID"}, inplace=True)
    print(f"time {time() - tstart}: merging INTIME on ICUSTAY_ID")
    df_tmp = df_tmp.merge(
        all_stays[["ICUSTAY_ID", "INTIME"]], how="left", on="ICUSTAY_ID"
    )  # merge with all_stays['intime'] on Icustay

    df_tmp["INTIME"] = pd.to_datetime(df_tmp["INTIME"])
    df_tmp["t_start_DT"] = df_tmp["INTIME"] + pd.to_timedelta(
        df_tmp["t_start"], unit="h"
    )

    print(f"time {time() - tstart}: inserting note data by stay")

    df_NOTE = df_tmp.groupby("ICUSTAY_ID", group_keys=True).progress_apply(
        partial(_insert_note_data_by_stay, how=args.noteid_mode)
    )

    print(f"time {time() - tstart}: populating time since note")
    df_NOTE = df_NOTE.progress_apply(_populate_time_since_note, axis=1)

    # print(f"time {time() - tstart}: merging text on NOTE_ID") # FIXME move this down the pipeline
    # df_NOTE = df_NOTE.merge(notes_to_use[['NOTE_ID', 'text']], how = 'left', on = 'NOTE_ID')

    df_NOTES.append(df_NOTE)

mimic_iv_train_NOTE, mimic_iv_test_NOTE = df_NOTES

mimic_iv_train_NOTE.to_csv(out_trainpath, index=False)
mimic_iv_test_NOTE.to_csv(out_testpath, index=False)

print(f"wrote to {out_trainpath}")
print(f"wrote to {out_testpath}")


if args.use_wandb:
    wandb.login(key=os.getenv("WANDB_KEY"), relogin=True)

    log_artifact(
        artifact_path=out_trainpath,
        artifact_name=(
            os.path.splitext(os.path.basename(out_trainpath))[0] + ".test"
            if args.test
            else ""
        ),
        artifact_description="MIMIC IV joined with note data for finetuning",
        artifact_metadata=dict(args._get_kwargs()),
        project_name=os.getenv("WANDB_PROJECT_NAME"),
        do_filter=True,
    )

    log_artifact(
        artifact_path=out_testpath,
        artifact_name=(
            os.path.splitext(os.path.basename(out_testpath))[0] + ".test"
            if args.test
            else ""
        ),
        artifact_description="MIMIC IV joined with note data for finetuning",
        artifact_metadata=dict(args._get_kwargs()),
        project_name=os.getenv("WANDB_PROJECT_NAME"),
        do_filter=True,
    )
