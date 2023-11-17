import os 
import pandas as pd
import ast
import torch
from typing import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def tokenization(tokenizer, batched_text, max_length, truncation = True):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=truncation, 
                        max_length = max_length)

def find_next_dir_index(directory_path):
    def safe_string_to_int(s):
        try:
            return int(s)
        except ValueError:
            return 0
    # Get a list of existing indices in the directory
    existing_indices = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

    # Find the maximum index if there are existing indices
    if existing_indices:
        max_index = max([safe_string_to_int(index) for index in existing_indices])
        next_index = max_index + 1
    else:
        next_index = 1

    # Create a new directory with the next index
    return next_index

def convert_to_list(s):
    return ast.literal_eval(s)

def load_all_notes(note_type):
    if note_type == 'radiology':
        all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv'
    if note_type == 'discharge':
        all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
    all_notes = pd.read_csv(all_notes_path)
    all_notes.rename(columns={'note_id': 'NOTE_ID'}, inplace=True)
    return all_notes

def explode_train_target(train_target):
    '''
    explode train['NOTE_ID_SEQ'] for meta-finetuning
    drop duplicate notes
    '''
    positive_notes = train_target[train_target['label'] == 1]['NOTE_ID'].tolist()
    train_sequences = pd.DataFrame(train_target[['ICUSTAY_ID', 'NOTE_ID', 'NOTE_ID_SEQ']].explode('NOTE_ID_SEQ').drop_duplicates(subset='NOTE_ID_SEQ'))
    train_sequences.rename(columns={'NOTE_ID' : 'NOTE_ID_RECENT'}, inplace=True)
    train_sequences.rename(columns={'NOTE_ID_SEQ' : 'NOTE_ID'}, inplace=True)
    train_sequences['label'] = train_sequences['NOTE_ID'].apply(lambda x: x in positive_notes)
    train_sequences['label'].replace({True: 1, False: 0}, inplace=True)
    train_sequences.reset_index(inplace=True, drop=True)
    # train_sequences.rename(columns={'index':'agg_index'}, inplace=True)
    return train_sequences

def merge_embs_to_seq(train_target, train_embs) -> List:
    '''
    given an an exploded df of embeddings, merge with train_target df by NOTE_ID_SEQ

    args:
        train_embs: df consisting of {NOTE_ID, 0, 1, ... 63} corresponding to each element in the embedding 
        train_target: df containing NOTE_ID_SEQ
    '''
    train_embs_seq_list = []
    for note_id_seq in train_target['NOTE_ID_SEQ']:
        note_id_seq_df = pd.DataFrame(note_id_seq, columns=['NOTE_ID'])
        train_embs_seq = pd.merge(note_id_seq_df,
                                train_embs,
                                how = 'left',
                                on = ['NOTE_ID'])
        train_embs_seq_list.append(train_embs_seq['emb'].values.tolist())
    train_target_embseq = train_target.copy()
    train_target_embseq['emb_seq'] = train_embs_seq_list
    return train_target_embseq 


def group_train_val(ID) -> Tuple[List[int], List[int]]:
    ''' Creates shuffled 80/20 split for train and test data using indices.
    Args:
        ID: a pandas column of indexes
    
    Returns:
        A Tuple containing lists of indexes for train and test
    '''
    ID             = ID.astype(int)
    ID_unique_srtd = np.unique(ID)
    np.random.seed(40)
    np.random.shuffle(ID_unique_srtd)    

    num_train_ids = int(.80 * len(ID_unique_srtd))
    train_ids = ID_unique_srtd[:num_train_ids]
    val_ids = ID_unique_srtd[num_train_ids:]

    train = ID[ID.isin(train_ids)]
    val = ID[ID.isin(val_ids)]

    assert(len(train) + len(val) == len(ID))
    assert(len(train_ids) + len(val_ids) == len(ID_unique_srtd))
    assert(len(train_ids) + len(val_ids) == len(ID_unique_srtd))
    return list(train.index), list(val.index)

def validate_train_target_embseq(train_target_embseq):
    train_target_embseq['emb_SEQ_len'] = train_target_embseq['emb_SEQ'].apply(len)
    train_target_embseq['NOTE_ID_SEQ_len'] = train_target_embseq['NOTE_ID_SEQ'].apply(len)
    assert((train_target_embseq['emb_SEQ_len'] == train_target_embseq['NOTE_ID_SEQ_len']).all())
    print('validation of emb_SEQ_len and NOTE_ID_SEQ_len passed!')

def compute_metrics(pred) -> Dict[str, float]:
    ''' Uses prediction label_ids and predicttions to compute precision recall, accuracy and f1. 
    '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
if __name__ == '__main__':
    train_target_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/testing/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_rad_all.csv'
    train_target = pd.read_csv(train_target_path, converters = {'NOTE_ID_SEQ': convert_to_list})
    target = 'delta_in_2_days' 
    train_target = train_target.rename(columns = {target:'label'})
    train_target_exploded = explode_train_target(train_target)

    train_target_exploded['emb_FAKE'] = train_target_exploded.index

    print(train_target.head())
    print(train_target_exploded.head())

    train_target_embseq = merge_embs_to_seq(train_target, train_embs = train_target_exploded,)
    print(train_target_embseq.head())
    validate_train_target_embseq(train_target_embseq.copy())
    

    train_embs_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs{"/testing"}/{"Clinical-T5-Base"}_{"rad"}_{"all"}_out/from_epoch1/1/train_embs.pt'
    train_embs = torch.load(train_embs_path)

    
    
