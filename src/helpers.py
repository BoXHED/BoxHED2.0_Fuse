import os 
import pandas as pd
from tqdm import tqdm
import ast
import torch
from typing import *
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score


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

def convert_to_nested_list(s):
    return ast.literal_eval(s.replace('array', ''))




def load_all_notes(note_type):
    if note_type == 'radiology':
        all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv'
    if note_type == 'discharge':
        all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
    all_notes = pd.read_csv(all_notes_path)
    all_notes.rename(columns={'note_id': 'NOTE_ID'}, inplace=True)
    return all_notes

def explode_train_target(train_target, target_label):
    '''
    explode train['NOTE_ID_SEQ'] for meta-finetuning
    drop duplicate notes

    args:
        train_target: train dataframe with targets
        target: str, time_until_event or delta_in_2_days

        if a note has delta within 2 days, we put a label of 1
        or, for regression, if a note has an event in n days, we put time_until_
    '''
    if target_label == 'delta_in_2_days':
        positive_notes = train_target[train_target['label'] == 1]['NOTE_ID'].tolist()
        train_sequences = pd.DataFrame(train_target[['ICUSTAY_ID', 'NOTE_ID', 'NOTE_ID_SEQ']].explode('NOTE_ID_SEQ').drop_duplicates(subset='NOTE_ID_SEQ'))
        train_sequences.rename(columns={'NOTE_ID' : 'NOTE_ID_RECENT'}, inplace=True)
        train_sequences.rename(columns={'NOTE_ID_SEQ' : 'NOTE_ID'}, inplace=True)
        train_sequences['label'] = train_sequences['NOTE_ID'].apply(lambda x: x in positive_notes)
        train_sequences['label'].replace({True: 1, False: 0}, inplace=True)
        train_sequences.reset_index(inplace=True, drop=True)
        # train_sequences.rename(columns={'index':'agg_index'}, inplace=True)
    elif target_label == 'time_until_event':
        train_target.rename(columns={'time_until_event':'label'}, inplace=True)
        train_sequences = pd.DataFrame(train_target[['ICUSTAY_ID', 'NOTE_ID', 'NOTE_ID_SEQ', 'label']].explode('NOTE_ID_SEQ').drop_duplicates(subset='NOTE_ID_SEQ'))
        train_sequences.rename(columns={'NOTE_ID' : 'NOTE_ID_RECENT'}, inplace=True)
        train_sequences.rename(columns={'NOTE_ID_SEQ' : 'NOTE_ID'}, inplace=True)
        breakpoint()
    return train_sequences

def merge_text(data, note_type):
    all_notes = load_all_notes(note_type)
    data = pd.merge(data, all_notes[['NOTE_ID','text']], on='NOTE_ID', how='left')
    return data

def merge_embs_to_seq(train_target, train_embs) -> List:
    '''
    given an an exploded df of embeddings, merge with train_target df by NOTE_ID_SEQ

    args:
        train_embs: df consisting of {NOTE_ID, 0, 1, ... 63} corresponding to each element in the embedding 
        train_target: df containing NOTE_ID and NOTE_ID_SEQ
    '''
    train_embs_seq_list = []
    for note_id_seq in tqdm(train_target['NOTE_ID_SEQ']):
        note_id_seq_df = pd.DataFrame(note_id_seq, columns=['NOTE_ID'])
        train_embs_seq = pd.merge(note_id_seq_df,
                                train_embs,
                                how = 'left',
                                on = ['NOTE_ID'])
        train_embs_seq_list.append(train_embs_seq['emb'].values.tolist())
    train_target_embseq = train_target.copy()
    train_target_embseq['emb_seq'] = train_embs_seq_list
    validate_train_emb_seq(train_target_embseq, train_target)
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

# def validate_train_target_embseq(train_target_embseq):
#     train_target_embseq['emb_SEQ_len'] = train_target_embseq['emb_SEQ'].apply(len)
#     train_target_embseq['NOTE_ID_SEQ_len'] = train_target_embseq['NOTE_ID_SEQ'].apply(len)
#     assert((train_target_embseq['emb_SEQ_len'] == train_target_embseq['NOTE_ID_SEQ_len']).all())
#     print('validation of emb_SEQ_len and NOTE_ID_SEQ_len passed!')

def validate_train_emb_seq(train_embseq, train_target):
    train_embseq = train_embseq.copy()
    train_embseq['emb_seq_len'] = train_embseq['emb_seq'].apply(len)
    train_target['NOTE_ID_SEQ_len'] = train_target['NOTE_ID_SEQ'].apply(len)
    assert((train_target['NOTE_ID_SEQ_len'] == train_embseq['emb_seq_len']).all())



def compute_metrics(pred, num_labels) -> Dict[str, float]:
    ''' Uses prediction label_ids and predicttions to compute precision recall, accuracy and f1. 
    '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    if num_labels == 2:
        average = "binary"
    elif num_labels > 2:
        average = "weighted"
    breakpoint()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)

    if num_labels <= 2:
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError as e:
            print(f"Error: {e}")
            auc = None
        auprc = average_precision_score(labels, preds)
    
        return {
            'auc' : auc,
            'auprc' : auprc,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    else:
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

def compute_metrics_LSTM(labels, preds) -> Dict[str, float]:
    ''' 
    Uses predictions, labels to compute precision recall, accuracy and f1. 
    '''
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    # precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    # acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError as e:
        print(f"Error: {e}")
        auc = None

    auprc = average_precision_score(labels, preds)
    return {
        'auc' : auc,
        'auprc' : auprc,
    }

from torch.utils import data
# maximum sequence length
# doc_emb_size = 64 # 768
    
class Sequential_Dataset(data.Dataset):

    def __init__(self, ds, doc_emb_size, max_num_notes):
        'Initialization'
        self.doc_emb_size = doc_emb_size
        self.ds = ds
        self.max_num_notes = max_num_notes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ds)

    def __getitem__(self, index):

        'Generates one sample of data'
        # Select sample
        # Load data and get label 
        print('start getitem')
        t0 = time.time()       
        y = self.ds['label'][index]
        emb_seq = self.ds['emb_seq'][index]
        emb_seq_out = torch.zeros(size=(self.max_num_notes, self.doc_emb_size), dtype=torch.float)  

        print(f'emb_seq_out (zeroos) took {time.time() - t0} seconds')
        t0 = time.time()
        if len(emb_seq) > self.max_num_notes:
            emb_seq_out[:self.max_num_notes] = emb_seq[:self.max_num_notes]
        else:
            emb_seq_out[:len(emb_seq)] = emb_seq
        
        print(f'populated emb_seq_out for getitem, took {time.time() - t0} seconds')

        return emb_seq_out.cuda(), y
    
class Sequential_Dataset_FAST(data.Dataset):

    def __init__(self, train_target, train_target_exploded, train_emb, max_num_notes):
        'Initialization'
        self.train_target = train_target
        self.train_target_exploded = train_target_exploded
        self.train_embs = train_emb
        self.doc_emb_size = train_emb.shape[1] # 64
        self.max_num_notes = max_num_notes

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_target)

    def __getitem__(self, index):

        'Generates one sample of data'
        # t0 = time.time()
        y = self.train_target['label'].iloc[index]
        note_id_seq = self.train_target.NOTE_ID_SEQ.iloc[index]
        exploded_idx = pd.merge(pd.DataFrame({'NOTE_ID': note_id_seq}, columns=['NOTE_ID']),
                self.train_target_exploded,
                how='left',
                on=['NOTE_ID'])
        emb_seq = self.train_embs[exploded_idx.index]
        # print(f'getting emb_seq took {time.time() - t0} seconds')

        emb_seq_out = torch.zeros(size=(self.max_num_notes, self.doc_emb_size), dtype=torch.float)  
        # print(f'emb_seq_out (zeros) took {time.time() - t0} seconds')
        # t0 = time.time()
        if len(emb_seq) > self.max_num_notes:
            emb_seq_out[:self.max_num_notes] = emb_seq[:self.max_num_notes]
        else:
            emb_seq_out[:len(emb_seq)] = emb_seq
        
        # print(f'populated emb_seq_out for getitem, took {time.time() - t0} seconds')
        return emb_seq_out.cuda(), y

class PredictionObject:
    def __init__(self, label_ids, predictions):
        self.label_ids = label_ids
        self.predictions = predictions

if __name__ == '__main__':
    # Example with correct and incorrect predictions
    label_ids_example = torch.tensor([1, 3, 2, 0, 4, 5])  # Ground truth labels
    predictions_example = torch.tensor([[0.8, 1, 0.05, 1, 0.02, 0.01],
                                        [0.01, 0.1, 0.2, 0.5, 0.05, 0.14],
                                        [0.01, 0.05, 0.7, 0.02, 0.1, 0.12],
                                        [0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
                                        [0.9, 0.05, 0.02, 0.01, 0.01, 0.01],
                                        [0.9, 0.05, 0.02, 0.01, 0.01, 0.01]])
    pred = PredictionObject(label_ids = label_ids_example, predictions = predictions_example)
    print(compute_metrics(pred = pred, num_labels = 6))
    # train_target_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/testing/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_rad_all.csv'
    # train_target = pd.read_csv(train_target_path, converters = {'NOTE_ID_SEQ': convert_to_list})
    # target = 'delta_in_2_days' 
    # train_target = train_target.rename(columns = {target:'label'})
    # train_target_exploded = explode_train_target(train_target)

    # train_target_exploded['emb_FAKE'] = train_target_exploded.index

    # print(train_target.head())
    # print(train_target_exploded.head())

    # train_target_embseq = merge_embs_to_seq(train_target, train_embs = train_target_exploded,)
    # print(train_target_embseq.head())
    # validate_train_target_embseq(train_target_embseq.copy())
    

    # train_embs_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs{"/testing"}/{"Clinical-T5-Base"}_{"rad"}_{"all"}_out/from_epoch1/1/train_embs.pt'
    # train_embs = torch.load(train_embs_path)

    
    
