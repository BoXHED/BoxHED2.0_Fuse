import argparse
from typing import *
import os
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel, LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig, T5Config
from datasets import Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
import os
import argparse
from functools import partial
import traceback

from BoXHED_Fuse.src.helpers import tokenization, convert_to_list, find_next_dir_index, load_all_notes, explode_train_target, compute_metrics, group_train_val, merge_text
from sklearn.decomposition import PCA

'''
Perform a sweep over all T5 configurations and note types. 
Use PCA to reduce dim to 64.
'''

def extract_embeddings(dataloader, classifier):
    with(torch.no_grad()):
        start_time = time()
        embeddings = []
        for step, batch in enumerate(dataloader):
            input_ids, attention_mask = batch
            emb = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, return_embeddings=True)
            embeddings.append(emb)
            print(f"Step {step}/{len(dataloader)} | Time {time() - start_time : .2f} seconds")

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu()
    return embeddings

def generate_dataloader(tokenized_notes, batch_size, device):
    input_ids = torch.tensor(tokenized_notes.input_ids, dtype=torch.long)
    attention_mask = torch.tensor(tokenized_notes.attention_mask, dtype=torch.long)

    tdataset = TensorDataset(input_ids.to(device),
                            attention_mask.to(device),
                            )
    dataloader = DataLoader(tdataset, batch_size=batch_size, shuffle=False, drop_last=False) 
    # shuffle is false so that notes retain their order for concat with df
    return dataloader

def extract_embeddings(dataloader):
    '''
    Pass note through encoder, then into PCA
    '''
    
    # ENCODE NOTES


    # PERFORM PCA
    # pca = PCA(n_components=k)
    # reduced_embeddings = pca.fit_transform(note_embeddings)

def df_to_tokens_ds(data):
    '''
    Convert pandas DataFrame to Dataset containing text and lq
    abel
    '''
    data = Dataset.from_pandas(data).select_columns(['text', 'label'])
    data = data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(data) // 10) # def tokenization(tokenizer, batched_text, max_length):
    data = data.remove_columns('text')
    return data

if __name__ == '__main__':

    # ===== Initialize Args =====   
    TESTING = True
    USE_WANDB = False
    GPU_NO = -1
    NOTE_TYPE = 'radiology'
    SWEEP = False
    SWEEP_ID = -1
    MODEL_NAME = 'Clinical-T5-Base'
    MODEL_DIR = os.path.join('/home/ugrads/a/aa_ron_su/BoXHED_Fuse/BoXHED_Fuse/models', MODEL_NAME)
    TRAIN_PATH = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_rad_recent.csv'
    TEST_PATH = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_rad_recent.csv'
    DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NO)  # use the correct gpu

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TESTING:
        train = train.iloc[:2000].copy()
        test = test.iloc[:2000].copy()
    
    train = merge_text(train, NOTE_TYPE)
    test = merge_text(test, NOTE_TYPE)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    encoder = model.get_encoder() # we only need the clinical-t5 encoder for our purposes

    print('loading notes into train and test dataframes')
    train = merge_text(train, NOTE_TYPE)
    test = merge_text(test, NOTE_TYPE)
    
    tokenized_train_notes = df_to_tokens_ds(train)
    tokenized_test_notes = df_to_tokens_ds(test)
    print(f'tokenized_train_notes dataset:, {tokenized_train_notes}')
    print(f'tokenized_test_notes dataset:, {tokenized_test_notes}')

    tokenized_train_notes = pd.DataFrame(tokenized_train_notes)
    tokenized_test_notes = pd.DataFrame(tokenized_test_notes)
    print('tokenized notes converted back to dataframes for extraction...')
    
    batch_size = 24 # 48
    train_dataloader = generate_dataloader(tokenized_train_notes, batch_size, DEVICE)
    test_dataloader = generate_dataloader(tokenized_test_notes, batch_size, DEVICE)
    breakpoint()




