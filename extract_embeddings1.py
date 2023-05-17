import os
import torch
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified')
parser.add_argument('--ft-model-path', dest = 'ft_model_path', help='')
args = parser.parse_args()

testing = args.test
GPU_NO = int(args.GPU_NO)
finetuned_model_path = args.ft_model_path

print(f'Test mode: {testing}')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NO)  # use the correct gpu
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


# import torch.nn as nn
# import sys

root = '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0'
# finetuned_model_path = root + '/model_from_ckpt1/meta_ft_classify.pt' # modify this line!
temivef_train_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT_rad.csv'
temivef_test_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET1_FT_rad.csv'
temivef_train_NOTE_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_rad.csv'
temivef_test_NOTE_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_rad.csv'
outdir = f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/final_rad1_{"test" if testing else ""}/' # modify this line!
train_outpath = os.path.join(outdir, 'till_end_mimic_iv_extra_features_train.csv')
test_outpath = os.path.join(outdir, 'till_end_mimic_iv_extra_features_test.csv') 
from transformers import AutoTokenizer
model_name = "Clinical-T5-Base"
tokenizer = AutoTokenizer.from_pretrained("Clinical-T5-Base")

if not os.path.exists(outdir):
    os.makedirs(outdir)

tensor_dir = f"{os.environ.get('CLINICAL_DIR')}tokenized_notes_rad"
train_tokenizer_outputs_path = os.path.join(tensor_dir, "train_tensor.pt")
test_tokenizer_outputs_path = os.path.join(tensor_dir, "test_tensor.pt")

# if not os.path.exists(tensor_dir):
    # notes_to_extract = temivef_train_NOTE_TARGET1_FT_path['text']
    # texts = notes_to_extract.tolist()
    # tokenized_notes = tokenizer(texts, truncation=True, padding=True, return_tensors = "pt")
    # torch.save(tokenized_notes, train_tokenizer_outputs_path)

    # notes_to_extract = temivef_test_NOTE_TARGET1_FT_path['text']
    # texts = notes_to_extract.tolist()
    # tokenized_notes = tokenizer(texts, truncation=True, padding=True, return_tensors = "pt")
    # torch.save(tokenized_notes, test_tokenizer_outputs_path)

tokenized_train_notes = torch.load(train_tokenizer_outputs_path)
tokenized_test_notes = torch.load(test_tokenizer_outputs_path)

# if testing:
#     tokenized_train_notes = tokenized_train_notes[:12]
#     tokenized_test_notes = tokenized_test_notes[:12]
#     print(f"testing mode truncated tokenized_notes to length {len(tokenized_train_notes)}")


from torch.utils.data import DataLoader, TensorDataset
from time import time
import numpy as np
import torch

def generate_dataloader(tokenized_notes, batch_size, device):
    inputs = tokenized_notes.input_ids
    labels = torch.tensor([-1] * len(inputs)) # not actually used in this case, since we are not evaluating loss
    dataset = TensorDataset(inputs.to(device), labels.to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False) 
    # shuffle is false so that notes retain their order for concat with df
    return dataloader

batch_size = 6
train_dataloader = generate_dataloader(tokenized_train_notes, batch_size, device)
test_dataloader = generate_dataloader(tokenized_test_notes, batch_size, device)
print("dataloaders generated")

classifier = torch.load(finetuned_model_path)
classifier.encoder.eval() # makes sure dropout does not occur
classifier.classifier.eval()
print(f"loaded classifier from {finetuned_model_path}")

def extract_embeddings(dataloader, classifier):
    with(torch.no_grad()):
        start_time = time()
        embeddings = []
        for step, batch in enumerate(dataloader):
                inputs, _ = batch
                emb = classifier.forward(inputs, return_embeddings=True)
                embeddings.append(emb)
                print(f"Step {step}/{len(dataloader)} | Time {time() - start_time : .2f} seconds")

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu()
    embeddings_df = pd.DataFrame(embeddings.detach().numpy()).add_prefix('emb')
    print(f"Extracted {len(embeddings_df)} note embeddings. Shape: {embeddings_df.shape}") # should be size 64
    return embeddings_df

train_embeddings_df = extract_embeddings(train_dataloader, classifier)
print("finished train embedding extraction") 
test_embeddings_df = extract_embeddings(test_dataloader, classifier)
print("finished test embedding extraction")


for mode in ["train", "test"]:
    print(f"reading from temivef_{mode}_NOTE_TARGET1_FT_path")
train = pd.read_csv(temivef_train_NOTE_TARGET1_FT_path)
test = pd.read_csv(temivef_test_NOTE_TARGET1_FT_path)
print(f"loaded train notes to extract from {temivef_train_NOTE_TARGET1_FT_path}")
print(f"loaded test notes to extract from {temivef_test_NOTE_TARGET1_FT_path}")

#concat notes with ICUSTAY column for merging later
train_df_small = pd.concat([train[['ICUSTAY_ID', 'NOTE_ID']], train_embeddings_df], axis = 1)
test_df_small = pd.concat([test[['ICUSTAY_ID', 'NOTE_ID']], test_embeddings_df], axis = 1)

print(f"concatenating train and train_embeddings_df with shape {train.shape} and {train_embeddings_df.shape} respectively")
print(f"concatenating test and test_embeddings_df with shape {test.shape} and {test_embeddings_df.shape} respectively")


for mode in ["train", "test"]:
    print(f"reading from temivef_{mode}_NOTE_path")
train_df_big = pd.read_csv(temivef_train_NOTE_path)
test_df_big = pd.read_csv(temivef_test_NOTE_path)
print(f"loaded train df to merge with from {temivef_train_NOTE_path}")
print(f"loaded test notes to merge with from {temivef_test_NOTE_path}")


def merge_and_fill_embeddings(df_small, df_big):
    print(f"BEFORE merge: len = {len(df_big)}")
    out_df = df_big.merge(df_small, on = ['ICUSTAY_ID','NOTE_ID'], how = 'left')
    print(f"AFTER merge: len = {len(out_df)}")
    def fill_embedding_na(note_id_group):
        note_id_group = note_id_group.fillna(method='ffill').fillna(method='bfill')
        return note_id_group

    emb_cols = ['emb' + str(i) for i in range(64)]

    print(f"BEFORE fill na: len = {len(out_df)}")
    out_df[emb_cols] = out_df.groupby('NOTE_ID')[emb_cols].transform(fill_embedding_na) # transform preserves the shape of the original
    print(f"AFTER fill na: len = {len(out_df)}")


    print(f"copied {len(df_small)} embeddings into rows with the correct NOTE_ID")
    print(f"len(out_df): {len(out_df)}")
    print(f"No. nonnull embedding rows in out_df: {len(out_df[pd.notna(out_df['emb0'])])}")
    return out_df

train_out_df = merge_and_fill_embeddings(train_df_small, train_df_big)
print("merged and filled embeddings for train_out_df")
test_out_df = merge_and_fill_embeddings(test_df_small, test_df_big)
print("merged and filled embeddings for test_out_df")


# format like mimic_iv_train
def format_cols(df):
    df.drop(['text', 'NOTE_ID', 't_start_DT','INTIME'], axis=1, inplace=True)
    df.rename(columns = {
        'SUBJECT_ID':'subject',
        'ICUSTAY_ID':'Icustay'
        }, inplace = True)
    return df

train_out_df = format_cols(train_out_df)
test_out_df = format_cols(test_out_df)

train_out_df.to_csv(train_outpath, index = False)
print("wrote to", train_outpath)
test_out_df.to_csv(test_outpath, index = False)
print("wrote to", test_outpath)