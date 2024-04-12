import numpy as np
import os
import torch
import argparse
import pandas as pd
import re
from torch.utils.data import DataLoader, TensorDataset
from time import time
import torch
from datasets import Dataset
from BoXHED_Fuse.src.helpers import tokenization, find_next_dir_index, explode_train_target, convert_to_list, merge_text
from functools import partial
from transformers import LongformerTokenizerFast, AutoTokenizer



def df_to_tokens_ds(data):
    '''
    Convert pandas DataFrame to Dataset containing text and lq
    abel
    '''
    data = Dataset.from_pandas(data).select_columns(['text', 'label'])
    data = data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(data) // 10) # def tokenization(tokenizer, batched_text, max_length):
    data = data.remove_columns('text')
    return data

def generate_dataloader(tokenized_notes, batch_size, device):
    input_ids = torch.tensor(tokenized_notes.input_ids, dtype=torch.long)
    attention_mask = torch.tensor(tokenized_notes.attention_mask, dtype=torch.long)

    tdataset = TensorDataset(input_ids.to(device),
                            attention_mask.to(device),
                            )
    dataloader = DataLoader(tdataset, batch_size=batch_size, shuffle=False, drop_last=False) 
    # shuffle is false so that notes retain their order for concat with df
    return dataloader

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

def embs_tensor_to_df(embeddings):
    embeddings_df = pd.DataFrame(embeddings.detach().numpy()).add_prefix('emb')
    print(f"Extracted {len(embeddings_df)} note embeddings. Shape: {embeddings_df.shape}") # should be size 64
    return embeddings_df

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

# format like mimic_iv_train
def format_cols(df):
    df.drop(['text', 'NOTE_ID', 't_start_DT','INTIME'], axis=1, inplace=True, errors = 'ignore')
    df.rename(columns = {
        'SUBJECT_ID':'subject',
        'ICUSTAY_ID':'Icustay'
        }, inplace = True)
    return df

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

if __name__ == '__main__':
    '''
    Use finetuned model to extract embeddings from train and test notes.

    If noteid_mode is "all", save embeddings to an embeddings only csv, which can be used later in sequential embeddings.
    If noteid_mode is "recent", populate train and test with embeddings.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', 
        help='enable args.test mode')
    parser.add_argument('--gpu-no', 
        dest = 'GPU_NO', 
        help='use GPU_NO specified')
    parser.add_argument('--ckpt-dir', 
        dest = 'ckpt_dir', 
        help='FULL PATH of directory where model checkpoint is stored')
    parser.add_argument('--ckpt-model-name', 
        dest = 'ckpt_model_name', 
        help='name of model checkpoint is stored')
    parser.add_argument('--note-type', 
        dest = 'note_type', 
        help='which notes, radiology or discharge?')
    parser.add_argument('--model-name', 
        dest = 'model_name')
    parser.add_argument('--model-type', 
        dest = 'model_type', help = 'T5 or Longformer')
    parser.add_argument('--noteid-mode',                
                        dest = 'noteid_mode', 
                        help = 'kw: all or recent')
    parser.add_argument('--target',                 
                        dest = 'target', 
                        help = 'what target are we using? binary, multiclass classification, or regression? Ex: "2", "1,3,10,30,100", "-1"')
    args = parser.parse_args()

    assert(args.note_type == 'radiology' or args.note_type == 'discharge')
    assert(args.model_type == 'T5' or args.model_type == 'Longformer')
    assert(args.noteid_mode == 'all' or args.noteid_mode == 'recent')
    args.GPU_NO = int(args.GPU_NO)
    finetuned_model_path = os.path.join(args.ckpt_dir, args.ckpt_model_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_NO)  # use the correct gpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    train_NOTE_TARGET_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}.csv'
    test_NOTE_TARGET_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}.csv'
    train_NOTE_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
    test_NOTE_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
    epoch = re.findall(r'\d+', args.ckpt_model_name)[-1]
    outfolder = f"{args.model_name}_{args.note_type[:3]}_{args.noteid_mode}_out/from_epoch{epoch}"
    out_dir = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/final{"/testing" if args.test else ""}/{outfolder}'
    out_embs_dir = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/embs{"/testing" if args.test else ""}/{outfolder}'

    if not os.path.exists(out_dir) and args.noteid_mode == 'recent':
        os.makedirs(out_dir)
    if not os.path.exists(out_embs_dir) and args.noteid_mode == 'all':
        os.makedirs(out_embs_dir)

    if args.noteid_mode == 'recent':
        run_cntr = find_next_dir_index(out_dir)
    elif args.noteif_mode == 'all':
        run_cntr = find_next_dir_index(out_embs_dir)

    out_dir = os.path.join(out_dir, str(run_cntr))
    out_embs_dir = os.path.join(out_embs_dir, str(run_cntr))

    if args.noteid_mode == 'recent':
        os.makedirs(out_dir)
        print(f'created all dirs in out_dir {out_dir}')
    else:
        os.makedirs(out_embs_dir)
        print(f'created all dirs in out_embs_dir {out_embs_dir}')

    out_embs_train_path = f'{out_embs_dir}/train_embs.pt'
    out_embs_test_path = f'{out_embs_dir}/test_embs.pt'

    train_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_train.csv')
    test_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_test.csv') 

    if args.test:
        train_NOTE_TARGET_path = os.path.join(os.path.dirname(train_NOTE_TARGET_path), 'testing', os.path.basename(train_NOTE_TARGET_path))
        test_NOTE_TARGET_path = os.path.join(os.path.dirname(test_NOTE_TARGET_path), 'testing', os.path.basename(test_NOTE_TARGET_path))
        train_NOTE_path = os.path.join(os.path.dirname(train_NOTE_path), 'testing', os.path.basename(train_NOTE_path))
        test_NOTE_path = os.path.join(os.path.dirname(test_NOTE_path), 'testing', os.path.basename(test_NOTE_path))



    tokenizer = None
    if args.model_type == 'T5':
        model_dir = os.path.join(f'{os.getenv("BHF_ROOT")}/models', args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    elif args.model_type == 'Longformer':
        model_path = "yikuan8/Clinical-Longformer"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
    assert(tokenizer != None)

    to_rename = args.target
    if ',' in args.target:
        to_rename = "delta_in_X_hours"
    elif re.match(r'^\d+$', args.target):
        to_rename = f"delta_in_{args.target}_hours"
    elif args.target == '-1':
        to_rename = "time_until_event"
    else:
        raise ValueError('invalid value for args.target')

    train_target_exploded = pd.read_csv(train_NOTE_TARGET_path, 
                               converters = {'NOTE_ID_SEQ': convert_to_list}).rename(columns = {to_rename:'label'})
                                                            
    test_target_exploded = pd.read_csv(test_NOTE_TARGET_path, 
                              converters = {'NOTE_ID_SEQ': convert_to_list}).rename(columns = {to_rename:'label'})
    
    if args.noteid_mode == 'all':
        train_target_exploded = explode_train_target(train_target_exploded)
        test_target_exploded = explode_train_target(test_target_exploded)

    train_target_exploded = merge_text(train_target_exploded, args.note_type)
    test_target_exploded = merge_text(test_target_exploded, args.note_type)
    tokenized_train_notes = df_to_tokens_ds(train_target_exploded)
    tokenized_test_notes = df_to_tokens_ds(test_target_exploded)
    print(f'tokenized_train_notes dataset:, {tokenized_train_notes}')
    print(f'tokenized_test_notes dataset:, {tokenized_test_notes}')

    tokenized_train_notes = pd.DataFrame(tokenized_train_notes)
    tokenized_test_notes = pd.DataFrame(tokenized_test_notes)
    print('tokenized notes converted back to dataframes for extraction...')
    # train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) # FIXME this is more elegant...
    # val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    batch_size = 24 # 48
    train_dataloader = generate_dataloader(tokenized_train_notes, batch_size, device)
    test_dataloader = generate_dataloader(tokenized_test_notes, batch_size, device)
    print("train, test, dataloaders generated")

    classifier = torch.load(finetuned_model_path) 
    print(f'classifier loaded from {finetuned_model_path}, class is {classifier.__class__}')

    classifier.encoder.eval()
    classifier.classifier.eval()
    classifier.eval() # here's some redundancy, but just in case...

    train_embeddings = extract_embeddings(train_dataloader, classifier)
    print("finished train embedding extraction") 
    test_embeddings = extract_embeddings(test_dataloader, classifier)
    print("finished test embedding extraction")

    if args.noteid_mode == 'all':
        '''
        save embeddings as a dataframe of 
        '''
        assert(len(train_embeddings) == len(train_target_exploded))
        assert(len(test_embeddings) == len(test_target_exploded))
        torch.save(train_embeddings, out_embs_train_path)
        print(f'saved to {out_embs_train_path}')
        torch.save(test_embeddings, out_embs_test_path)
        print(f'saved to {out_embs_test_path}')

    elif args.noteid_mode == 'recent':
        train_embeddings_df = embs_tensor_to_df(train_embeddings)
        test_embeddings_df = embs_tensor_to_df(test_embeddings)

        print("reading from", train_NOTE_TARGET_path)
        print("reading from", test_NOTE_TARGET_path)
        train = pd.read_csv(train_NOTE_TARGET_path)
        test = pd.read_csv(test_NOTE_TARGET_path)
        print(f"loaded train notes to extract from {train_NOTE_TARGET_path}")
        print(f"loaded test notes to extract from {test_NOTE_TARGET_path}")

        #concat notes with ICUSTAY column for merging later
        train_df_small = pd.concat([train[['ICUSTAY_ID', 'NOTE_ID']], train_embeddings_df], axis = 1)
        test_df_small = pd.concat([test[['ICUSTAY_ID', 'NOTE_ID']], test_embeddings_df], axis = 1)

        print(f"concatenating train and train_embeddings_df with shape {train.shape} and {train_embeddings_df.shape} respectively")
        print(f"concatenating test and test_embeddings_df with shape {test.shape} and {test_embeddings_df.shape} respectively")

        print(f"reading from temivef_train_NOTE_path")
        print(f"reading from temivef_test_NOTE_path")
        train_df_big = pd.read_csv(train_NOTE_path)
        test_df_big = pd.read_csv(test_NOTE_path)
        print(f"loaded train df to merge with data from {train_NOTE_path}")
        print(f"loaded test notes to merge with data from {test_NOTE_path}")
        
        train_df_big = merge_text(train_df_big, note_type = args.note_type)
        test_df_big = merge_text(test_df_big, note_type = args.note_type)
        print("merging and filling embeddings...")
        train_out_df = merge_and_fill_embeddings(train_df_small, train_df_big)
        print("merged and filled embeddings for train_out_df")
        test_out_df = merge_and_fill_embeddings(test_df_small, test_df_big)
        print("merged and filled embeddings for test_out_df")
        train_out_df = format_cols(train_out_df)
        test_out_df = format_cols(test_out_df)
        train_out_df.to_csv(train_outpath, index = False)
        print("wrote to", train_outpath)
        test_out_df.to_csv(test_outpath, index = False)
        print("wrote to", test_outpath)


