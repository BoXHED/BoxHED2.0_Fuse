import os
import torch
import argparse
import pandas as pd
import re
from torch.utils.data import DataLoader, TensorDataset
from time import time
import numpy as np
import torch
from datasets import Dataset
from BoXHED_Fuse.src.helpers import tokenization, load_all_notes
from functools import partial
from transformers import LongformerTokenizerFast, AutoTokenizer

def merge_text(data):
    all_notes = load_all_notes(args.note_type)
    data = pd.merge(data, all_notes[['NOTE_ID','text']], on='NOTE_ID', how='left')
    return data

def df_to_tokens_ds(data):
    '''
    Convert pandas DataFrame to Dataset containing text and lq
    abel
    '''
    data = Dataset.from_pandas(data).select_columns(['text', 'label'])
    data = data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(train_data) // 10) # def tokenization(tokenizer, batched_text, max_length):
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
    # FIXME save this to a new csv containing just embeddings
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
    df.drop(['text', 'NOTE_ID', 't_start_DT','INTIME'], axis=1, inplace=True)
    df.rename(columns = {
        'SUBJECT_ID':'subject',
        'ICUSTAY_ID':'Icustay'
        }, inplace = True)
    return df


if __name__ == '__main__':
    '''
    Use finetuned model to extract embeddings from train and test notes.

    If noteid_mode is "all", save embeddings to an embeddings only csv, which can be used later in sequential embeddings.
    If noteid_mode is "recent", populate train and test with embeddings.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='enable args.test mode')
    parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified')
    parser.add_argument('--ckpt-dir', dest = 'ckpt_dir', help='FULL PATH of directory where model checkpoint is stored')
    parser.add_argument('--ckpt-model-name', dest = 'ckpt_model_name', help='directory where model checkpoint is stored')
    parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
    parser.add_argument('--model-name', dest = 'model_name')
    parser.add_argument('--model-type', dest = 'model_type', help = 'T5 or Longformer')
    parser.add_argument('--run-cntr', dest = 'run_cntr', help = 'appended to dirname for storing additional runs')
    parser.add_argument('--noteid-mode', dest = 'noteid_mode', help = 'kw: all or recent')
    args = parser.parse_args()

    assert(args.note_type == 'radiology' or args.note_type == 'discharge')
    assert(args.model_type == 'T5' or args.model_type == 'Longformer')
    assert(args.noteid_mode == 'all' or args.noteid_mode == 'recent')
    args.GPU_NO = int(args.GPU_NO)
    finetuned_model_path = os.path.join(args.ckpt_dir, args.ckpt_model_name)
    print("extract_embeddings.py args:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_NO)  # use the correct gpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # finetuned_model_path = root + '/model_from_ckpt1/meta_ft_classify.pt' # modify this line!
    temivef_train_NOTE_TARGET_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_{args.noteid_mode}.csv'
    temivef_test_NOTE_TARGET_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_2_{args.note_type[:3]}_{args.noteid_mode}.csv'
    temivef_train_NOTE_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
    temivef_test_NOTE_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_{args.noteid_mode}.csv'
    epoch = re.findall(r'\d+', args.ckpt_model_name)[-1]
    outfolder = f"{args.model_name}_{args.note_type[:3]}_{'test_' if args.test else ''}out/{args.run_cntr}/from_epoch{epoch}"
    out_dir = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/final/{outfolder}'

    if args.test:
        temivef_train_NOTE_TARGET_path = os.path.join(os.path.dirname(temivef_train_NOTE_TARGET_path), 'testing', os.path.basename(temivef_train_NOTE_TARGET_path))
        temivef_test_NOTE_TARGET_path = os.path.join(os.path.dirname(temivef_test_NOTE_TARGET_path), 'testing', os.path.basename(temivef_test_NOTE_TARGET_path))
        temivef_train_NOTE_path = os.path.join(os.path.dirname(temivef_train_NOTE_path), 'testing', os.path.basename(temivef_train_NOTE_path))
        temivef_test_NOTE_path = os.path.join(os.path.dirname(temivef_test_NOTE_path), 'testing', os.path.basename(temivef_test_NOTE_path))
        out_dir = os.path.join(os.path.dirname(out_dir), 'testing', os.path.basename(out_dir))

    train_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_train.csv')
    test_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_test.csv') 

    if not os.path.exists(out_dir):
        print('made dir:', out_dir)
        os.makedirs(out_dir)

    tokenizer = None
    if args.model_type == 'T5':
        model_dir = os.path.join('BoXHED_Fuse/models', args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    elif args.model_type == 'Longformer':
        model_path = "yikuan8/Clinical-Longformer"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
    assert(tokenizer != None)


    target = 'delta_in_2_days'
    train_data = pd.read_csv(temivef_train_NOTE_TARGET_path).rename(columns = {target:'label'})
    test_data = pd.read_csv(temivef_test_NOTE_TARGET_path).rename(columns = {target:'label'})

    train_data = merge_text(train_data)
    test_data = merge_text(test_data)
    tokenized_train_notes = df_to_tokens_ds(train_data)
    tokenized_test_notes = df_to_tokens_ds(test_data)
    print(f'tokenized_train_notes dataset:, {tokenized_train_notes}')
    print(f'tokenized_test_notes dataset:, {tokenized_test_notes}')

    tokenized_train_notes = pd.DataFrame(tokenized_train_notes)
    tokenized_test_notes = pd.DataFrame(tokenized_test_notes)
    print('tokenized notes converted back to dataframes for extraction...')

    classifier = torch.load(finetuned_model_path) 
    print(f'classifier loaded from {finetuned_model_path}, class is {classifier.__class__}')

    if args.test:
        tokenized_train_notes = tokenized_train_notes[:12]
        tokenized_test_notes = tokenized_test_notes[:12]
        print(f"args.test mode truncated tokenized_notes to length {len(tokenized_train_notes)}")


    # train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label']) # FIXME this is more elegant...
    # val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    batch_size = 48
    train_dataloader = generate_dataloader(tokenized_train_notes, batch_size, device)
    test_dataloader = generate_dataloader(tokenized_test_notes, batch_size, device)
    print("train, test, dataloaders generated")

    # classifier = torch.load(finetuned_model_path)
    # classifier.encoder.eval() # makes sure dropout does not occur
    # classifier.classifier.eval()
    # print(f"loaded classifier from {finetuned_model_path}")


    classifier.encoder.eval()
    classifier.classifier.eval()
    classifier.eval() # here's some redundancy, but just in case...


    train_embeddings_df = extract_embeddings(train_dataloader, classifier)
    print("finished train embedding extraction") 
    test_embeddings_df = extract_embeddings(test_dataloader, classifier)
    print("finished test embedding extraction")

    # for step, batch in enumerate(train_dataloader):
    #     # Assuming each batch consists of input_ids, attention_mask, and labels
    #     input_ids, attention_mask = batch

    #     # Print or examine the data in each batch
    #     print('step:', step)
    #     print('Input IDs:', input_ids)
    #     print('Attention Mask:', attention_mask)

    #     print('---')


    print(f"reading from", temivef_train_NOTE_TARGET_path)
    print(f"reading from", temivef_test_NOTE_TARGET_path)
    train = pd.read_csv(temivef_train_NOTE_TARGET_path)
    test = pd.read_csv(temivef_test_NOTE_TARGET_path)
    print(f"loaded train notes to extract from {temivef_train_NOTE_TARGET_path}")
    print(f"loaded test notes to extract from {temivef_test_NOTE_TARGET_path}")

    #concat notes with ICUSTAY column for merging later
    train_df_small = pd.concat([train[['ICUSTAY_ID', 'NOTE_ID']], train_embeddings_df], axis = 1)
    test_df_small = pd.concat([test[['ICUSTAY_ID', 'NOTE_ID']], test_embeddings_df], axis = 1)

    print(f"concatenating train and train_embeddings_df with shape {train.shape} and {train_embeddings_df.shape} respectively")
    print(f"concatenating test and test_embeddings_df with shape {test.shape} and {test_embeddings_df.shape} respectively")

    for mode in ["train", "test"]:
        print(f"reading from temivef_{mode}_NOTE_path")
    train_df_big = pd.read_csv(temivef_train_NOTE_path)
    test_df_big = pd.read_csv(temivef_test_NOTE_path)
    breakpoint()
    train_df_big = merge_text(train_df_big)
    test_df_big = merge_text(test_df_big)
    print(f"loaded train df to merge with data from {temivef_train_NOTE_path}")
    print(f"loaded test notes to merge with data from {temivef_test_NOTE_path}")
    breakpoint()
    print("merging and filling embeddings...")
    train_out_df = merge_and_fill_embeddings(train_df_small, train_df_big)
    print("merged and filled embeddings for train_out_df")
    test_out_df = merge_and_fill_embeddings(test_df_small, test_df_big)
    print("merged and filled embeddings for test_out_df")
    breakpoint()
    train_out_df = format_cols(train_out_df)
    test_out_df = format_cols(test_out_df)

    train_out_df.to_csv(train_outpath, index = False)
    print("wrote to", train_outpath)
    test_out_df.to_csv(test_outpath, index = False)
    print("wrote to", test_outpath)
