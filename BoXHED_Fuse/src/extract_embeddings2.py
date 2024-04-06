import os
import torch
import argparse
import pandas as pd
import re
from torch.utils.data import DataLoader, TensorDataset
from torch.utils import data
from time import time
import numpy as np
import torch
from datasets import Dataset
from functools import partial
from transformers import LongformerTokenizerFast, AutoTokenizer

from BoXHED_Fuse.src.helpers import tokenization, load_all_notes, find_next_dir_index, explode_train_target, convert_to_list, Sequential_Dataset, merge_embs_to_seq
from BoXHED_Fuse.models.ClinicalLSTM import ClinicalLSTM

def generate_dataloader(emb_seq, batch_size):
    params = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 0, 
            'drop_last': True} # do not shuffle here. We will need to merge with IDs later
    
    my_data = emb_seq[['emb_seq', 'label']]
    my_data = Dataset.from_pandas(data)
    my_data.set_format('torch', columns=['label', 'emb_seq'])
    my_data = Sequential_Dataset(my_data)
    generator = data.DataLoader(my_data, **params)
    return generator

    # tdataset = Sequential_Dataset()

    
    # input_ids = torch.tensor(.input_ids, dtype=torch.long)
    # attention_mask = torch.tensor(tokenized_notes.attention_mask, dtype=torch.long)

    # tdataset = TensorDataset(input_ids.to(device),
    #                         attention_mask.to(device),
    #                         )
    # dataloader = DataLoader(tdataset, batch_size=batch_size, shuffle=False, drop_last=False) 
    # # shuffle is false so that notes retain their order for concat with df
    # return dataloader

def extract_embeddings(dataloader, model):
    model.eval()
    with(torch.no_grad()):
        start_time = time()
        embeddings = []
        for step, batch in enumerate(dataloader):
            input_ids, attention_mask = batch
            emb = model.forward(input_ids = input_ids, attention_mask = attention_mask, return_embeddings=True)
            embeddings.append(emb)
            print(f"Step {step}/{len(dataloader)} | Time {time() - start_time : .2f} seconds")

    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.cpu()
    return embeddings


if __name__ == '__main__':
    # INPUTS: 
    # - train and test note embeddings (stored as tensors)
    # - LSTM model checkpoint (stored as model.pt)
    # OUTPUTS:
    # - sequential embeddings (stored as tensors)
    # - final train and test df merged with sequential embeddings

    # ===== Initialize Args =====   
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='enable args.test mode')
    parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified')
    parser.add_argument('--ckpt-dir', dest = 'ckpt_dir', help='FULL PATH of directory where model checkpoint is stored')
    parser.add_argument('--ckpt-model-name', dest = 'ckpt_model_name', help='directory where model checkpoint is stored')
    parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
    parser.add_argument('--embs-dir', dest = 'embs_dir', help = 'dir of train.pt containing embeddings')
    args = parser.parse_args()
       
    assert(os.path.exists(args.embs_dir))
    assert(args.note_type == 'radiology' or args.note_type == 'discharge')
    args.GPU_NO = int(args.GPU_NO)
    finetuned_model_path = os.path.join(args.ckpt_dir, args.ckpt_model_name)

    MODEL_NAME = 'Clinical-LSTM'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_NO)  # use the correct gpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    train_target_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_all.csv'
    test_target_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_2_{args.note_type[:3]}_all.csv'
    train_NOTE_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_{args.note_type[:3]}_all.csv'
    test_NOTE_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_{args.note_type[:3]}_all.csv'
    epoch = re.findall(r'\d+', args.ckpt_model_name)[-1]
    outfolder = f"{MODEL_NAME}_{args.note_type[:3]}_all_out/from_epoch{epoch}"
    out_dir = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/final{"/testing" if args.test else ""}/{outfolder}'
    out_embs_dir = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/embs{"/testing" if args.test else ""}/{outfolder}'
    
    if not os.path.exists(out_embs_dir):
        os.makedirs(out_embs_dir)

    run_cntr = find_next_dir_index(out_embs_dir)
    out_dir = os.path.join(out_dir, str(run_cntr))
    out_embs_dir = os.path.join(out_embs_dir, str(run_cntr))
    os.makedirs(out_embs_dir)
    print(f'created all dirs in out_embs_dir {out_embs_dir}')

    train_emb_path = f'{args.embs_dir}/train_embs.pt'
    test_emb_path = f'{args.embs_dir}/test_embs.pt'

    # ===== Read Data =====
    print("READING train_emb, test_emb, train_target")
    train_emb = torch.load(train_emb_path)
    test_emb = torch.load(test_emb_path)
    train_target = pd.read_csv(train_target_path, converters = {'NOTE_ID_SEQ': convert_to_list})

    # ===== Merge data into {train_embs_seq, label} =====
    print("MERGING data into {train_embs_seq, label}")
    target = 'delta_in_2_days' 
    train_target.rename(columns = {target:'label'}, inplace=True)
    train_target_exploded = explode_train_target(train_target)
    train_emb_df = pd.DataFrame()
    train_emb_df['emb'] = [np.array(e) for e in train_emb]
    train_emb_df = pd.concat([train_target_exploded, train_emb_df], axis=1)
    train_emb_seq = merge_embs_to_seq(train_target, train_embs=train_emb_df)

    # train_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_train.csv')
    # test_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_test.csv') 

    if args.test:
        train_target_path = os.path.join(os.path.dirname(train_target_path), 'testing', os.path.basename(train_target_path))
        test_target_path = os.path.join(os.path.dirname(test_target_path), 'testing', os.path.basename(test_target_path))
        train_NOTE_path = os.path.join(os.path.dirname(train_NOTE_path), 'testing', os.path.basename(train_NOTE_path))
        test_NOTE_path = os.path.join(os.path.dirname(test_NOTE_path), 'testing', os.path.basename(test_NOTE_path))

    # ===== Load Model from Checkpoint =====
    model = ClinicalLSTM()
    model.cuda()
    checkpoint = torch.load(finetuned_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    # breakpoint()

    # ===== Extract Sequential Embeddings =====
    training_generator = generate_dataloader(emb_train, batch_size=32)
    



# model = ClinicalLSTM()
# # opt = torch.optim.Adam(model.parameters(), lr = 1e-5)

# checkpoint = torch.load('/home/ugrads/a/aa_ron_su/BoXHED_Fuse/BoXHED_Fuse/model_outputs/testing/clinical_lstm_rad_all_out/2/model.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# # opt.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']

# model.eval()
# # - or -
# model.train()