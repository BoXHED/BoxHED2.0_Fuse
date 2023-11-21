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
from functools import partial
from transformers import LongformerTokenizerFast, AutoTokenizer

from BoXHED_Fuse.src.helpers import tokenization, load_all_notes, find_next_dir_index, explode_train_target, convert_to_list
from BoXHED_Fuse.models.ClinicalLSTM import ClinicalLSTM


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

    train_NOTE_TARGET_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_all.csv'
    test_NOTE_TARGET_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_test_NOTE_TARGET_2_{args.note_type[:3]}_all.csv'
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

    embs_train_path = f'{out_embs_dir}/train_embs.pt'
    embs_test_path = f'{out_embs_dir}/test_embs.pt'

    train_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_train.csv')
    test_outpath = os.path.join(out_dir, 'till_end_mimic_iv_extra_features_test.csv') 

    if args.test:
        train_NOTE_TARGET_path = os.path.join(os.path.dirname(train_NOTE_TARGET_path), 'testing', os.path.basename(train_NOTE_TARGET_path))
        test_NOTE_TARGET_path = os.path.join(os.path.dirname(test_NOTE_TARGET_path), 'testing', os.path.basename(test_NOTE_TARGET_path))
        train_NOTE_path = os.path.join(os.path.dirname(train_NOTE_path), 'testing', os.path.basename(train_NOTE_path))
        test_NOTE_path = os.path.join(os.path.dirname(test_NOTE_path), 'testing', os.path.basename(test_NOTE_path))

    # ===== Load Model from Checkpoint =====
    model = ClinicalLSTM()
    model.cuda()
    checkpoint = torch.load(finetuned_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    breakpoint()

    # ===== Extract Sequential Embeddings =====




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