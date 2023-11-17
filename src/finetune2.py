from typing import *
import pandas as pd
import os
import torch.nn as nn
import torch
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig, T5Config
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import argparse
from functools import partial

from BoXHED_Fuse.src.helpers import find_next_dir_index, explode_train_target, merge_embs_to_seq, convert_to_list
from BoXHED_Fuse.src.MyTrainer import MyTrainer 
from BoXHED_Fuse.models.ClinicalLSTM import ClinicalLSTM


if __name__ == '__main__':
    # ===== Initialize Args =====   
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='enable testing mode')
    parser.add_argument('--use-wandb', action = 'store_true', help = 'enable wandb', default=False)
    parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified (this may be a single number or several. eg: 1 or 1,2,3,4)')
    parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
    parser.add_argument('--num-epochs', dest = 'num_epochs', help = 'num_epochs to train')
    parser.add_argument('--noteid-mode', dest = 'noteid_mode', help = 'kw: all or recent')
    args = parser.parse_args()
    args.num_epochs = int(args.num_epochs)

    model_name_ft1 = 'Clinical-T5-Base'# FIXME add this as an arg
    train_target_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_{args.noteid_mode}.csv'
    train_embs_path =   f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs{"/testing" if args.test else ""}/{model_name_ft1}_{args.note_type[:3]}_{args.noteid_mode}_out/from_epoch1/1/train_embs.pt'
    model_name = 'clinical_lstm'
    model_out_dir = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/model_outputs/{model_name}_{args.note_type[:3]}_{args.noteid_mode}_out'
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    run_cntr = find_next_dir_index(model_out_dir)
    model_out_dir = os.path.join(model_out_dir, str(run_cntr))
    assert(not os.path.exists(model_out_dir))
    os.makedirs(model_out_dir)
    print(f'created all dirs in model_out_dir', model_out_dir)

    if args.test:
        train_target_path = os.path.join(os.path.dirname(train_target_path), 'testing', os.path.basename(train_target_path)) 
        model_out_dir = os.path.join(os.path.dirname(model_out_dir), 'testing', os.path.basename(model_out_dir))

    # ===== Read Data =====
    train_embs = torch.load(train_embs_path)
    train_target = pd.read_csv(train_target_path, converters = {'NOTE_ID_SEQ': convert_to_list})

    # ===== Merge data into {note_embs_seq, label}, where note_seq is a list of embs =====
    train_embs_seq = merge_embs_to_seq(train_embs, train_target)


    # ===== Train LSTM ===== 
    # clin_lstm = ClinicalLSTM()

    # trainer = MyTrainer(
    #     model=clin_lstm,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    # )

    # trainer.train()

    # ===== Save Sequential Embeddings =====
    # extract_emb_seq()

    # ===== Validate =====

    breakpoint()    
