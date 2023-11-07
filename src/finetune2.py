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

# FIXME import below need path from root
from BoXHED_Fuse.src.helpers import find_next_dir_index 
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

    train_embs_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_embs_{args.note_type[:3]}_{args.noteid_mode}.csv'
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
        train_embs_path = os.path.join(os.path.dirname(train_embs_path), 'testing', os.path.basename(train_embs_path))
        model_out_dir = os.path.join(os.path.dirname(model_out_dir), 'testing', os.path.basename(model_out_dir))

    # ===== Read Data =====
    # train_embs = pd.read_csv(train_embs_path)

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