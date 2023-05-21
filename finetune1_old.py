import pandas as pd
import os
import torch.nn as nn
import sys
import torch

import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
# parser.add_argument('--freeze-encoder', action='store_true', help='freeze the encoder layers')
parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified')
parser.add_argument('--model-ckpt-path', dest = 'MODEL_CKPT_PATH', help='use model checkpoint specified')
args = parser.parse_args()

testing = args.test
# freeze_encoder = args.freeze_encoder
GPU_NO = int(args.GPU_NO)
model_ckpt_path = args.MODEL_CKPT_PATH

print(f'Test mode: {testing}')
# print(f'Freeze encoder: {freeze_encoder}')
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NO)  # use the correct gpu
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
if model_ckpt_path:
    print(f"model checkpoint to use: {model_ckpt_path}")
else:
    print("no ckpt specified. Finetuning from base")
# print(f"memory reserved: {torch.cuda.memory_reserved(device=device)} bytes")

root = '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0/'
data_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv'
model_path = root + 'Clinical-T5-Base/'
temivef_train_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT_rad.csv'

out_dir = f"{os.environ.get('CLINICAL_DIR')}model{'_test' if testing else ''}_rad{'_from_ckpt1' if model_ckpt_path else ''}" #FIXME change to ckpt2 if necessary
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
model_out = os.path.join(out_dir,"meta_ft_classify.pt")
print(f"model_out:{model_out}")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "Clinical-T5-Base"
tokenizer = AutoTokenizer.from_pretrained("Clinical-T5-Base")
model = AutoModelForSeq2SeqLM.from_pretrained("Clinical-T5-Base")

train = pd.read_csv(temivef_train_NOTE_TARGET1_FT_path)
print(f"reading notes and target from {temivef_train_NOTE_TARGET1_FT_path}")

train_tensor_filename = f"{os.environ.get('CLINICAL_DIR')}tokenized_notes_rad/train_tensor.pt"
train_tensors = None
if os.path.isfile(train_tensor_filename):
    train_tensors = torch.load(train_tensor_filename)
    print(f"tokenized note input_ids loaded from {train_tensor_filename}")
# else:
#     train_texts = train['text'].tolist()
#     tokenized_train_notes = tokenizer(train_texts, truncation=True, padding=True, return_tensors = "pt")
#     train_tensor = tokenized_train_notes.input_ids
#     torch.save(train_tensor, train_tensor_filename)
#     print(f"train notes tokenized and saved to {train_tensor_filename}")
# test_texts = test['text'].tolist()
# tokenized_test_notes = tokenizer(test_texts, truncation=True, padding=True, return_tensors = "pt")
# print("test notes tokenized")


from T5EncoderForSequenceClassification import T5EncoderForSequenceClassification

from transformers import T5Config

encoder = model.get_encoder() # we only need the clinical-t5 encoder for our purposes

config = T5Config(
    hidden_size=768,
    classifier_dropout=None,
    num_labels=2,
    hidden_dropout_prob=0.01,
    last_hidden_size=64
)

classifier = (
    T5EncoderForSequenceClassification(encoder, config).to(device) 
    if model_ckpt_path == None 
    else torch.load(model_ckpt_path)
)

print(f"CLASSIFIER LOADED TO GPU: {torch.cuda.memory_allocated() / 2**20} Megabytes")

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import get_linear_schedule_with_warmup
import numpy as np

# Define training data
train_inputs = train_tensors.input_ids.to(device)
train_labels = torch.tensor(train['delta_in_2_days'].to_numpy()).to(device)
train_dataset = TensorDataset(train_inputs, train_labels) 

print(f"DATASET LOADED TO GPU: {torch.cuda.memory_allocated() / 2**20} Megabytes")

# FROM https://github.com/BoXHED/BoXHED2.0/blob/master/packages/boxhed/boxhed/model_selection.py
# from boxhed.utils import temp_seed
def group_k_fold(ID, num_folds, seed=None):
    ID             = ID.astype(int)
    ID_unique_srtd = np.sort(np.unique(ID))

    ID_counts = [len(ID_unique_srtd)//num_folds] * num_folds
    for i in range(len(ID_unique_srtd)%num_folds):
        ID_counts[i]+=1

    assert sum(ID_counts)==len(ID_unique_srtd)

    fold_idxs = np.hstack([[i]*id_count for i, id_count in enumerate(ID_counts)])

    # if seed is not None:
    #     with temp_seed(seed):
    #         np.random.shuffle(fold_idxs)
    # else:
    np.random.shuffle(fold_idxs)

    gkf = []
    for fold_idx in range(num_folds):
        train_ids = ID_unique_srtd[np.where(fold_idxs!=fold_idx)[0]]
        test_ids  = ID_unique_srtd[np.where(fold_idxs==fold_idx)[0]]
        gkf.append((np.where(np.isin(ID, train_ids))[0], np.where(np.isin(ID, test_ids))[0]))
    
    return gkf

# define 5 fold cv
num_folds = 3
kfold = group_k_fold(train['ICUSTAY_ID'], num_folds)


batch_size = 5
num_epochs = 10
learning_rate = 1e-8
# max_grad_norm = 1.0


from torch.utils.data import Subset
import torch.optim as optim
from tqdm import tqdm
from time import time
import random
from transformers import Adafactor

criterion = nn.MSELoss()
optimizer = Adafactor(
    classifier.parameters(),
    lr=learning_rate, 
    scale_parameter=False, 
    relative_step=False
    # scale_parameter=True,
    # relative_step=True
)
# optimizer = optim.AdamW(classifier.classifier.parameters(), lr=learning_rate, eps=adam_epsilon)

# freeze encoder weights
# if freeze_encoder:
#     for param in classifier.encoder.parameters():
#         param.requires_grad = False
#     classifier.encoder.eval()
# else:
classifier.encoder.train()
# set classifier head to train
classifier.classifier.train()

def check_encoder_grad(encoder):
    # check if encoder parameters have gradients
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            print(name, param.grad is not None)

def forward(model, batch):
        inputs, labels = batch
        # print(f"BEFORE TRAIN FORWARD: {torch.cuda.memory_allocated() / 2**20} Megabytes")
        outputs = model.forward(inputs, labels=labels)
        # print(f"AFTER TRAIN FORWARD: {torch.cuda.memory_allocated() / 2**20} Megabytes")
        return outputs

def do_step(loss, model):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss.backward()
        # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        # optimizer.step()
        # optimizer.zero_grad()
        
        # scheduler.step()
        # # if step % 5 == 0:

def print_status(mode, epoch, num_epochs, fold, num_folds, step, num_steps, loss_val, start_time):
    print(f"Epoch {epoch + 1}/{num_epochs} | Fold {fold + 1}/{num_folds} | Step {step}/{num_steps} | Loss_{mode} {loss_val:.4f} | Time {time() - start_time : .2f} seconds | GPU_USAGE: {torch.cuda.memory_allocated() / 2**20} Megabytes")        


# num_batches = len(train_dataset) / batch_size * num_folds * ((num_folds - 1)/ num_folds)
# total_steps = int(num_batches * num_epochs)
# print(f"num_batches = {len(train_dataset)} / {batch_size} * {num_folds} * ({(num_folds - 1)}/ {num_folds}) = {num_batches}")
# print(f"total_steps = {num_batches} * {num_epochs} = {total_steps}")

# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=total_steps 
# )

# FINETUNE ###########################################################################
print("starting finetune")
start_time = time()

best_val_loss = float("inf")
epochs_since_improvement = 0
patience = 2

all_train_step_losses = []
all_val_step_losses = []

epoch_train_losses = []
epoch_val_losses = []

for epoch in range(num_epochs):
    # if testing and epoch >= 1:
    #     break

    fold_train_losses = []
    fold_val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold):
        print("len train_idx:", len(train_idx))
        train_set = Subset(train_dataset, train_idx)
        # pdb.set_trace()
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        batch_train_losses = []
        batch_val_losses = []

        for step, batch in enumerate(train_dataloader):
            if (testing and step >= 10):
                break

            outputs = forward(classifier, batch)
            do_step(outputs.loss, classifier)

            batch_train_losses.append(outputs.loss.item())
            if step % 10 == 0:
                print_status('train', epoch, num_epochs, fold, num_folds, step, len(train_dataloader), outputs.loss.item(), start_time)
        
        torch.cuda.empty_cache()

        print("len val_idx:", len(val_idx))
        val_set = Subset(train_dataset, val_idx)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                if (testing and step >= 10):
                    break
                
                outputs = forward(classifier, batch)
                batch_val_losses.append(outputs.loss.item())
                if step % 10 == 0:
                    print_status('val', epoch, num_epochs, fold, num_folds, step, len(val_dataloader), outputs.loss.item(), start_time)

        fold_train_losses.append(np.mean(batch_train_losses))
        fold_val_losses.append(np.mean(batch_val_losses))

        all_train_step_losses.append(batch_train_losses)
        all_val_step_losses.append(batch_val_losses)

        print(f'\n\tFOLD {fold + 1} HAS AVG TRAIN, VAL LOSS: {fold_train_losses[fold]},{fold_val_losses[fold]}\n')

    epoch_train_losses.append(np.mean(fold_train_losses))
    epoch_val_losses.append(np.mean(fold_val_losses))

    print(f'\n\tEPOCH {epoch + 1} HAS AVG TRAIN, VAL LOSS: {epoch_train_losses[epoch]},{epoch_val_losses[epoch]}\n')
    if epoch_val_losses[epoch] < best_val_loss:
        best_val_loss = epoch_val_losses[epoch]
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1
    if epochs_since_improvement >= patience:
        print(f"Validation loss did not improve for {patience} epochs. Early stopping...")
        break


torch.save(classifier, model_out)

print(f"model weights saved to path {model_out}")

import json
# train_losses_json = json.dumps(train_losses)
# val_losses_json = json.dumps(val_losses)



with open(os.path.join(out_dir,"epoch_train_losses.json"), "w") as outfile:
    json.dump(epoch_train_losses, outfile)

with open(os.path.join(out_dir,"epoch_val_losses.json"), "w") as outfile:
    json.dump(epoch_val_losses, outfile)

with open(os.path.join(out_dir,"all_train_step_losses.json"), "w") as outfile:
    json.dump(all_train_step_losses, outfile)

with open(os.path.join(out_dir,"all_val_step_losses.json"), "w") as outfile:
    json.dump(all_val_step_losses, outfile)

print(f"losses saved to path {out_dir}")
