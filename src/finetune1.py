#!/usr/bin/env python
# coding: utf-8


# In[1]:

import pandas as pd
import os
import torch.nn as nn
import pandas as pd
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig, T5Config


import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
parser.add_argument('--use-wandb', action = 'store_true', help = 'enable wandb')
parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified (this may be a single number or several. eg: 1 or 1,2,3,4)')
parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
parser.add_argument('--model-name', dest = 'model_name', help='model to finetune. ex: "Clinical-T5-Base"')
parser.add_argument('--model-type',  dest = 'model_type', help="T5 or LongFormer or Ckpt?")
parser.add_argument('--run-cntr', dest = 'run_cntr', help = 'append this to dirname for storing additional runs')
parser.add_argument('--num-epochs', dest = 'num_epochs', help = 'num_epochs to train')
parser.add_argument('--ckpt-dir', dest = 'ckpt_dir', help = 'ckpt directory to load trainer from')
parser.add_argument('--ckpt-model-path', dest = 'ckpt_model_path', help = 'ckpt path to load model from')

args = parser.parse_args()

args_list = [arg for arg in vars(args) if getattr(args, arg) is not None]
required_arguments = ['GPU_NO', 'note_type', 'run_cntr', 'num_epochs', 'model_type', 'model_name']  # List of required arguments
# Check if any of the required arguments are missing
if any(arg not in args_list for arg in required_arguments):
    raise ValueError(f"One or more required arguments are missing from {required_arguments}")

testing = args.test
use_wandb = args.use_wandb
GPU_NO = args.GPU_NO # this may be a single number or several ('1' or '1,2,3,4')
note_type = args.note_type
model_name = args.model_name
model_type = args.model_type
run_cntr = args.run_cntr
num_epochs = int(args.num_epochs)
ckpt_dir = args.ckpt_dir
ckpt_model_path = args.ckpt_model_path

print("finetune1.py args:")
for arg in vars(args):
    print(f"\t{arg}: {getattr(args, arg)}")

if use_wandb:
    wandb.login(key='2d62d7b2eea887cdb7783efd1978840a648f3fca') # suaaron
    # wandb.login(key='7b5d4393e8517657a9e973ce0133b4ffbd97ad3d', relogin=True) # aa_ron_su

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_NO
os.environ["WANDB_DISABLED"] = f"{'true' if not use_wandb else 'false'}"

root = '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0/'
temivef_train_NOTE_TARGET1_FT_path = \
f'/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/\
data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT\
{"_rad" if note_type == "radiology" else ""}.csv'
out_dir = f"{model_name}_{'rad' if note_type == 'radiology' else 'dis'}_{'test_' if testing else ''}out/{run_cntr}"


# In[4]:
from transformers import AutoTokenizer, T5Config, AutoConfig, LongformerTokenizerFast, AutoModelForSequenceClassification, AutoModel, LongformerForSequenceClassification
from T5EncoderForSequenceClassification import T5EncoderForSequenceClassification
from ClinicalLongformerForSequenceClassification import ClinicalLongformerForSequenceClassification
tokenizer, classifier = None, None
if model_type == 'T5':
    from transformers import AutoModelForSeq2SeqLM, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if ckpt_model_path:
        classifier = torch.load(ckpt_model_path)
        print('loaded model from ckpt model path:', ckpt_model_path)
    else:
        model = AutoModel.from_pretrained(model_name)
        encoder = model.get_encoder() # we only need the clinical-t5 encoder for our purposes
        config_new = encoder.config
        config_new.num_labels=2
        config_new.last_hidden_size=64
        classifier = T5EncoderForSequenceClassification(encoder, config_new)

elif model_type == 'Longformer':
    model_path = "yikuan8/Clinical-Longformer"
    tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
    if ckpt_model_path:
        classifier = torch.load(ckpt_model_path)
        print('loaded model from ckpt model path:', ckpt_model_path)
    else:
        from transformers import AutoModel
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2, gradient_checkpointing = True)
        longformer = model.get_submodule('longformer')
        config_new = longformer.config
        config_new.num_labels=2
        config_new.last_hidden_size=64
        config_new.gradient_checkpointing=True
        classifier = ClinicalLongformerForSequenceClassification(longformer, config_new)
else:
    print("incorrect model_type specified. Should be T5 or Longformer")
    exit(1)

# In[5]:

train = pd.read_csv(temivef_train_NOTE_TARGET1_FT_path)
print(f"reading notes and target from {temivef_train_NOTE_TARGET1_FT_path}")

# In[6]:
def group_train_test(ID):
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

train_idxs, val_idxs = group_train_test(train['ICUSTAY_ID'])
# In[7]:


from datasets import Dataset
target = 'delta_in_2_days'
train = train.rename(columns = {target:'label'})

train_data = train.iloc[train_idxs]
val_data = train.iloc[val_idxs]

if testing:
    train_data = train_data.iloc[:500]
    val_data = val_data.iloc[:500]

train_data = Dataset.from_pandas(train_data).select_columns(['text', 'label'])
val_data = Dataset.from_pandas(val_data).select_columns(['text', 'label'])

from functools import partial
from helpers import tokenization


if not os.path.exists(f'{out_dir}/data_cache'):
    # define a function that will tokenize the model, and will return the relevant inputs for the model
    train_data = train_data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(train_data) // 10)
    val_data = val_data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(val_data) // 10)

    train_data.save_to_disk(f'{out_dir}/data_cache/tokenized_train_data')
    val_data.save_to_disk(f'{out_dir}/data_cache/tokenized_val_data')

else: 
    print(f'loading train, val from', f'{out_dir}/data_cache/')
    train_data = train_data.load_from_disk(f'{out_dir}/data_cache/tokenized_train_data')
    val_data = val_data.load_from_disk(f'{out_dir}/data_cache/tokenized_val_data')

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_data = train_data.remove_columns('text')
val_data = val_data.remove_columns('text')

# In[8]:


# define accuracy metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # argmax(pred.predictions, axis=1)
    #pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


# In[10]:


# define the training arguments
training_args = TrainingArguments(
    output_dir = f'{out_dir}/results',
    num_train_epochs = num_epochs,
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    fp16 = True,
    logging_dir=f'{out_dir}/logs',
    dataloader_num_workers = 0,
    run_name = f'{model_name}_{note_type}_run{run_cntr}',
)
if model_type == 'Longformer':
    training_args.learning_rate = 2e-5
    training_args.per_device_batch_size = 2
    training_args.gradient_accumulation_steps = 2 #3  # 8
    training_args.per_device_eval_batch_size = 4
    training_args.logging_steps = 4
    # training_args.fp16_backend="amp"    
elif model_type == 'T5':
    training_args.per_device_train_batch_size = 2 #5 # 2
    training_args.gradient_accumulation_steps = 8 #3  # 8
    training_args.per_device_eval_batch_size= 4 #10  # 4
    training_args.logging_steps = 4

# In[11]:
# from transformers.callbacks import ModelCheckpoint
# checkpoint_callback = ModelCheckpoint(
#     dirpath=output_dir,
#     filename="model-{epoch:02d}",
#     save_top_k=-1,  # Save all models
#     monitor="epoch",
# )

# instantiate the trainer class and check for available devices
from MyTrainer import MyTrainer

trainer = MyTrainer(
    model=classifier,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
if ckpt_dir:
    trainer._load_from_checkpoint(ckpt_dir)
    print('loaded trainer from trainer ckpt dir:', ckpt_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In[12]:

if use_wandb:
    resume = ckpt_dir != None
    wandb.init(project='finetune llm', name=training_args.run_name, resume=resume)
    wandb.run.name = training_args.run_name
    print(wandb.run.get_url())

trainer.train()
# torch.save(trainer.best_model, f'{out_dir}/best_model.pt')

import logging
logging.basicConfig(filename=f'{out_dir}/evaluation.log', level=logging.INFO, filemode='w')
evaluation_result = trainer.evaluate()
logging.info(evaluation_result)

best_checkpoint_path = trainer.state.best_model_checkpoint
logging.info(f"best_checkpoint_path: {best_checkpoint_path}")

print(f"RUN {training_args.run_name} FINISHED: check out_dir! {out_dir}")


# %%
