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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified')
args = parser.parse_args()

for arg in ['test, GPU_NO']:
    print("arg:", args.arg)

exit()


# In[2]:


config = T5Config()
config


# In[3]:


root = '/home/ugrads/a/aa_ron_su/physionet.org/files/clinical-t5/1.0.0/'
data_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
model_path = root + 'Clinical-T5-Base/'
finetune_model_path = root + 'Clinical-T5-Base_ft_vent/'
temivef_train_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT_rad.csv'
model_name = "Clinical-T5-Base"
out_dir = f"{model_name}_out"


# In[4]:


from transformers import T5Config
from T5EncoderForSequenceClassification import T5EncoderForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
encoder = model.get_encoder() # we only need the clinical-t5 encoder for our purposes

config = T5Config(
    hidden_size=768,
    classifier_dropout=None,
    num_labels=2,
    hidden_dropout_prob=0.01,
    last_hidden_size=64,
    gradient_checkpointing=True
)
classifier = T5EncoderForSequenceClassification(encoder, config)


# 

# In[5]:


train = pd.read_csv(temivef_train_NOTE_TARGET1_FT_path)
print(f"reading notes and target from {temivef_train_NOTE_TARGET1_FT_path}")


# In[6]:


def group_train_test(ID):
    ID             = ID.astype(int)
    ID_unique_srtd = np.unique(ID)
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

train_data = Dataset.from_pandas(train_data).select_columns(['text', 'label'])
val_data = Dataset.from_pandas(val_data).select_columns(['text', 'label'])

if not os.path.exists(f'{out_dir}/data_cache'):
    # define a function that will tokenize the model, and will return the relevant inputs for the model
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, max_length = 512)

    train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data) // 10)
    val_data = val_data.map(tokenization, batched = True, batch_size = len(val_data) // 10)

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
    num_train_epochs = 5,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8,    
    per_device_eval_batch_size= 4,
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    logging_steps = 8,
    fp16 = True,
    logging_dir=f'{out_dir}/logs',
    dataloader_num_workers = 0,
    run_name = 't5_radiology_run1'
)


# In[11]:


# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=classifier,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[12]:


import wandb
wandb.init()
print(wandb.run.get_url())
trainer.train()


# %%
