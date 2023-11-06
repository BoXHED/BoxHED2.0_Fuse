
from typing import *
import os
import torch.nn as nn
import pandas as pd
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig, T5Config
from datasets import Dataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os
import argparse
from functools import partial

from BoXHED_Fuse.src.helpers import tokenization, convert_to_list

'''
EXAMPLE CALL:
python -m BoXHED_Fuse.src.finetune1 --test --use-wandb --gpu-no 3 --note-type radiology --model-name Clinical-T5-Base --model-type T5 --run-cntr 1 --num-epochs 1 --noteid-mode all
'''


# import sys
# print(sys.path)
# exit()

def group_train_test(ID) -> Tuple[List[int], List[int]]:
    ''' Creates shuffled 80/20 split for train and test data using indices.
    Args:
        ID: a pandas column of indexes
    
    Returns:
        A Tuple containing lists of indexes for train and test
    '''
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

def do_tokenization(train : pd.DataFrame, train_idxs : List[int], 
                    val_idxs : List[int]) -> Tuple[Dataset, Dataset]:
    ''' Turns a train pandas dataset into 
    Args:
        train: pandas dataframe containing training note data and the associated target label
        train_idxs: list of indexes for training data
        val_idxs: list of indexes for validation data
    
    Returns:
        A tuple of Datasets containing train and val data
    '''


    train_data = train.iloc[train_idxs]
    val_data = train.iloc[val_idxs]


    # FIXME get text

    train_data = Dataset.from_pandas(train_data).select_columns(['text', 'label'])
    val_data = Dataset.from_pandas(val_data).select_columns(['text', 'label'])




    if not os.path.exists(f'{out_dir}/data_cache'):
        # define a function that will tokenize the model, and will return the relevant inputs for the model
        train_data = train_data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(train_data) // 10)
        val_data = val_data.map(partial(tokenization, tokenizer, max_length=512), batched = True, batch_size = len(val_data) // 10)

        token_path_train = f'{out_dir}/data_cache/tokenized_train_data'
        token_path_val = f'{out_dir}/data_cache/tokenized_val_data'
        train_data.save_to_disk(token_path_train)
        val_data.save_to_disk(token_path_val)
        print(f'saved train, val tokens to {os.path.dirname(token_path_train)}')

    else: 
        print(f'loading train, val from', f'{out_dir}/data_cache/')
        train_data = train_data.load_from_disk(f'{out_dir}/data_cache/tokenized_train_data')
        val_data = val_data.load_from_disk(f'{out_dir}/data_cache/tokenized_val_data')

    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    train_data = train_data.remove_columns('text')
    val_data = val_data.remove_columns('text')

    return (train_data, val_data)

def compute_metrics(pred) -> Dict[str, float]:
    ''' Uses prediction label_ids and predicttions to compute precision recall, accuracy and f1. 
    '''
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

def explode(train):
    '''
    explode train['NOTE_ID_SEQ'] for meta-finetuning
    '''
    positive_notes = train[train['label'] == 1]['NOTE_ID'].tolist()
    train_sequences = pd.DataFrame(train[['ICUSTAY_ID', 'NOTE_ID_SEQ']].explode('NOTE_ID_SEQ').drop_duplicates())
    train_sequences.rename(columns={'NOTE_ID_SEQ' : 'NOTE_ID'}, inplace=True)
    train_sequences['label'] = train_sequences['NOTE_ID'].apply(lambda x: x in positive_notes)
    train_sequences['label'].replace({True: 1, False: 0}, inplace=True)
    train = train_sequences
    return train

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='enable testing mode')
parser.add_argument('--use-wandb', action = 'store_true', help = 'enable wandb', default=False)
parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified (this may be a single number or several. eg: 1 or 1,2,3,4)')
parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
parser.add_argument('--model-name', dest = 'model_name', help='model to finetune. ex: "Clinical-T5-Base"')
parser.add_argument('--model-type',  dest = 'model_type', help="T5 or Longformer?")
parser.add_argument('--run-cntr', dest = 'run_cntr', help = 'append this to dirname for storing additional runs')
parser.add_argument('--num-epochs', dest = 'num_epochs', help = 'num_epochs to train')
parser.add_argument('--ckpt-dir', dest = 'ckpt_dir', help = 'ckpt directory to load trainer from')
parser.add_argument('--ckpt-model-path', dest = 'ckpt_model_path', help = 'ckpt path to load model from') # change this to artifact
parser.add_argument('--noteid-mode', dest = 'noteid_mode', help = 'kw: all or recent')
# FIXME add target name functionality!!!
args = parser.parse_args()
args.num_epochs = int(args.num_epochs)

assert(args.note_type in  ['radiology', 'discharge'])
assert(args.model_name in ['Clinical-T5-Base', 'Clinical-T5-Large', 'Clinical-T5-Sci', 'Clinical-T5-Scratch', 'yikuan8/Clinical-Longformer'])
assert(args.model_type in ['T5', 'Longformer'])
assert(args.noteid_mode in ['recent', 'all'])
# assert(os.path.exists(args.ckpt_dir))
# assert(os.path.exists(args.ckpt_model_path))

print("finetune1.py args:")
for arg in vars(args):
    print(f"\t{arg}: {getattr(args, arg)}")

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NO

temivef_train_NOTE_TARGET1_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_{args.noteid_mode}.csv'
print(f'read from {temivef_train_NOTE_TARGET1_path}')
out_dir = f"model_outputs/{args.model_name}_{args.note_type[:3]}_{args.noteid_mode}_out/{args.run_cntr}"

if args.test:
    temivef_train_NOTE_TARGET1_path = os.path.join(os.path.dirname(temivef_train_NOTE_TARGET1_path), 'testing', os.path.basename(temivef_train_NOTE_TARGET1_path))
    out_dir = os.path.join(os.path.dirname(out_dir), 'testing', os.path.basename(out_dir))

# assert(not os.path.exists(out_dir)) # comment this out for testing purposes
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f'created all dirs in out dir', out_dir)

from transformers import AutoTokenizer, T5Config, AutoConfig, LongformerTokenizerFast, AutoModelForSequenceClassification, AutoModel, LongformerForSequenceClassification
from BoXHED_Fuse.models.T5EncoderForSequenceClassification import T5EncoderForSequenceClassification
from BoXHED_Fuse.models.ClinicalLongformerForSequenceClassification import ClinicalLongformerForSequenceClassification
tokenizer, classifier = None, None
if args.model_type == 'T5':
    from transformers import AutoModelForSeq2SeqLM, AutoModel

    model_dir = os.path.join('BoXHED_Fuse/models', args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)


    if args.ckpt_model_path:
        classifier = torch.load(args.ckpt_model_path)
        print('loaded model from ckpt model path:', args.ckpt_model_path)
    else:
        model = AutoModel.from_pretrained(model_dir)
        encoder = model.get_encoder() # we only need the clinical-t5 encoder for our purposes
        config_new = encoder.config
        config_new.num_labels=2
        config_new.last_hidden_size=64
        classifier = T5EncoderForSequenceClassification(encoder, config_new)

elif args.model_type == 'Longformer':
    model_path = "yikuan8/Clinical-Longformer"
    tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
    if args.ckpt_model_path:
        classifier = torch.load(args.ckpt_model_path)
        print('loaded model from ckpt model path:', args.ckpt_model_path)
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

print(f"reading notes and target from {temivef_train_NOTE_TARGET1_path}")
train = pd.read_csv(temivef_train_NOTE_TARGET1_path, converters = {'NOTE_ID_SEQ': convert_to_list})
target = 'delta_in_2_days'
train = train.rename(columns = {target:'label'})

if args.noteid_mode == 'all':
    print(f'noteid_mode {args.noteid_mode}: exploding NOTE_ID_SEQ')
    train = explode(train)

# if args.note_type == 'radiology':
#     all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv'
# if args.note_type == 'discharge':
#     all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
# print(f"reading all notes from {all_notes_path}")
# all_notes = pd.read_csv(all_notes_path)
# all_notes.rename(columns={'note_id': 'NOTE_ID'}, inplace=True)

from BoXHED_Fuse.src.helpers import load_all_notes
all_notes = load_all_notes(args.note_type)

# join train with all_notes
train = pd.merge(train, all_notes[['NOTE_ID','text']], on='NOTE_ID', how='left')

train_idxs, val_idxs = group_train_test(train['ICUSTAY_ID'])
train_data, val_data = do_tokenization(train, train_idxs, val_idxs)

# define the training arguments
training_args = TrainingArguments(
    output_dir = f'{out_dir}/results',
    num_train_epochs = args.num_epochs,
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    warmup_steps=200,
    weight_decay=0.01,
    fp16 = True,
    logging_dir=f'{out_dir}/logs',
    dataloader_num_workers = 0,
    run_name = f'{args.model_name}_{args.note_type}_run{args.run_cntr}',
)
if args.model_type == 'Longformer':
    training_args.learning_rate = 2e-5
    training_args.per_device_batch_size = 2
    training_args.gradient_accumulation_steps = 2 #3  # 8
    training_args.per_device_eval_batch_size = 4
    training_args.logging_steps = 4
    # training_args.fp16_backend="amp"    
elif args.model_type == 'T5':
    training_args.per_device_train_batch_size = 2 #5 # 2
    training_args.gradient_accumulation_steps = 8 #3  # 8
    training_args.per_device_eval_batch_size= 4 #10  # 4
    training_args.logging_steps = 4

# instantiate the trainer class and check for available devices
from BoXHED_Fuse.src.MyTrainer import MyTrainer

trainer = MyTrainer(
    model=classifier,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
if args.ckpt_dir:
    trainer._load_from_checkpoint(args.ckpt_dir)
    print('loaded trainer from trainer ckpt dir:', args.ckpt_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.use_wandb:
    # wandb.login(key=os.getenv('WANDB_KEY_PERSONAL'), relogin = True)
    wandb.login(key=os.getenv('WANDB_KEY_TAMU'), relogin = True)
    
    resume = args.ckpt_dir != None
    wandb.init(project='BoXHED_Fuse', name=training_args.run_name, resume=resume)
    wandb.run.name = training_args.run_name
    print(wandb.run.get_url())

trainer.train()
# torch.save(trainer.best_model, f'{out_dir}/besls
# t_model.pt')

import logging
logging.basicConfig(filename=f'{out_dir}/evaluation.log', level=logging.INFO, filemode='w')
evaluation_result = trainer.evaluate()
logging.info(evaluation_result)

best_checkpoint_path = trainer.state.best_model_checkpoint
logging.info(f"best_checkpoint_path: {best_checkpoint_path}")

print(f"RUN {training_args.run_name} FINISHED. Out dir: {out_dir}")