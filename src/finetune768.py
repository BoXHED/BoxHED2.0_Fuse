from typing import *
import pandas as pd
import os
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torch.utils import data
import torch
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig, T5Config
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import argparse
from functools import partial

from BoXHED_Fuse.src.log_artifact import log_artifact
from BoXHED_Fuse.src.helpers import find_next_dir_index, merge_embs_to_seq, convert_to_list, convert_to_nested_list, compute_metrics_LSTM, Sequential_Dataset, Sequential_Dataset_FAST, group_train_val, explode_train_target, validate_train_emb_seq
from BoXHED_Fuse.src.MyTrainer import MyTrainer 
from BoXHED_Fuse.models.ClinicalLSTM import ClinicalLSTM
import copy
# from Ranger import Ranger




def test_finetune(data_generator, model):
    '''
    tests finetuned model
    '''
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (output, label) in enumerate(data_generator):
        output = output.permute(1,0,2)
        score = model(output)
        
        m = torch.nn.Sigmoid()
        logits = torch.flatten(torch.squeeze(m(score)))
        loss_fct = torch.nn.BCELoss()    
        label = torch.from_numpy(np.array(label)).float().cuda()

        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        
        label_ids = label.to('cpu').numpy()
        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()
        outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    loss = loss_accumulate/count

    metrics = compute_metrics_LSTM(y_label, y_pred)

    return metrics, y_pred, loss.item()

def finetune(train_target, train_emb, lr, batch_size, train_epoch):
    '''
    finetune the model. Here, we are finetuning the LSTM!
    
    
    '''

    lr = lr
    BATCH_SIZE = batch_size
    train_epoch = train_epoch
    
    loss_history = []
    
    model = ClinicalLSTM()
    model.cuda()
    
    if torch.cuda.device_count() > 1:
        # breakpoint()
        # ARE YOU SURE YOU WANT TO USE MULTIPLE DEVICES?
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, dim = 1)
            
    print('--- Data Preparation ---')
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 0, 
              'drop_last': True}
    
    train_target_exploded = explode_train_target(train_target)
    train_target_exploded.reset_index(inplace=True)
    train_idxs, val_idxs = group_train_val(train_target['ICUSTAY_ID'])
    # train_data = train_emb_seq[['emb_seq', 'label']].iloc[train_idxs]
    # val_data = train_emb_seq[['emb_seq', 'label']].iloc[val_idxs]
    # train_data = Dataset.from_pandas(train_data)
    # val_data = Dataset.from_pandas(val_data)
    # train_data.set_format('torch', columns=['label', 'emb_seq'])
    # val_data.set_format('torch', columns=['label', 'emb_seq'])

    train_data = train_target.iloc[train_idxs]
    val_data = train_target.iloc[val_idxs]

    training_set = Sequential_Dataset_FAST(train_data, train_target_exploded, train_emb)
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Sequential_Dataset_FAST(val_data, train_target_exploded, train_emb)
    training_generator = data.DataLoader(validation_set, **params)
    
    
    opt = torch.optim.Adam(model.parameters(), lr = lr)
    # opt = Ranger(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(training_generator), epochs=train_epoch)
    # early stopping
    max_auc = 0
    model_max = copy.deepcopy(model)
   
    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    tstart = time.time()
    t0 = time.time()
    for epo in range(train_epoch):
        model.train()
        for i, (output, label) in enumerate(training_generator):
            print(f'time for loading batch: {time.time() - t0}')
            t0 = time.time()
            output = output.permute(1,0,2)
            score = model(output.cuda())
            print(f'time after forward: {time.time() - t0}')
            t0 = time.time()

            # print('output.shape:', output.shape)
            # print('score.shape:', score.shape)
            # print('label.shape:', label.shape)
            # breakpoint()
            label = torch.from_numpy(np.array(label)).float().cuda()
            loss_fct = torch.nn.BCELoss()
            m = torch.nn.Sigmoid()
            logits = torch.flatten(torch.squeeze(m(score)))
            print(f'time for flatten, squeeze, ... {time.time() - t0}')
            t0 = time.time()
            # breakpoint()
            loss = loss_fct(logits, label)
            loss_history.append(loss.item())

            if args.use_wandb:
                wandb.log({"Train Step Loss": loss.item(),
                    'Learning Rate': opt.param_groups[0]['lr'],
                    'Epoch': epo + (i+1) / (len(training_generator) - 1)})
            print("Train Step Loss", loss.item(),
                    'Learning Rate', opt.param_groups[0]['lr'],
                    'Epoch', epo + (i+1) / (len(training_generator) - 1),
                    'Time', int(time.time() - tstart))
            opt.zero_grad()
            loss.backward()
            print(f'time for backward: {time.time() - t0}')
            t0 = time.time()
            opt.step()
            scheduler.step()
            print(f'time for step: {time.time() - t0}')

           
        # every epoch test
        with torch.set_grad_enabled(False):
            # return test_finetune(validation_generator, model)
            metrics, logits, loss = test_finetune(validation_generator, model)
            
            if metrics['auc']:
                if metrics['auc'] > max_auc:
                    model_max = copy.deepcopy(model)
                    max_auc = metrics['auc']
                 
            if args.use_wandb:
                wandb.log({"Validation Loss": loss, "AUC": metrics['auc'], "AUPRC": metrics['auprc']})
            print('Validation at Epoch '+ str(epo + 1) + ' , AUC: '+ str(metrics['auc']) + ' , AUPRC: ' + str(metrics['auprc']))

        torch.save({
            'epoch': epo,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : opt.state_dict(),
            'loss': loss,
        }, f'{model_out_dir}/model_checkpoint_epoch{epo+1}.pt')
        print(f'saved checkpoint {epo+1} to {model_out_dir}/model_checkpoint_epoch{epo}.pt')
    # print('--- Go for Testing ---')
    # try:
    #     with torch.set_grad_enabled(False):
            
    #         auc, auprc, logits, loss = test_finetune(testing_generator, model_max)
    #         print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , Test loss: '+str(loss))
    # except:
    #     print('testing failed')
    return model_max, loss_history


if __name__ == '__main__':
    # ===== Initialize Args =====   
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='enable testing mode')
    parser.add_argument('--use-wandb', action = 'store_true', help = 'enable wandb', default=False)
    parser.add_argument('--gpu-no', dest = 'GPU_NO', help='use GPU_NO specified (this may be a single number or several. eg: 1 or 1,2,3,4)')
    parser.add_argument('--note-type', dest = 'note_type', help='which notes, radiology or discharge?')
    parser.add_argument('--num-epochs', dest = 'num_epochs', help = 'num_epochs to train', type=int)
    parser.add_argument('--noteid-mode', dest = 'noteid_mode', help = 'kw: all or recent')
    parser.add_argument('--embs-dir', dest = 'embs_dir', help = 'dir of train.pt containing embeddings')
    parser.add_argument('--batch-size', dest = 'batch_size', default=4, type=int)

    args = parser.parse_args()
    assert(os.path.exists(args.embs_dir))


    train_target_path = f'{os.getenv("BHF_ROOT")}/JSS_SUBMISSION_NEW/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_2_{args.note_type[:3]}_{args.noteid_mode}.csv'
    # train_embs_path = f'/home/ugrads/a/aa_ron_su/BoXHED_Fuse/BoXHED_Fuse/JSS_SUBMISSION_NEW/data/embs{"/testing"}/{"Clinical-T5-Base"}_{"rad"}_{"all"}_out/from_epoch1/10/train_embs.pt'

    MODEL_NAME = 'Clinical-LSTM'
    model_out_dir = f'{os.getenv("BHF_ROOT")}/model_outputs/{MODEL_NAME}_{args.note_type[:3]}_{args.noteid_mode}_out'
    
    if args.test:
        train_target_path = os.path.join(os.path.dirname(train_target_path), 'testing', os.path.basename(train_target_path)) 
        model_out_dir = os.path.join(os.path.dirname(model_out_dir), 'testing', os.path.basename(model_out_dir))
    
    train_emb_seq_path = f'{model_out_dir}/train_emb_seq.csv' # tmp csv for multiple runs

    
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    RUN_CNTR = find_next_dir_index(model_out_dir)
    model_out_dir = os.path.join(model_out_dir, str(RUN_CNTR))
    assert(not os.path.exists(model_out_dir))
    os.makedirs(model_out_dir)
    print(f'created all dirs in model_out_dir', model_out_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_NO)
    assert(args.batch_size % len(args.GPU_NO.split(',')) == 0)

    # ===== Read Data =====
    print('reading data')
    train_emb = torch.load(f'{args.embs_dir}/train_embs.pt')
    train_target = pd.read_csv(train_target_path, converters = {'NOTE_ID_SEQ': convert_to_list})
    # ===== Merge data into {train_embs_seq, label} =====
    print('merging data into df with cols {train_embs_seq, label}')
    target = 'delta_in_2_days' 
    train_target.rename(columns = {target:'label'}, inplace=True)


    # train_emb_seq_path = f"train_emb_seq_path_TMP_{find_next_dir_index('.')}" # FIXME debug purposes
    # if (not os.path.exists(train_emb_seq_path)):
    #     print('merging data into df with cols {train_embs_seq, label}')
    #     target = 'delta_in_2_days' 
    #     train_target.rename(columns = {target:'label'}, inplace=True)
    #     train_target_exploded = explode_train_target(train_target)
    #     train_emb_df = pd.DataFrame()
    #     train_emb_df['emb'] = [np.array(e) for e in train_emb]
    #     train_emb_df = pd.concat([train_target_exploded, train_emb_df], axis=1)
    #     print('merging on NOTE_ID_SEQ...')
    #     train_emb_seq = merge_embs_to_seq(train_target, train_embs=train_emb_df)
    #     validate_train_emb_seq(train_emb_seq, train_target)
    #     train_emb_seq.to_csv(train_emb_seq_path, index=False)
    # else:
    #     print('train_embs_seq exists!')
    #     train_emb_seq = pd.read_csv(train_emb_seq_path, converters={'emb_seq': convert_to_nested_list})
    #     # breakpoint()

    # ===== Train LSTM ===== 
    RUN_NAME = f'{MODEL_NAME}_{args.note_type[:3]}_{args.noteid_mode}_{RUN_CNTR}'
    if args.use_wandb:
        # wandb.login(key=os.getenv('WANDB_KEY_PERSONAL'), relogin = True)
        wandb.login(key=os.getenv('WANDB_KEY_TAMU'), relogin = True)
        
        # resume = args.ckpt_dir != None
        wandb.init(project='BoXHED_Fuse', name=RUN_NAME)
        print(wandb.run.get_url())
    
    # log_artifact(artifact_path = out_trainpath,
    #             artifact_name = os.path.splitext(os.path.basename(out_trainpath))[0] + '.test' if args.test else '',
    #             artifact_description = "MIMIC IV joined with note data for finetuning",
    #             artifact_metadata= dict(args._get_kwargs()),
    #             project_name = os.getenv('WANDB_PROJECT_NAME'),
    #             do_filter=True,)

    lr = 1e-5 
    batch_size = args.batch_size
    train_epoch = args.num_epochs

    out = finetune(train_target, train_emb, lr, batch_size, train_epoch)
    # out = finetune(train_target, train_embs, lr, batch_size, train_epoch)
    # breakpoint()
    # ===== Save Sequential Embeddings =====
    # extract_emb_seq()

    # ===== Validate =====

