from typing import *
import os
import pandas as pd
from transformers import (
    LongformerTokenizerFast,
    TrainingArguments,
)
from datasets import Dataset
import torch.nn as nn
import torch
import wandb
import os
import argparse
from functools import partial
import traceback

from BoXHED_Fuse.src.helpers import (
    tokenization,
    convert_to_list,
    find_next_dir_index,
    load_all_notes,
    explode_train_target,
    compute_metrics,
    group_train_val,
)


"""
EXAMPLE CALL:
python -m BoXHED_Fuse.src.finetune1 --test --use-wandb --gpu-no 3 --note-type radiology --model-name Clinical-T5-Base --model-type T5 --run-cntr 1 --num-epochs 1 --noteid-mode all
"""


# import sys
# print(sys.path)
# exit()


def do_tokenization(
    train: pd.DataFrame, train_idxs: List[int], val_idxs: List[int]
) -> Tuple[Dataset, Dataset]:
    """Turns a train pandas dataset into
    Args:
        train: pandas dataframe containing training note data and the associated target label
        train_idxs: list of indexes for training data
        val_idxs: list of indexes for validation data

    Returns:
        A tuple of Datasets containing train and val data
    """

    train_data = train.iloc[train_idxs]
    val_data = train.iloc[val_idxs]

    # FIXME get text
    train_data = Dataset.from_pandas(train_data).select_columns(["text", "label"])
    val_data = Dataset.from_pandas(val_data).select_columns(["text", "label"])

    if not any(os.listdir(DATA_CACHE_DIR)):
        # define a function that will tokenize the model, and will return the relevant inputs for the model
        train_data = train_data.map(
            partial(tokenization, tokenizer, max_length=512),
            batched=True,
            batch_size=len(train_data) // 10,
        )
        val_data = val_data.map(
            partial(tokenization, tokenizer, max_length=512),
            batched=True,
            batch_size=len(val_data) // 10,
        )

        TOKEN_PATH_TRAIN = f"{DATA_CACHE_DIR}/tokenized_train_data"
        TOKEN_PATH_VAL = f"{DATA_CACHE_DIR}/tokenized_val_data"
        train_data.save_to_disk(TOKEN_PATH_TRAIN)
        val_data.save_to_disk(TOKEN_PATH_VAL)
        print(f"saved train, val tokens to {os.path.dirname(TOKEN_PATH_TRAIN)}")

    else:
        print(f"loading train, val from {DATA_CACHE_DIR}")
        train_data = train_data.load_from_disk(f"{DATA_CACHE_DIR}/tokenized_train_data")
        val_data = val_data.load_from_disk(f"{DATA_CACHE_DIR}/tokenized_val_data")

    train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_data = train_data.remove_columns("text")
    val_data = val_data.remove_columns("text")

    return (train_data, val_data)


def finetune(trainer):
    resume = args.ckpt_dir != None
    wandb.init(project=PROJECT_NAME, name=RUN_NAME, resume=resume)
    wandb.run.name = RUN_NAME
    print(wandb.run.get_url())

    trainer.train()
    # torch.save(trainer.best_model, f'{out_dir}/besls
    # t_model.pt')

    import logging

    logging.basicConfig(
        filename=f"{MODEL_OUT_DIR}/evaluation.log", level=logging.INFO, filemode="w"
    )
    evaluation_result = trainer.evaluate()
    logging.info(evaluation_result)

    best_checkpoint_path = trainer.state.best_model_checkpoint
    logging.info(f"best_checkpoint_path: {best_checkpoint_path}")

    print(f"RUN {RUN_NAME} FINISHED. Out dir: {MODEL_OUT_DIR}")


def sweep_func():
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)
    config = wandb.config
    print("Current wandb.config:", config)
    try:
        finetune(trainer=trainer)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="enable testing mode")
    parser.add_argument(
        "--use-wandb", action="store_true", help="enable wandb", default=False
    )
    parser.add_argument(
        "--gpu-no",
        dest="GPU_NO",
        help="use GPU_NO specified (this may be a single number or several. eg: 1 or 1,2,3,4)",
    )
    parser.add_argument(
        "--note-type", dest="note_type", help="which notes, radiology or discharge?"
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        help='model to finetune. ex: "Clinical-T5-Base"',
    )
    parser.add_argument("--model-type", dest="model_type", help="T5 or Longformer?")
    parser.add_argument("--num-epochs", dest="num_epochs", help="num_epochs to train")
    parser.add_argument(
        "--ckpt-dir", dest="ckpt_dir", help="ckpt directory to load trainer from"
    )
    parser.add_argument(
        "--ckpt-model-path", dest="ckpt_model_path", help="ckpt path to load model from"
    )  # change this to artifact
    parser.add_argument("--noteid-mode", dest="noteid_mode", help="kw: all or recent")
    parser.add_argument(
        "--target",
        dest="target",
        help='what target are we using? binary, multiclass classification, or regression? Ex: "2", "1,3,10,30,100", "-1"',
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="enable sweep, do not store checkpoints (BAYESIAN SWEEPS RUN FOREVER UNTIL MANUALLY STOPPED)",
        default=False,
    )
    parser.add_argument(
        "--sweep-id",
        dest="sweep_id",
        help="what sweep to attach to. If not specified, create a new sweep",
    )

    args = parser.parse_args()
    args.num_epochs = int(args.num_epochs)
    if args.sweep:
        args.use_wandb = True

    assert args.note_type in ["radiology", "discharge"]
    assert args.model_name in [
        "Clinical-T5-Base",
        "Clinical-T5-Large",
        "Clinical-T5-Sci",
        "Clinical-T5-Scratch",
        "yikuan8/Clinical-Longformer",
    ]
    assert args.model_type in ["T5", "Longformer"]
    assert args.noteid_mode in ["recent", "all"]
    # assert(os.path.exists(args.ckpt_dir))
    # assert(os.path.exists(args.ckpt_model_path))

    print("finetune1.py args:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NO

    train_path = f"{os.getenv('BHF_ROOT')}/data/targets/till_end_mimic_iv_extra_features_train_NOTE_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}.csv"
    print(f"read from {train_path}")

    MODEL_OUT_DIR = f"{os.getenv('BHF_ROOT')}/model_outputs/{args.model_name}_TARGET_{args.target}_{args.note_type[:3]}_{args.noteid_mode}_out"
    DATA_CACHE_DIR = os.path.join(MODEL_OUT_DIR, "data_cache")

    if "," in args.target:
        args.target = args.target.split(",")
        args.target = [int(t) for t in args.target]
        TARGET_LABEL = "delta_in_X_hours"
    else:
        args.target = int(args.target)
        TARGET_LABEL = (
            f"delta_in_{args.target}_hours" if args.target != -1 else "time_until_event"
        )
    print("target:", args.target)
    print("TARGET_LABEL:", TARGET_LABEL)

    if args.test:
        train_path = os.path.join(
            os.path.dirname(train_path), "testing", os.path.basename(train_path)
        )
        MODEL_OUT_DIR = os.path.join(
            os.path.dirname(MODEL_OUT_DIR), "testing", os.path.basename(MODEL_OUT_DIR)
        )

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    RUN_CNTR = find_next_dir_index(MODEL_OUT_DIR)
    MODEL_OUT_DIR = os.path.join(MODEL_OUT_DIR, str(RUN_CNTR))
    assert not os.path.exists(MODEL_OUT_DIR)
    os.makedirs(MODEL_OUT_DIR)
    print(f"created all dirs in model_out_dir", MODEL_OUT_DIR)

    from transformers import (
        AutoTokenizer,
        T5Config,
        AutoConfig,
        LongformerTokenizerFast,
        AutoModelForSequenceClassification,
        AutoModel,
        LongformerForSequenceClassification,
    )
    from BoXHED_Fuse.models.T5EncoderForSequenceClassification import T5EncoderWithHead
    from BoXHED_Fuse.models.ClinicalLongformerForSequenceClassification import (
        ClinicalLongformerForSequenceClassification,
    )

    tokenizer, full_model = None, None
    if TARGET_LABEL == "time_until_event":
        num_labels = 1  # regression
    elif isinstance(args.target, list):
        num_labels = len(args.target) + 1  # multiclass classification
    else:
        num_labels = 2  # binary classification

    if args.model_type == "T5":
        model_dir = os.path.join(f"{os.getenv('BHF_ROOT')}/models", args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if args.ckpt_model_path:
            full_model = torch.load(args.ckpt_model_path)
            print("loaded model from ckpt model path:", args.ckpt_model_path)
        else:
            model = AutoModel.from_pretrained(model_dir)
            encoder = (
                model.get_encoder()
            )  # we only need the clinical-t5 encoder for our purposes
            config_new = encoder.config
            config_new.num_labels = num_labels
            config_new.last_hidden_size = 64
            full_model = T5EncoderWithHead(encoder, config_new)

    elif args.model_type == "Longformer":
        model_path = "yikuan8/Clinical-Longformer"
        tokenizer = LongformerTokenizerFast.from_pretrained(model_path)
        if args.ckpt_model_path:
            full_model = torch.load(args.ckpt_model_path)
            print("loaded model from ckpt model path:", args.ckpt_model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=num_labels, gradient_checkpointing=True
            )
            longformer = model.get_submodule("longformer")
            config_new = longformer.config
            config_new.num_labels = num_labels
            config_new.last_hidden_size = 64
            config_new.gradient_checkpointing = True
            full_model = ClinicalLongformerForSequenceClassification(
                longformer, config_new
            )
    else:
        raise ValueError("incorrect model_type specified. Should be T5 or Longformer")

    print(f"reading notes and target from {train_path}")
    train = pd.read_csv(train_path, converters={"NOTE_ID_SEQ": convert_to_list})
    train = train.rename(columns={TARGET_LABEL: "label"})

    if args.noteid_mode == "all":
        print(f"noteid_mode {args.noteid_mode}: exploding NOTE_ID_SEQ")
        train = explode_train_target(train, TARGET_LABEL)

    # if args.note_type == 'radiology':
    #     all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv'
    # if args.note_type == 'discharge':
    #     all_notes_path = '/data/datasets/mimiciv_notes/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'
    # print(f"reading all notes from {all_notes_path}")
    # all_notes = pd.read_csv(all_notes_path)
    # all_notes.rename(columns={'note_id': 'NOTE_ID'}, inplace=True)

    all_notes = load_all_notes(args.note_type)

    # join train with all_notes
    if "text" not in train.columns:
        train = pd.merge(
            train, all_notes[["NOTE_ID", "text"]], on="NOTE_ID", how="left"
        )
    train_idxs, val_idxs = group_train_val(train["ICUSTAY_ID"])
    train_data, val_data = do_tokenization(train, train_idxs, val_idxs)

    # define the training arguments
    training_args = TrainingArguments(
        output_dir=f"{MODEL_OUT_DIR}/results",
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        disable_tqdm=False,
        load_best_model_at_end=True,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=True,
        logging_dir=f"{MODEL_OUT_DIR}/logs",
        dataloader_num_workers=0,
    )
    if args.model_type == "Longformer":
        training_args.learning_rate = 2e-5
        training_args.per_device_batch_size = 2
        training_args.gradient_accumulation_steps = 2  # 3  # 8
        training_args.per_device_eval_batch_size = 4
        training_args.logging_steps = 4
        # training_args.fp16_backend="amp"
    elif args.model_type == "T5":
        training_args.learning_rate = 2e-5
        training_args.per_device_train_batch_size = 2  # 5 # 2
        training_args.gradient_accumulation_steps = 8  # 3  # 8
        training_args.per_device_eval_batch_size = 4  # 10  # 4
        training_args.logging_steps = 4

    if args.sweep:
        training_args.save_checkpoint = False
    else:
        training_args.save_checkpoint = True

    # instantiate the trainer class and check for available devices
    from BoXHED_Fuse.src.MyTrainer import MyTrainer

    compute_metrics_partial = partial(compute_metrics, num_labels=num_labels)
    trainer = MyTrainer(
        model=full_model,
        args=training_args,
        compute_metrics=compute_metrics_partial,
        train_dataset=train_data,
        eval_dataset=val_data,
    )
    if args.ckpt_dir:
        trainer._load_from_checkpoint(args.ckpt_dir)
        print("loaded trainer from trainer ckpt dir:", args.ckpt_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    PROJECT_NAME = "BoXHED_Fuse"
    RUN_NAME = f"{args.model_name}_TARGET_{TARGET_LABEL}_{args.note_type[:3]}_{args.noteid_mode}_{RUN_CNTR}"

    # ==== SWEEP 1 ====
    sweep_configuration = {
        "method": "bayes",
        "name": f"sweep_{RUN_NAME}",
        "metric": {"goal": "minimize", "name": "Validation Loss"},
        "parameters": {
            "batch_size": {"values": [64, 128, 256]},
            "lr": {"max": 1e-4, "min": 1e-6},
            "scheduler": {"values": [True, False]},
        },
    }
    if args.use_wandb:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(key=os.getenv("WANDB_KEY"), relogin=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    if args.sweep:
        if args.sweep_id:
            print("Creating parallel agent on existing sweep with ID", args.sweep_id)
            wandb.agent(
                sweep_id=args.sweep_id, function=sweep_func, project=PROJECT_NAME
            )
        else:
            sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)
            print("Starting new sweep with ID", sweep_id)
            wandb.agent(sweep_id=sweep_id, function=sweep_func)
    else:
        finetune(trainer)
