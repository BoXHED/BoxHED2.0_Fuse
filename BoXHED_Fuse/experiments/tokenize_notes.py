# import pandas as pd
# import torch
# import os

# temivef_train_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_train_NOTE_TARGET1_FT_rad.csv'
# temivef_test_NOTE_TARGET1_FT_path = '/home/ugrads/a/aa_ron_su/JSS_SUBMISSION_NEW/data/till_end_mimic_iv_extra_features_test_NOTE_TARGET1_FT_rad.csv'

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# model_name = "Clinical-T5-Base"
# tokenizer = AutoTokenizer.from_pretrained("Clinical-T5-Base")

# train = pd.read_csv(temivef_train_NOTE_TARGET1_FT_path)
# print(f"reading notes and target from {temivef_train_NOTE_TARGET1_FT_path}")

# test = pd.read_csv(temivef_test_NOTE_TARGET1_FT_path)
# print(f"reading notes and target from {temivef_test_NOTE_TARGET1_FT_path}")

# print('CLINICAL_DIR:', os.environ.get('CLINICAL_DIR'))
# tensor_dir = os.path.join(os.environ.get('CLINICAL_DIR'), "tokenized_notes_rad")
# train_tensor_path = os.path.join(tensor_dir, "train_tensor.pt")
# test_tensor_path = os.path.join(tensor_dir, "test_tensor.pt")

# if not os.path.exists(tensor_dir):
#     os.makedirs(tensor_dir)


# train_texts = train['text'].tolist()
# tokenized_train_notes = tokenizer(train_texts, truncation=True, padding=True, return_tensors = "pt")
# train_tensor = tokenized_train_notes
# torch.save(train_tensor, train_tensor_path)
# print(f"train notes tokenized and saved to {train_tensor_path}")


# test_texts = test['text'].tolist()
# tokenized_test_notes = tokenizer(test_texts, truncation=True, padding=True, return_tensors = "pt")
# test_tensor = tokenized_test_notes
# torch.save(test_tensor, test_tensor_path)
# print(f"test notes tokenized and saved to {test_tensor_path}")