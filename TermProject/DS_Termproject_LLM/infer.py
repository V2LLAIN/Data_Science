import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import warnings
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from config import args
from model import CustomModel
from dataset import collate, prepare_input, LOGGER
from util import seed_everything, sep, sigmoid

# ======= OPTIONS =========
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is: {device}")
warnings.filterwarnings("ignore")


"""
Utils
"""
seed_everything(seed=args.SEED)
test_df = pd.read_csv(args.TEST_ESSAYS, sep=',')
print(f"Test summaries dataframe has shape: {test_df.shape}"), sep()


"""
Tokenizer
"""
# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(args.MODEL_PATH + "/tokenizer")
# === Add special tokens ===
vocabulary = tokenizer.get_vocab()
total_tokens = len(vocabulary)
print("Total number of tokens in the tokenizer:", total_tokens)
print(tokenizer)


"""
Dataset
"""
def prepare_input(cfg, text, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=args.max_len,
        padding='max_length', # TODO: check padding to max sequence in batch
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long) # TODO: check dtypes
    return inputs

lengths = []
tqdm_loader = tqdm(test_df['text'].fillna("").values, total=len(test_df))
for text in tqdm_loader:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)

LOGGER.info(f"max_len: {args.max_len}")
_ = plt.hist(lengths, bins=25)


class CustomDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.texts = df['text'].values
        self.tokenizer = tokenizer
        self.ids = df['id'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        output = {}
        output["inputs"] = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        output["ids"] = self.ids[item]
        return output

if args.DEBUG:
    # ======== DATASETS ==========
    test_dataset = CustomDataset(args, test_df, tokenizer)

    # ======== DATALOADERS ==========
    test_loader = DataLoader(test_dataset,
                             batch_size=args.BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=args.NUM_WORKERS,
                             pin_memory=True, drop_last=False)

    # === Let's check one sample ===
    sample = test_dataset[0]
    print(f"Encoding keys: {sample.keys()} \n")
    print(sample)



"""
Infer
"""


def inference_fn(config, test_df, tokenizer, device):
    # ======== DATASETS ==========
    test_dataset = CustomDataset(config, test_df, tokenizer)

    # ======== DATALOADERS ==========
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE_TEST,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True, drop_last=False)

    # ======== MODEL ==========
    model = CustomModel(config, config_path=args.MODEL_PATH + "/config.pth", pretrained=False)
    state = torch.load(args.BEST_MODEL_PATH)
    model.load_state_dict(state)
    model.to(device)
    model.eval()  # set model in evaluation mode
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, batch in enumerate(tqdm_test_loader):
            inputs = batch.pop("inputs")
            ids = batch.pop("ids")
            inputs = collate(inputs)  # collate inputs
            for k, v in inputs.items():
                inputs[k] = v.to(device)  # send inputs to device
            with torch.no_grad():
                y_preds = model(inputs)  # forward propagation pass
            preds.append(y_preds.to('cpu').numpy())  # save predictions

    prediction_dict["predictions"] = np.concatenate(preds)  # np.array() of shape (fold_size, target_cols)
    prediction_dict["ids"] = ids
    return prediction_dict

predictions = inference_fn(args, test_df, tokenizer, device)
submission = pd.read_csv(args.SUBMISSION_CSV)
submission["generated"] = predictions["predictions"]
submission["generated"] = submission["generated"].apply(lambda x: sigmoid(x))
submission.to_csv("submission.csv", index=False)