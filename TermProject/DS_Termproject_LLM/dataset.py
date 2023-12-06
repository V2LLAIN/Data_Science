import torch
import warnings
import pandas as pd
from config import args
from util import sep, LOGGER
from tqdm.auto import tqdm
from tokenizer import tokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

# ======= OPTIONS =========
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current device is: {device}")
warnings.filterwarnings("ignore")

# ======= Load Data =========
train_df = pd.read_csv(args.TRAIN_ESSAYS, sep=',')
external_df = pd.read_csv(args.EXTERNAL_DATA, sep=',')
train_prompts = pd.read_csv(args.TRAIN_PROMPTS, sep=',')

# ===== Additional Data ======
external_df1 = external_df[["id", "source_text"]]
external_df1.columns = ["id", "text"]
external_df1['text'] = external_df['text'].str.replace('\n', '')
external_df1["generated"] = 1
external_df2 = external_df[["id", "text"]]
external_df2["generated"] = 0

train_df.drop(columns=["prompt_id"],inplace=True)
train_df = pd.concat([train_df, external_df1, external_df2])
train_df.reset_index(inplace=True, drop=True)
print(f"Train dataframe has shape: {train_df.shape}"), sep()


# ===== Validation Data ======
skf = StratifiedKFold(n_splits=5)
X = train_df.loc[:, train_df.columns != "generated"]
y = train_df.loc[:, train_df.columns == "generated"]

for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
    train_df.loc[valid_index, "fold"] = i

print(train_df.groupby("fold")["generated"].value_counts())





# ===== Dataset ======

import matplotlib.pyplot as plt

lengths = []
tqdm_loader = tqdm(train_df['text'].fillna("").values, total=len(train_df))
for text in tqdm_loader:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)

# config.MAX_LEN = max(lengths) + 3 # cls & sep & sep
LOGGER.info(f"max_len: {args.MAX_LEN}")
plt.hist(lengths, bins=25)
plt.xlabel('Token Length')  # x 레이블 추가
plt.ylabel('Frequency')     # y 레이블 추가
plt.show()

def prepare_input(cfg, text, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.MAX_LEN,
        padding='max_length', # TODO: check padding to max sequence in batch
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long) # TODO: check dtypes
    return inputs


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max()) # Get batch's max sequence length
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs


class CustomDataset(Dataset):
    def __init__(self, cfg, df, tokenizer):
        self.cfg = cfg
        self.texts = df['text'].values
        self.labels = df['generated'].values
        self.tokenizer = tokenizer
        self.text_ids = df['id'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        output = {}
        output["inputs"] = prepare_input(self.cfg, self.texts[item], self.tokenizer)
        output["labels"] = torch.tensor(self.labels[item], dtype=torch.float) # TODO: check dtypes
        output["ids"] = self.text_ids[item]
        return output



if args.DEBUG:
    # ======== SPLIT ==========
    fold = 0
    train_folds = train_df[train_df['fold'] != fold].reset_index(drop=True)
    valid_folds = train_df[train_df['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds['generated'].values

    # ======== DATASETS ==========
    train_dataset = CustomDataset(args, train_folds, tokenizer)
    valid_dataset = CustomDataset(args, valid_folds, tokenizer)

    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=args.BATCH_SIZE_TRAIN, # TODO: split into train and valid
                              shuffle=True,
                              num_workers=args.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=args.NUM_WORKERS, pin_memory=True, drop_last=False)

    # === Let's check one sample ===
    sample = train_dataset[0]
    print(f"Encoding keys: {sample.keys()} \n")
    print(sample)