import pandas as pd
from config import args
from datasets import Dataset
from tokenizers import models, normalizers, pre_tokenizers, trainers, Tokenizer

test = pd.read_csv(args.test)
sub = pd.read_csv(args.sub)
org_train = pd.read_csv(args.org_train)
train = pd.read_csv(args.train, sep=',')

train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

LOWERCASE = args.LOWERCASE
VOCAB_SIZE = args.VOCAB_SIZE

# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[['text']])