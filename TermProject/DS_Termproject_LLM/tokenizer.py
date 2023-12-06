import tokenizers
import transformers
from config import args
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.MODEL)
tokenizer.save_pretrained(args.OUTPUT_DIR + '/tokenizer/')
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
print(tokenizer)