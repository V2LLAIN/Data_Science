import argparse
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument('--APEX', dest='APEX', type=bool, default=True)
parser.add_argument('--BATCH_SCHEDULER', dest='BATCH_SCHEDULER', type=bool, default=True)
parser.add_argument('--BATCH_SIZE_TRAIN', dest='BATCH_SIZE_TRAIN', type=int, default=32)
parser.add_argument('--BATCH_SIZE_VALID', dest='BATCH_SIZE_VALID', type=int, default=16)
parser.add_argument('--BETAS', dest='BETAS', nargs='+', type=float, default=[0.9, 0.999])
parser.add_argument('--DEBUG', dest='DEBUG', type=bool, default=False)
parser.add_argument('--DECODER_LR', dest='DECODER_LR', type=float, default=2e-5)
parser.add_argument('--ENCODER_LR', dest='ENCODER_LR', type=float, default=2e-5)
parser.add_argument('--EPOCHS', dest='EPOCHS', type=int, default=5)
parser.add_argument('--EPS', dest='EPS', type=float, default=1e-6)
parser.add_argument('--FOLDS', dest='FOLDS', type=int, default=4)
parser.add_argument('--GRADIENT_ACCUMULATION_STEPS', dest='GRADIENT_ACCUMULATION_STEPS', type=int, default=1)
parser.add_argument('--GRADIENT_CHECKPOINTING', dest='GRADIENT_CHECKPOINTING', type=bool, default=True)
parser.add_argument('--MAX_GRAD_NORM', dest='MAX_GRAD_NORM', type=int, default=1000)
parser.add_argument('--MAX_LEN', dest='MAX_LEN', type=int, default=512)
parser.add_argument('--MIN_LR', dest='MIN_LR', type=float, default=1e-6)
parser.add_argument('--NUM_CYCLES', dest='NUM_CYCLES', type=float, default=0.5)
parser.add_argument('--NUM_WARMUP_STEPS', dest='NUM_WARMUP_STEPS', type=int, default=0)
parser.add_argument('--NUM_WORKERS', dest='NUM_WORKERS', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--PRINT_FREQ', dest='PRINT_FREQ', type=int, default=20)
parser.add_argument('--SCHEDULER', dest='SCHEDULER', default='cosine')
parser.add_argument('--SEED', dest='SEED', type=int, default=27)
parser.add_argument('--TRAIN', dest='TRAIN', action='store_true', default=True)
parser.add_argument('--TRAIN_FOLDS', dest='TRAIN_FOLDS', nargs='+', type=int, default=[0, 1, 2, 3])
parser.add_argument('--WANDB', dest='WANDB', action='store_true', default=False)
parser.add_argument('--WEIGHT_DECAY', dest='WEIGHT_DECAY', type=float, default=0.01)
parser.add_argument('--OUTPUT_DIR', dest='OUTPUT_DIR', default='./kaggle/working/output')
parser.add_argument('--EXTERNAL_DATA', dest='EXTERNAL_DATA', default='./kaggle/input/daigt-external-dataset/daigt_external_dataset.csv')
parser.add_argument('--TRAIN_PROMPTS', dest='TRAIN_PROMPTS', default='./kaggle/input/llm-detect-ai-generated-text/train_prompts.csv')
parser.add_argument('--TRAIN_ESSAYS', dest='TRAIN_ESSAYS', default='./kaggle/input/llm-detect-ai-generated-text/train_essays.csv')
parser.add_argument('--TEST_ESSAYS', dest='TEST_ESSAYS', default='./kaggle/input/llm-detect-ai-generated-text/test_essays.csv')

parser.add_argument('--MODEL_PATH', dest='MODEL_PATH', default="./kaggle/working/output")
parser.add_argument('--BEST_MODEL_PATH', dest='BEST_MODEL_PATH', default="./kaggle/working/output/microsoft_deberta-v3-base_fold_0_best.pth")
parser.add_argument('--SUBMISSION_CSV', dest='SUBMISSION_CSV', default="./kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")

parser.add_argument('--wandb_api_key', dest='wandb_api_key', default="eed81e1c0a41dd8dd67a4ca90cea1be5a06d4eb0")
parser.add_argument('--wandb_project', dest='wandb_project', default="DS_Termproject")
parser.add_argument('--wandb_entity', dest='wandb_entity', default="hcim")
parser.add_argument('--wandb_name', dest='wandb_name', default="DeBERTa")
parser.add_argument('--wandb_log_model', dest='wandb_log_model', default="all")

parser.add_argument('--MODEL', dest='MODEL', default='microsoft/deberta-v3-base')


parser.add_argument('--BATCH_SIZE_TEST', dest='BATCH_SIZE_TEST', type=int, default=8)
parser.add_argument('--max_len', dest='max_len', type=int, default=1024)


args = parser.parse_args()


if args.DEBUG:
    args.EPOCHS = 2
    args.TRAIN_FOLDS = [0]