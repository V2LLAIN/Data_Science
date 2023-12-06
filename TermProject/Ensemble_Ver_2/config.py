import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test', dest='test', default="./kaggle/input/llm-detect-ai-generated-text/test_essays.csv")
parser.add_argument('--sub', dest='sub', default="./kaggle/input/llm-detect-ai-generated-text/sample_submission.csv")
parser.add_argument('--org_train', dest='org_train', default="./kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
parser.add_argument('--train', dest='train', default="./kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
parser.add_argument('--submission', dest='submission', default="submission.csv")

parser.add_argument('--max_iter', dest='max_iter', type=int, default=8000)

parser.add_argument('--p6_learning_rate', dest='p6_learning_rate', type=float, default=0.05073909898961407)
parser.add_argument('--p6_metric', dest='p6_metric', default='auc')
parser.add_argument('--p6_max_depth', dest='p6_max_depth', type=int, default=23)
parser.add_argument('--p6_max_bin', dest='p6_max_bin', type=int, default=898)

parser.add_argument('--l2_leaf_reg', dest='l2_leaf_reg', type=float, default=6.6591278779517808)
parser.add_argument('--cat_iter', dest='cat_iter', type=int, default=1000)
parser.add_argument('--cat_learning_rate', dest='cat_learning_rate', type=float, default=0.005689066836106983)
parser.add_argument('--cat_loss', dest='cat_loss', default="CrossEntropy")

parser.add_argument('--VOCAB_SIZE', dest='VOCAB_SIZE', type=int, default=30522)
parser.add_argument('--MAX_LEN', dest='MAX_LEN', type=int, default=512)
parser.add_argument('--LOWERCASE', dest='LOWERCASE', type=bool, default=False)

# 리스트 형태의 입력 값을 받을 수 있도록 설정 (nargs='+' 사용)
parser.add_argument('--weights', dest='weights', metavar='W', nargs='+', type=float, default=[0.05, 0.05, 0.29, 0.29, 0.32])
parser.add_argument('--voting', dest='voting', default="soft")


args = parser.parse_args()
