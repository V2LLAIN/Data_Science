# Dataset 및 추가 Data path:
https://github.com/V2LLAIN/Data_Science/tree/main/TermProject/Dataset

https://drive.google.com/file/d/1J0RKORgYZWjCIENYacyAsbPru-NjRRkd/view?usp=sharing
#
# 📕 학습진행하는방법:

### weight 조정 
#### 순서대로 MultinomialNB, SGDClassifier, LBBMClassifier, CatBoostClassifier
    python train.py --weights 0.1 0.2 0.3 0.4

### p6과 catboostclassifier의 learning_rate 조정
    python train.py --p6_learning_rate 2e-4 --cat_learning_rate 2e-4

### VOCAB_SIZE, MAX_LEN 조정
    python train.py --MAX_LEN 1024 --VOCAB_SIZE 31000

### JR Dataset으로 score 출력하는방법.  
(--test="경로" 설정한 후)
(아래코드 train.py에 추가및 변경)
    y_preds = ensemble.predict_proba(tf_test)[:, 1]
    sub = test.copy()
    sub['generated'] = y_preds > 0.5
    sub.to_csv(args.submission, index=False)
    sub['generated'] = y_preds > 0.5
    print(float(np.sum(test['generated'] == sub['generated'])) / len(test))




와 같은 방식으로 train 및 submission파일 생성이 가능합니다.

📌 추가적으로 
- 파일경로(test, sub, org_train, train, submission)
- max_iter 및 learning_rate, metric
- max_depth, max_bin, loss
- VOCAB_SIZE, MAX_LEN, LOWERCASE
- weights, voting종류
또한 terminal에서 학습 시 조정가능.
##
📌 Additionally, adjustable during training in the terminal:
- File paths (test, sub, org_train, train, submission)
- max_iter, learning_rate, metric
- max_depth, max_bin, loss
- VOCAB_SIZE, MAX_LEN, LOWERCASE
- weights, types of voting
##
