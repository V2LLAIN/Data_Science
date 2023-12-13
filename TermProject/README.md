# 📌 Term Project
## ❗️Title: LLM - Detect AI Generated Text
### Identify which essay was written by a large language model

<img width="698" alt="스크린샷 2023-11-21 오후 8 39 17" src="https://github.com/V2LLAIN/Data_Science/assets/104286511/4378e801-2de8-4e1e-96b5-6fd138a6af16">

### 📕Datasets:
(https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data)

#### Additional Dataset for python code
(https://drive.google.com/file/d/1J0RKORgYZWjCIENYacyAsbPru-NjRRkd/view?usp=sharing)

#### Trained LLM Models Download link for your inference
https://drive.google.com/drive/folders/18HeRTCD9CG6f7ZIKmFzpLbbQROJjjzmU?usp=sharing
#
#
### 📕Overview
최근 몇 년 동안 대형 언어 모델(Large Language Models, LLMs)은 점점 더 정교해져서 인간이 쓴 텍스트와 거의 구분하기 어려운 텍스트를 생성할 수 있게 되었습니다. 본 대회에서는 실제 세계에서 적용 가능한 AI 감지 기술에 대한 연구와 투명성을 촉진하고자 합니다.
이 대회는 참가자들에게 학생이 쓴 에세이와 다양한 LLMs가 생성한 에세이를 혼합한 데이터셋에서 에세이가 학생에 의해 쓰였는지, 아니면 LLM에 의해 쓰였는지를 정확하게 감지할 수 있는 기계 학습 모델을 개발하도록 도전합니다!

#
#
#
### 📕Evaluation & Metric
#### 🚩 ROC-Curve아래영역(AUROC)
제출물은 예측 확률과 실제 대상 간의 ROC–Curve 아래 영역(Area Under the ROC Curve)으로 평가됩니다.

(https://huggingface.co/spaces/evaluate-metric/roc_auc)

#
#
#
### 📕Submission
테스트 세트의 각 ID에 대해 해당 에세이가 생성된 확률을 예측해야 합니다. 파일은 헤더를 포함하고 다음 형식을 가져야 합니다:

<img width="158" alt="스크린샷 2023-11-21 오후 8 43 24" src="https://github.com/V2LLAIN/Data_Science/assets/104286511/121da6f2-7596-41a6-bead-0969d92d14b3">


#
#
#
### ❗️Results: 
#### 🚩 Confusion Matrix about each Models
<img width="1024" alt="스크린샷 2023-11-21 오후 9 11 11" src="https://github.com/V2LLAIN/Data_Science/assets/104286511/5bfc60e8-5c09-492b-b838-888606de7fc4">
<img width="379" alt="스크린샷 2023-12-11 오후 10 31 39" src="https://github.com/V2LLAIN/Data_Science/assets/104286511/22ac947c-19ef-4a18-853c-b28024207b3d">



#
## 🚩 Submission_Results: 96% with Ensemble_Ver_1 Model
<img width="1205" alt="스크린샷 2023-12-06 오후 1 52 17" src="https://github.com/V2LLAIN/Data_Science/assets/104286511/851e2ec6-3828-431b-ad42-29ee13918c94">
