# pstage_03_semantic_segmentation

## Getting Started    
python3 train.py --MODEL 'segnet' --FILE_NAME 'first'
(config 파일 참조)

모델은 최저 val loss 기준으로 saved에 {model_name}_{file_name}.pt 식으로 저장됩니다. ex) segnet_first.pt

### 
evaluation.py
---
save_model 구현

dataset.py
---
데이터셋, 데이터로더
train, val loader와 test loader로 분리

inference.py
---
데이터를 불러와 추론 결과를 저장하는 공간
한번에 만들 수 있도록 구현

model.py
---
구현된 모델불러오기

utils.py
---
seed 고정 함수 
토론글 코드를 토대로 전체 val에 대한 mIoU 구하는 함수로 수정

loss.py
---
stage 1의 파일

config.py
---
args를 설정하는 파일

train.py
---
모델을 학습시키고 검증하여 val loss를 기준으로 모델을 저장. 상단의 WANDB = True/False 로 wandb 적용가능.

wandb를 처음 하신다면 
터미널에서 pip install wandb 후
wandb.ai 사이트에서 구글로 가입 후 로그인
터미널에서 wandb login 입력 
나타나는 url으로 ctrl + click
key 입력
상단의 WANDB = True
끝.

