# COMMON_AI

# COMMON_AI
> AI 공통 테스트를 위한 `학습 전용 프로그램`으로 명명합니다.<br>

## Installation

```
git clone https://github.com/airiss-github/COMMON_AI
```

## Tech/Framework Used

- `Python 3.8.8`

__built with__
- `PyCharm`
- `VSCODE`


## Requirements

```
flask==1.1.2
flask-sqlalchemy==2.4.4
SQLAlchemy==1.3.24
psutil
requests
gpuinfo
pycryptodome
pyinstaller==4.5.1
tinyaes
opencv-python==4.5.3.56
colorama==0.4.4
PyQt5==5.15.4
pyqt5-plugins==5.15.4.2.2
PyQt5-Qt5==5.15.2
PyQt5-sip==12.9.0
pyqt5-tools==5.15.4.3.2
PyQtWebEngine==5.15.5
PyQtWebEngine-Qt5==5.15.2
python-dotenv==0.19.1
qt5-applications==5.15.2.2.2
qt5-tools==5.15.2.1.2
```

## Project List

```
library
┗ai
 ┗MaskRCNN
  ┗config
   ┗__init__.py
   ┗predict_config.py
   ┗train_config.py
   ┗view_config.py
  ┗dataset
   ┗dataset.py
  ┗jupyter
   ┗sample_predict.ipynb
   ┗sample_train.ipynb
  ┗model
   ┗model.py
  ┗utils
   ┗header.py
   ┗io.py
   ┗log.py
   ┗view.py
  ┗vision
  main.py
┗log  
┗dataset
.gitignore
predict.py
train.py
requirements.txt
README.md
```

## Explain

- ai/MaskRCNN
    - config : Train/Predict과 관련된 데이터 셋 파라메터 및 코딩에 따른 하이퍼 파라메터를 관리하는 모듈
    - dataset : COCO 형식으로 지정된 *.json 파일을 Dataset으로 변환하는 모듈
    - jupyter : 받은 *.ipynb 파일
    - utils : python에서 처리할 util 관련 모듈
    - vision : pytorch에서 예시로 등록된 mask rcnn 모듈
  
- log : 학습한 결과를 저장하는 모듈
- dataset

### TODO
- main.py : (클레스별 모듈 분류 및 재설정 진행, train의 restart 포함)

