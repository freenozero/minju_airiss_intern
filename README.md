# minju_airiss_intern
> 230103~230224 동안 진행한 동계방학 인턴십입니다. <br>

## Installation

```
git clone https://github.com/airiss-data-analysis-with-intern/minju_airiss_intern
```

## Tech/Framework Used
- `Python 3.8.8`

__built with__
- `VSCODE`

## Requirements

```
absl-py==0.12.0
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
asttokens==2.2.1
attrs==22.2.0
autopep8==1.5.6
backcall==0.2.0
beautifulsoup4==4.11.1
bleach==6.0.0
cachetools==4.2.2
certifi==2020.12.5
cffi==1.15.1
chardet==4.0.0
charset-normalizer==3.0.1
colorama==0.4.6
comm==0.1.2
comtypes==1.1.14
contourpy==1.0.7
cycler==0.10.0
Cython==0.29.23
d2l==0.17.6
debugpy==1.6.6
decorator==5.1.1
defusedxml==0.7.1
entrypoints==0.4
escape==1.1
executing==1.2.0
fastjsonschema==2.16.2
fonttools==4.38.0
fqdn==1.5.1
google-auth==1.30.0
google-auth-oauthlib==0.4.4
grpcio==1.37.0
idna==2.10
importlib-metadata==4.0.1
importlib-resources==5.10.2
ipykernel==6.20.2
ipython==8.8.0
ipython-genutils==0.2.0
ipywidgets==8.0.4
isoduration==20.11.0
jedi==0.18.2
Jinja2==3.1.2
jsonpointer==2.3
jsonschema==4.17.3
jupyter==1.0.0
jupyter-console==6.4.4
jupyter-events==0.6.3
jupyter_client==7.4.9
jupyter_core==5.1.5
jupyter_server==2.1.0
jupyter_server_terminals==0.4.4
jupyterlab-pygments==0.2.2
jupyterlab-widgets==3.0.5
kiwisolver==1.3.1
Markdown==3.3.4
MarkupSafe==2.1.2
matplotlib==3.5.1
matplotlib-inline==0.1.6
mistune==2.0.4
natsort==8.2.0
nbclassic==0.5.1
nbclient==0.7.2
nbconvert==7.2.9
nbformat==5.7.3
nest-asyncio==1.5.6
notebook==6.5.2
notebook_shim==0.2.2
numpy==1.21.5
oauthlib==3.1.0
opencv-python==4.7.0.68
packaging==23.0
pandas==1.2.4
pandocfilters==1.5.0
parso==0.8.3
pickleshare==0.7.5
Pillow==9.4.0
pkgutil_resolve_name==1.3.10
platformdirs==2.6.2
prometheus-client==0.16.0
prompt-toolkit==3.0.36
protobuf==3.15.8
psutil==5.9.4
pure-eval==0.2.2
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycocotools==2.0.4
pycodestyle==2.7.0
pycparser==2.21
Pygments==2.14.0
pyparsing==2.4.7
pypiwin32==223
pyrsistent==0.19.3
python-dateutil==2.8.2
python-engineio==4.3.4
python-json-logger==2.0.4
pyttsx3==2.90
pytz==2022.7.1
pywin32==305
pywinpty==2.0.10
PyYAML==6.0
pyzmq==25.0.0
qtconsole==5.4.0
QtPy==2.3.0
requests==2.25.1
requests-oauthlib==1.3.0
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rsa==4.7.2
Send2Trash==1.8.0
sgmllib3k==1.0.0
six==1.15.0
sniffio==1.3.0
soupsieve==2.3.2.post1
stack-data==0.6.2
tensorboard==2.5.0
tensorboard-data-server==0.6.0
tensorboard-plugin-wit==1.8.0
terminado==0.17.1
tinycss2==1.2.1
toml==0.10.2
torch==1.13.1+cu117
torchmetrics==0.11.1
torchvision==0.14.1+cu117
tornado==6.2
traitlets==5.8.1
transforms==0.1
typing-extensions==3.7.4.3
uri-template==1.2.0
urllib3==1.26.4
utils==1.0.1
wcwidth==0.2.6
webcolors==1.12
webencodings==0.5.1
websocket-client==1.5.0
Werkzeug==1.0.1
widgetsnbextension==4.0.5
zipp==3.4.1
```

## Explain
### Project Folder
1. data_augmentation
```
ㄴ (원본) data_augmentation.py: 맨 처음 만든 augementation 하는 파일(이후에 리팩토링)

(원본) data_augmentation.py를 리팩토링한 모듈들
ㄴ run.py : 메인 run 파일
ㄴ augmentation
  ㄴ augmentation.py : 모든 폴더에 있는 files를 augmentation하는 파일
ㄴ library
  ㄴ utils
     ㄴ filesfolder.py : file 로딩 저장 관련
     ㄴ header.py
     ㄴ image.py : image 로딩 저장 관련
     ㄴ json.py : json 로딩 저장 관련
     ㄴ reshape.py : 이미지 reshpae 하는 파일
 
ㄴ augmentation_cateogrical.py: 칼, 총, 배터리, 레이저포인터가 여러 폴더로 나눠져있는데 해당 폴더 카테고리 정리 후 augmentation 하는 파일(리팩토링 하지 못함..)

```

2. manipulation: 하라고 요청이 들어온 순서가 뒤죽박죽이라.. 진행되는 방향이 난해한 편...
```
ㄴ libarary
    ㄴ utils
        ㄴ algorithm.py : 각종 필요한 알고리즘
        ㄴ header.py
        ㄴ io.py : 모든 파일 입출력 관련
     ㄴ background_manipulation.py : 이미지 합성
     ㄴ groundtruths.py : 합성한 이미지 카테고리 별로 json 잘 저장되어있는지 확인
     ㄴ jitter.py : jitter 시키기
     ㄴ split.py : run에서 설정한 세팅대로 train, val, test로 데이터 나누기
ㄴ run.py : 메인 run 파일

ㄴ resize.py : 이미지 700x700으로 맞추기
ㄴ test_image_del.py : test 이미지에서 사용하지 않는 카테고리 json에서 삭제

```

3. mask_rcnn: mask_rcnn으로 만든 이미지 학습하기
```
ㄴ library
  ㄴ config : 학습, 검증 이미지, groundtruths 이미지 설정
  ㄴ dataset : cocodataset
  ㄴ utils: 앞에 처럼 header, io, log, veiw 폴더 존재(특정 모적에 사용되도록 만듦)
  ㄴ vision/references: vision : torchvision 코드 (https://github.com/freenozero/vision)
  ㄴ main.py : dataset 로드, predict, train 함수 작성
ㄴ run.py : 메인 run 파일

(사수님께 받은 코드에서 몇몇 버그 수정하고 predict 코드, ap 구하기(11점 보간법) 작성)
```


### Remain Folder
1. tutorial_pytorch
```
- pytorch를 처음 하다보니 튜토리얼 사이트를 보면서 하나하나 클론코딩하면서 공부
```
2. Pytorch_Fintuning_Example
```
- 사수님께 받은 fintuning 예제 코드 클론 코딩하면서 공부
```
3. issue
```
- 원래 회사 레퍼지스토리에서 계속 관리한 issue가 있었는데, 끝나고 해당 레퍼지스토리 삭제하니 폴더로 따로 업로드
```
4. 발표
```
- 인턴십을 하면서 진행한 모든 것 발표 자료 정리
```
