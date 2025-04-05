# PyTorch Image Classification - 화성암 및 우주암석 분류 모델

## 프로젝트 개요
이 프로젝트는 **PyTorch와 ResNet-50을 활용하여 화성암 및 우주암석을 분류하는 딥러닝 이미지 분류 모델**입니다.
사용자는 학습된 모델(`moonrockmodel.pth`)을 활용하여 새로운 이미지에 대한 예측을 수행할 수 있습니다.

## 주요 기능
- **ResNet-50 전이 학습(Transfer Learning) 적용**
- **이미지 데이터를 224x224 크기로 변환하여 학습**
- **화성암(Basalt 등) 및 다양한 우주암석 데이터 활용**
- **Jupyter Notebook 기반 실험 및 데이터 시각화**

## 프로젝트 구조
```
PyTorch_excersize/
│── ClassifySpaceRockCode.ipynb  # PyTorch 이미지 분류 모델 학습 코드
│── moonrockmodel.pth  # 학습된 ResNet-50 모델
│── requirements.txt  # 필요 라이브러리 목록
│── README.md  # 프로젝트 설명 파일
│── data/
│   ├── atlantis.csv  # 데이터셋 CSV 파일
│   ├── Basalt/  # 화성암 이미지 데이터
│── .devcontainer/
│   ├── devcontainer.json  # VS Code 개발 환경 설정
└── LICENSE  # 프로젝트 라이선스
```

## 설치 및 실행 방법
### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. Jupyter Notebook 실행
```bash
jupyter notebook
```
Jupyter Notebook에서 `ClassifySpaceRockCode.ipynb` 파일을 열어 실행합니다.

### 3. PyTorch 모델 실행
```python
import torch
from torchvision import transforms
from PIL import Image

# 모델 로드
model = torch.load("moonrockmodel.pth")
model.eval()

# 이미지 변환 및 예측
image = Image.open("data/Basalt/sample.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
predicted_class = output.argmax(1).item()
print("Predicted class:", predicted_class)
```

## 필요 라이브러리
- `torch`
- `torchvision`
- `matplotlib`
- `pandas`
- `numpy`
- `PIL`

## 기여 방법
1. 본 레포지토리를 포크합니다.
2. 새로운 브랜치를 생성합니다.
3. 변경 사항을 커밋하고 푸시합니다.
4. Pull Request를 생성하여 기여합니다.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

