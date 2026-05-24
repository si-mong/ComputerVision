
# HW3: Deep Learning Analysis: Loss, Activation, and Optimization

이 레포지토리는 딥러닝 다층 퍼셉트론(MLP) 구조에서 **손실 함수(Loss Function), 활성화 함수(Activation Function), 최적화 알고리즘(Optimizer)**이 모델의 학습 속도, 수렴 안정성, 최종 성능에 미치는 영향을 정량적·시각적으로 분석한 실험 코드를 포함하고 있습니다.

##  프로젝트 구조

- **`실험A.ipynb`**: CrossEntropy Loss와 MSE Loss(with Softmax)의 수렴 속도 및 기울기 흐름 분석 (데이터셋: Fashion-MNIST)
- **`실험B.ipynb`**: ReLU, LeakyReLU, Sigmoid의 특성 분석, Dead ReLU 현상 유도 및 시각화 (데이터셋: make_moons)
- **`실험C.ipynb`**: SGD, SGD+Momentum, Adam의 성능 및 학습률(LR) 변화, Exponential Decay 효과 분석 (데이터셋: Fashion-MNIST)

## 주요 실험 내용 및 결과

### 1) [실험 A] 손실 함수 비교 (CrossEntropy vs MSE)
- **목표:** 분류 문제에서 두 손실 함수의 수렴 속도 및 안정성 비교
- **핵심 분석:** MSE 사용 시 출력층 포화 영역에서 발생하는 Gradient Vanishing 문제와 CrossEntropy가 오차에 비례해 강력한 역전파 신호를 주는 기전 증명 (Gradient Flow 시각화 완료)

### 2) [실험 B] 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)
- **목표:** 가중치 초기화를 의도적으로 작게 설정(`std=0.01`)하여 Dead ReLU 현상 유도 및 분석
- **핵심 분석:** - ReLU 사용 시 약 36.5%의 Dead 뉴런 발생 확인 및 LeakyReLU를 통한 완화(0%) 정량 측정 및 히트맵 시각화
  - Sigmoid 사용 시 입력층 방향(dense1)으로 갈수록 극심해지는 기울기 소실 현상 추적

### 3) [실험 C] 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)
- **목표:** 세 가지 Optimizer의 학습 곡선 및 학습률(0.1, 0.01, 0.001)에 따른 Overshooting/정체 분석
- **핵심 분석:** 적응형 학습률(Adaptive LR)을 적용한 Adam의 초기 수렴 우수성 및 후반부 Exponential Decay 스케줄러를 통한 학습 안정성 개선 검증 (`@tf.function`을 통한 런타임 최적화 적용)

## 개발 환경 및 활용 기술
- **Framework:** TensorFlow 2.x / Keras
- **Environment:** Google Colab (T4 GPU 환경 최적화)
- **Dataset:** Fashion-MNIST (실험 A, C), Scikit-Learn `make_moons` (실험 B)
- **Visualization:** Matplotlib / Seaborn (Gradient 히스토그램, Dead ReLU 히트맵, Decision Boundary 시각화)