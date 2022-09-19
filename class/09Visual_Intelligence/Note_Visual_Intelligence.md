# Visual Intellgence - 9/19~9/23

## 1. Review
> ### 1) Batch Normalization
> * 2015년 Sergey loffe & Christian Szegedy의 논문 Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift에서 그레디언트 소실과 폭주 문제를 해결하기 위한 배치 정규화(Batch Normalization)기법 제안
> * 단순히 입력을 원점에 맞추고 정규화한 다음, 각 층에서 두개의 새로운 파라미터로 결과값의 스케일을 조정하고 이동시킴
> * 배치 정규화 알고리즘
>   *
> * 장점 : 배치 정규화는 규제와 같은 역할을 하여 다른 규제 기법의 필요성을 줄여줌
>   * 배치 정규화는 전체 데이터셋이 아니고 미니배치마다 평균과 표준편차를 계산하므로 훈련 데이터에 일종의 잡음을 넣는 것으로 볼 수 있고, 이 잡음은 훈련 세트에 과대적합되는 것을 방지하는 규제의 효과를 가짐
> * 단점 : 모델의 복잡도를 키우며, 층마다 추가되는 계산이 신경망의 예측을 느리게 하기에 실행 시간 면에서도 손해
> * Internal Covariate Shift
>   * 
> ### 2) 규제를 통한 과적합 방지
> * L1 규제 & L2 규제
>   * ``` layer = keras.layers.Dense(256, activation='elu', kernel_initailizer='he_normal', kernel_regularizer=keras.regularizer.l2(0.01)```
>   * keras.regularizer.l1(), keras.regularizer.l2(), keras.regularizer.l1_l2()
> * Dropout
>   * 매 훈련 스텝에서 각 뉴런은 임시적으로 드롭아웃될 학률 p를 가지는데, 이번 훈련 스텝에는 완전히 무시되지만 다음 스텝에서는 활성화 될 수 있음
>   * 각 훈련 스텝에서 고유한 네트워크가 생성된다고 생각할 수 있으며, 결과적으로 만들어진 신경망은 이 모든 신경망을 평균한 앙상블로 볼 수 있음
>   * Dropout은 훈련하는 동안에만 활성화 됨
>   * ``` layer=keras.layers.Dropout(rate=0.2) ```
> * Max-Norm 규제
> * 과적합 방지
> * 451

## 2. CNN Basics
> ### 1) Convolution Layer
> *
> ### 2) Pooling Layer
