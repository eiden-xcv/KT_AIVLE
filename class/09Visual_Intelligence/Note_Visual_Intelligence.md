# Visual Intellgence - 9/19~9/23

## 1. Review
> ### 1) Batch Normalization
> * Internal Covariate Shift
>   * Covariate(공변량) : 독립변수 이외에 종속변수에 영향을 줄 수 있는 잡음인자(변수)
>   * Covariate Shift : train data와 test data의 distribution이 다른 현상
>   * Internal Covariate Shift : 가중치를 학습 시 이전 레이어의 값에 따라 다음 레이어의 가중치가 영향을 받는다. 이때 규제없이 가중치를 학습하다보면 매개변수 값의 범위가 넓어질 수 있는데, 이렇게 매개변수의 값들의 변동이 심해지는 현상
> * 2015년 Sergey loffe & Christian Szegedy의 논문 Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift에서 그레디언트 소실과 폭주 문제를 해결하기 위한 배치 정규화(Batch Normalization)기법 제안
> * 단순히 입력을 원점에 맞추고 정규화한 다음, 각 층에서 두개의 새로운 파라미터로 결과값의 스케일을 조정하고 이동시킴
> * 배치 정규화 알고리즘
>   * <img src="https://user-images.githubusercontent.com/110445149/191153342-925dba2e-0f31-43d2-9d21-638287896d9d.JPG" height="250" width="300"></img>
>   * 미니배치B에 대해 평가한 입력의 평균벡터, 표준편차벡터
>   * 3번은 평균이 0이고 정규화된 샘플i의 입력
>   * gamma는 층의 출력 스케일 파라미터 벡터, beta는 층의 출력 이동 파라미터 벡터
>   * 4번은 배치 정규화 연산의 출력, 즉 입력의 스케일을 조정하고 이동시킨 것
> * 장점 : 배치 정규화는 규제와 같은 역할을 하여 다른 규제 기법의 필요성을 줄여줌
>   * 배치 정규화는 전체 데이터셋이 아니고 미니배치마다 평균과 표준편차를 계산하므로 훈련 데이터에 일종의 잡음을 넣는 것으로 볼 수 있고, 이 잡음은 훈련 세트에 과대적합되는 것을 방지하는 규제의 효과를 가짐
> * 단점 : 모델의 복잡도를 키우며, 층마다 추가되는 계산이 신경망의 예측을 느리게 하기에 실행 시간 면에서도 손해

> ### 2) 규제를 통한 과적합 방지
> * L1 규제 & L2 규제
>   *
>    ``` 
>   layer = keras.layers.Dense(256, activation='elu', kernel_initailizer='he_normal', 
>                                 kernel_regularizer=keras.regularizer.l2(0.01))
>   ```
>   * keras.regularizer.l1(), keras.regularizer.l2(), keras.regularizer.l1_l2()
> * Dropout
>   * 매 훈련 스텝에서 각 뉴런은 임시적으로 드롭아웃될 학률 p를 가지는데, 이번 훈련 스텝에는 완전히 무시되지만 다음 스텝에서는 활성화 될 수 있음
>   * 각 훈련 스텝에서 고유한 네트워크가 생성된다고 생각할 수 있으며, 결과적으로 만들어진 신경망은 이 모든 신경망을 평균한 앙상블로 볼 수 있음
>   * Dropout은 훈련하는 동안에만 활성화 됨
>   * ``` layer=keras.layers.Dropout(rate=0.2) ```
> * Max-Norm 규제
>   *
>    ``` 
>   layer = keras.layers.Dense(256, activation='elu', kernel_initailizer='he_normal', 
>                                 kernel_constraint=keras.constraints.max_nomr(1.))
>   ```

## 2. CNN Basics
> ### 1) Convolution Layer
> * 입력이미지에 필터를 적용하여 새로운 feature map을 만드는 층
> * filter 수, 크기, stride, padding 설정
> ### 2) Pooling Layer
> * 계산량과 메모리 사용량, 과적합 위험을 줄여주기 위한 파라미터 수를 줄이기 위해 입력이미지의 부표본을 만드는 층
> * MaxPool2D, AvgPool2D
