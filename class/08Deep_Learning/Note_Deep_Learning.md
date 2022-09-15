# Deep Learning  - 9/13~9/16

## 1. Neural Network 구조 이해 및 코드 구현
> ### 0) 라이브러리
> ```
> import tensorflow as tf
> from tensorflow import keras
> ```
> 
> ### 1) Sequential API
> * 선형회귀
> ```
> # 1. 세션 클리어
> keras.backend.clear_session()
> # 2. 레이어 기초 생성
> model=keras.models.Sequential()
> # 3. 레이어 쌓기
> model.add(keras.layers.Input(shape=(n,)))     # n = feature 수
> model.add(keras.layers.Dense(1))              # activation의 default가 linear임
> # 4. 컴파일  -   모델을 학습시키기 위한 학습과정을 설정하는 단계
> model.compile(loss='mse', optimizer='adam')
> # 5. 학습
> model.fit(x, y, epochs=10, verbose=1)
> # 6. 모델 예측
> y_pred = model.predict(x)
> ```
> * 로지스틱 회귀 
> ```
> > # 1. 세션 클리어
> keras.backend.clear_session()
> # 2. 레이어 기초 생성
> model=keras.models.Sequential()
> # 3. 레이어 쌓기
> model.add(keras.layers.Input(shape=(n,)))               # n = feature 수
> model.add(keras.layers.Dense(1, activation='sigmoid'))  # 분류이기에 activation function 추가
> # 4. 컴파일
> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
>
> # 이진분류에서 loss를 'binary_crossentropy'로 두고, 직관적으로 보기 위해 metrics=['accuracy'] 추가
> # threshold를 0.5가 아닌 다른 값으로 설정하려면 
> # metrics=keras.metrics.BinaryAccuracy(threshold='원하는 값')
> 
> # 5. 학습
> model.fit(x, y, epochs=10, verbose=1)
> # 6. 모델 예측
> y_pred = model.predict(x)
> ```
> 
> ### 2) Functional API
> * 선형회귀
> ```
> # 1. 세션 클리어
> keras.backend.clear_session()
> # 2. 레이어끼리 연결
> il=keras.layers.Input(shape=(n,))               # 각 레이어 변수명에 저장
> ol=keras.layers.Dense(1)(il)                    # 앞레이어와 연결
> # 3. input, output layer 지정
> model=keras.models.Model(inputs=il, outputs=ol) # inputs, outputs 각각 지정
> # 4. 컴파일
> model.compile(loss = 'mse', optimizer = 'adam')
> # 5, 학습
> model.fit(x,y, epochs=10, verbose=1)
> # 6. 모델 예측
> y_pred = model.predict(x).reshape(-1)
> ```
> * 로지스틱 회귀
> ```
> # 1. 세션 클리어
> keras.backend.clear_session()
> # 2. 레이어끼리 연결
> il=keras.layers.Input(shape=(n,))
> ol=keras.layers.Dense(1, activation='sigmoid')(il)
> # 3. input, output layer 지정
> model=keras.models.Model(inputs=il, outputs=ol)
> # 4. 컴파일
> model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
> # 5, 학습
> model.fit(x,y, epochs=10, verbose=1)
> # 6. 모델 예측
> y_pred = model.predict(x).reshape(-1)
> ```
> 
> ### 3) 멀티 클래스 분류
> * One-Hot Encoding
> ```
> from tensorflow.keras.utils import to_categorical
> y = to_categorical(y, 3)
> ```
> * activation function을 softmax로 설정
> * loss는 categorical_crossentropy로 설정
>
> |   |선형회귀|로지스틱회귀|멀티클래스 분류|
> |------|---|---|---|
> |output layer activation|default(linear)|sigmoid|softmax|
> |loss|mse|binary_crossentropy|categorical_crossentropy|
>
> ### 4) Artificial Neural Network
> * 기계학습과 인지과학에서 생물학의 신경망에서 영감을 얻은 통계학적 학습 알고리즘
> * 은닉층(hidden layer) 추가
> ```
> keras.layers.Dense(n, activation='relu')
> ```
> * EarlyStopping
>   * 너무 많은 epoch는 overfitting을 일으키지만, 너무 적은 epoch는 underfitting을 일으킴
>   * 이런 상황에서 적절한 epoch에서 학습을 종료시키는데 필요한 것이 EarlyStopping!
> ```
> from tensorflow.keras.callbacks import EarlyStopping
> 
> es=EarlyStopping(monitor='val_loss',       # 학습 조기종료를 위해 관찰하는 대상 ( val_loss(default), val_accuracy 등 )
>                  min_delta=0,              # 개선되고 있는지 확인하기 위한 최소 변화량
>                  patience=5,               # 성능 개선되지 않더라도 바로 종료시키지 않고 몇 번의 epoch를 더 진행하며 기다릴지 결정
>                  verbose=1,
>                  restore_best_weights=True # 학습이 종료됐을 때, 최적의 가중치로 전환
>                  )
> ```

## +. 기타
> * **Loss Fuction**
>   * Logistic Regression loss function
>     * 로지스틱 회귀 비용함수는 볼록함수이므로 경사하강법이 전역 최소값을 찾는 것을 보장함
>     * <img src="https://user-images.githubusercontent.com/110445149/190160525-f70bdd6c-5a5e-4136-b52c-5f72dbb0b08f.JPG" height="50" width="300"></img>    
>   * Cross-entropy loss function
>     * <img src="https://user-images.githubusercontent.com/110445149/190157066-b3f90707-ad88-4024-9127-c17a13253485.JPG" height="50" width="300"></img>    
>     * Cross-entropy
>       * 실제 분포 q에 대해 알지 못하는 상태에서, 모델링을 통해 구한 분포 p를 통해 q를 예측하는 것
>         * <img src="https://user-images.githubusercontent.com/110445149/190157394-d3575290-5dd2-4fd8-8a3f-4fdd6539c3dc.JPG" height="40" width="200"></img>   
> * **Activation Function**
>   * sigmoid
>     * 로지스틱회귀에 사용되는 활성화 함수로, 0 ~ 1 의 값을 출력함
>     * <img src="https://user-images.githubusercontent.com/110445149/190160635-0223b23e-7416-4c4e-9711-71370759d830.JPG" height="35" width="200"></img> 
>   * softmax  
>     * 다중분류에서 사용되는 활성화 함수로, 각 클래스에 속할 확률을 추정함
>     * s(x) : 샘플 x에 대한 각 클래스의 점수를 담으 벡터
>     * <img src="https://user-images.githubusercontent.com/110445149/190160679-3bb4788c-a849-4450-bfa7-2183fb5c4815.JPG" height="50" width="300"></img> 
>   * relu  
>     * z=0에서 미분 가능하지 않음(기울기가 갑자기 변해서 경사하강법이 엉뚱한 곳으로 튈 수 있음)
>     * 그러나 실제로 잘 작동하고 계산속도가 빠르다느 장점
>     * <img src="https://user-images.githubusercontent.com/110445149/190161053-64705a39-e0ea-4c86-bd1a-12e9810dec03.JPG" height="25" width="200"></img> 
>   * tanh
>     * 로지스틱 홤수와 같이 s자 모양이며 연속적이고 미분가능함
>     * 출력값의 범위가 -1 ~ 1 임
>     * <img src="https://user-images.githubusercontent.com/110445149/190162385-ed67f0c7-b255-47e9-af31-e8ba958ce2a1.JPG" height="20" width="200"></img>
> * **경사하강법(Gradient Descent)** 
>   * 개념
>     * 여러 종류의 문제에서 최적의 해법을 찾을 수 있는 일반적인 최적화 알고리즘으로, 비용 함수를 최소화하기 위해 반복해서 파라미터를 조정하는 알고리즘
>     * 비용함수의 현재 gradient(비용함수의 미분값)을 계산하고, gradient가 감소하는 방향으로 최솟값에 수렴할 때까지 점진적으로 진행
>     * 중요 파라미터로는 learning rate!
>   * 배치 경사 하강법(Batch Gradient Descent)
>     * 매 경사 하강법 스텝에서 전체 훈련세트를 사용해 그레디언트를 계산함
>     * 단점 : 훈련세트가 커지면 매우 느려짐
>   * 확률적 경사 하강법(Stochastic Gradient Descent)
>     * 매 스텝에서 한 개의 샘플을 무작위로 선택하고 그 하나의 샘플에 대한 그레디언트를 계산함
>     * 단점
>       * 확률적(무작위)이므로 배치 경사 하강법보다 훨씬 불안정함
>       * 무작위성은 지역 최소값을 벗어날 수 있지만 전역 최소값에 다다르지 못할 수 있음
>     * 해결책 : **학습률을 점진적으로 감소**
>   * 미니배치 경사 하강법(Mini-Batch Gradient Descent)
>     * 임의의 작은 샘플 세트, 즉 미니배치에 대해 gradient를 계산
>     * 행렬 연산에 최적화된 하드웨어, 특히 GPU를 사용해 얻는 성능 향상
> * **배치 학습 vs 미니배치 학습**
>   * 머신러닝 시스템을 분류하는 데 사용하는 기준으로, 입력 데이터의 스트림부터 점진적으로 학습할 수 있는지 여부가 있음
>   * 배치 학습
>     * 가용한 데이털르 모두 사용해 훈련시키는 것으로, 시스템이 점진적으로 학습할 수 없음 
>     * 일반적으로 이 방식은 시간과 자원을 많이 소모하므로 보통 오프라인에서 수행되기에 오프라인 학습이라고도 함
>   * 미니배치 학습
>     * 데이터를 순차적으로 한 개씩 또는 작은 묶음 단위인 미니배치로 주입하여 시스템을 훈련시키는 방식
>     * 매 학습 단계가 빠르고 비용이 적게 들어 시스템은 데이터가 도착하는대로 즉시 학습 가능, 온라인 학습이라고도 함
