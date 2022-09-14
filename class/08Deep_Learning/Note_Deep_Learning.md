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
> # 4. 컴파일
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
>   * ```keras.layers.Dense(n, activation='relu')```

## +. 기타
> * Loss Fuction
>   * Logistic Regression loss function
>     * 로지스틱 회귀 비용함수는 볼록함수이므로 경사하강법이 전역 최소값을 찾는 것을 보장함
>     * <img src="https://user-images.githubusercontent.com/110445149/190160525-f70bdd6c-5a5e-4136-b52c-5f72dbb0b08f.JPG" height="50" width="300"></img>    
>   * Cross-entropy loss function
>     * <img src="https://user-images.githubusercontent.com/110445149/190157066-b3f90707-ad88-4024-9127-c17a13253485.JPG" height="50" width="300"></img>    
>     * Cross-entropy
>       * 실제 분포 q에 대해 알지 못하는 상태에서, 모델링을 통해 구한 분포 p를 통해 q를 예측하는 것
>         * <img src="https://user-images.githubusercontent.com/110445149/190157394-d3575290-5dd2-4fd8-8a3f-4fdd6539c3dc.JPG" height="40" width="200"></img>   
> * Activation Function
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


