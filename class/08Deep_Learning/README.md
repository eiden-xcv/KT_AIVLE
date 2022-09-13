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
> # 분류에서 loss를 'binary_crossentropy'로 두고, 직관적으로 보기 위해 metrics=['accuracy'] 추가
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
