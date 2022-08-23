# Machine Learning  - 8/22~8/29

## 1. 모델링 개요
> ### 1) 모델 및 모델링
> * 모델 : 데이터로부터 패턴(규칙, 반복, 즉 x와 y의 관계)을 찾아 수학식으로 정리해 놓은 것
> * 모델링 :오차가 적은 모델을 만드는 과정
> * 모델의 목적 : 표본을 가지고 모집단을 추정하기 위해
> * 실제값 = 모델(Signal) + _**오차(Noise)**_
>   * 오차 : 모델이 실제값으로부터 벗어난 정도
> * 패턴을 찾는 방법	
>   * 지도학습(Supervised Learning) : 답을 알려주면서 패턴을 찾는 학습
>       * Regression : 수치형 답을 찾음
>       * Classification : 범주형 답을 찾음
>   * 비지도학습(Unsupervised Learninig) : 스스로 비슷한 데이터를 찾아 패턴을 찾는 학습

## 2. Linear Regression
> ### 0) 전제조건
> *  NaN 조치, 가변수화, feature들간의 독립성을 충족해야함
> ### 1) Linear Regression : 단순 회귀
> * 단순회귀 : 하나의 예측변수로 하나의 결과변수를 예측
> * 데이터를 하나의 직선으로 요약
>   * 자료를 설명하는 직선은 여러 개가 될 수 있으며, 이 중에서 가장 잘 설명하는, 즉 전체 오차가 작은 직선을 선정하는 방법
>     * 최적화 : 오차를 조금씩 줄여가며 반복적으로 직선을 찾는 방법
>     * 해석적(계산적) 방법 : 최소 제곱법   
<img src="https://user-images.githubusercontent.com/110445149/186127415-cd3b6cbb-7aaa-43a0-b7bc-eec39141814f.png" height="50" width="300"></img>
> ### 2) Linear Regression : 다중 회귀
> * 다중회귀 : 복수의 예측변수로 하나의 결과변수를 예측
>   * feature들 간에 독립성을 가정하고 모델을 생성!!
> * 모델링 절차
>   1. 필요한 함수 불러오기
>   2. 모델 선언(설계) - 하이퍼파라미터 설정
>   3. 학습(모델링)
>   4. 검증 : 예측 및 평가
> * 모델 내부 확인
>   * 회귀계수 확인 : model.coef_
>   * 절편 확인 : model1.intercept_
```
path = 'url'
data = pd.read_csv(path)

target = 'col'
x = data.drop(target, axis=1) 
y = data.loc[:, target] 

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_val)

print(model.coef_, model.intercept_)
```
> ### 3) 회귀 모델 평가
> * 회귀모델은 두가지 관점으로 평가함
>   * [y의 평균] 모델의 오차 vs [회귀] 모델의 오차
>     * R2-SCORE : 오차의 비(평균모델 오차 대비 회귀모델이 해결한 비율) 설명력 결정계수
>   *	[실제값 y] vs [회귀모델의 예측값 y-hat]
>     * MSE / RMSE / MAE : 오차의 크기 
>     * MAPE : 오차율
> * 평균모델의 오차와 회귀모델의 오차   
<img src="https://user-images.githubusercontent.com/110445149/186127764-da76e255-257b-4972-9c4f-e72a5dc2929b.PNG" height="200" width="500"></img>
>   * 평균 모델과 실제 값과의 차이(SST) : baseline 오차
>   * 평균 모델과 회귀모델과의 차이(SSR)
>   * 실제 값과 회귀모델과의 차이(SSE) : 잔차, 남은오차(residual)
> * R2-Score
>   * R2 = 1- SSE/SST
>   * 평균모델의 오차 대비 회귀모델이 해결한 오차의 비율
>   * 회귀모델이 얼마나 오차를 해결(설명)했는지를 나타냄
>   * 결정계수, _설명력_ 이라고 함
```
SST= np.sum(np.power(y_val - y_val.mean(), 2))
SSE = np.sum(np.power(y_val - pred, 2))
# SSR=np.sum(np.power(pred-y_val.mean(),2))
# R2 Score = 1- SSE/SST
r2_score_1 = 1 - (SSE/SST)
#R2 SCORE
r2_score_2 = r2_score(y_val, pred)
```
> * 오차의 양과 비율로 평가하기
<img src="https://user-images.githubusercontent.com/110445149/186132150-483ea03b-b8e3-4ff4-8dd1-16080e0fd572.png" height="300" width="500"></img>
```
#MSE
mse = mean_squared_error(y_val, pred) 
#RMSE
rmse = mean_squared_error(y_val, pred, squared = False)
#MAE
mae = mean_absolute_error(y_val, pred)
# MAPE
mape = mean_absolute_percentage_error(y_val, pred)
```
> ### 4) 참고
> * 공선성(Colinearity)
>   * 하나의 독립변수가 다른 하나의 독립변수로 잘 예측되는 경우, 또는 서로  상관이 높은 경우
> * 다중 공선성(Multi-Collinearity)
>   * 하나의 독립변수가 다른 여러개의 독립변수들로 잘 예측되는 경우
>   * 다중 공선성이 있으면 계수 추정이 잘 되지 않거나 불안정해져서 데이터가 약간만 바뀌어도 추정치가 크게 달라짐
>   * 즉, 계수가 통계적으로 유의미하지 않은 것처럼 나올 수 있음
>   * 분상팽창지수(VIF, Variance Inflation Factor)
>     * VIF = 1 / (1-R^2)
>     * 5 이상이면 다중 공선성이 존재, 10 이상이면 강한 다중 공선성이 있다고 봄
>     * 그러나 다중 공선성이 항상 성능에 문제가 되는 것은 아닐 수 있음

## 3. K-Nearest Neighbors(KNN)
> ### 1) 원리
> 1. 예측해야할 데이터(x_val)와 주어진 데이터(x_train)의 모든 거리를 계산
> 2. 가까운 거리의 데이터를 k개 찾기
> 3. k개 값(y_train)의 평군을 계산하여 예측
> ### 2) 장단점
> * 장점
>   * 데이터의 분포 형태와 상관이 없음
>   * 설명변수의 개수가 많아도 무리없이 사용 가능
> * 단점
>   * 계산 시간이 오래걸림
>   * 훈련데이터를 모델에 함께 저장
>   * 해석하기 어려움
> ### 3) Scaling - KNN을 위한 전처리
>	* Normalization : MinMaxScaler()
>   * X=(x-a)/(b-a)
> * Standardization : StandardScaler()
>   * X=(x-mean)/std
> ### 4) 성능
> * Hyperparameter, 복잡도 결정 요인
>   *	k값이 클수록 단순한 모델, 작을수록 복잡한 모델
>   *	보통 k의 값은 데이터 건수의 제곱근 근처로 잡음
>   *	거리계산법에 의해서도 성능이 달라짐(Euclidean vs Manhattan)

## 4. Logistic Regression
> ### 1) 로지스틱 함수(Sigmoid fucntion)
> * 선형 판별식을 찾고(선형회귀분석과 동일), 선형 판별식으로부터의 거리를 (0, 1)로 변환
> * p(x)=1/(1+e^(-f(x))) 	# f(x)는 선형 판별식
> * 실제 f(x)의 범위는 (-inf, inf)이지만 실제 Y는 0과 1이기에 값을 (-inf, inf) => (0, 1)로 변환할 필요가 있다.
>   * (-inf, inf) => (0, inf) => (0, 1) 
>     * 아래의 과정을 반대로 생각하면 됨
>     * Odds Ratio(승산) : 사건이 일어날 가능성 대 사건이 일어나지 않을 가능성의 비
>     * Log Odds(로그 승산) : (0, inf)에 로그를 취하면 (-inf, inf)로 변환가능
> ### 2) 분류모델 평가
> * Confusion Matrix   
<img src="https://user-images.githubusercontent.com/110445149/186136389-1c19d941-c86a-4a86-be45-134875119b1b.PNG" height="300" width="400"></img>
> ###### 출처 : https://en.wikipedia.org/wiki/Confusion_matrix
> * 성능지표
>   * 전체관점
>     * Accuracy(정분류율, 정확도)
>       * 전체 중에 맞춘 비율
>       * (TP + TN) / Total   
>   * 특정 class 관점
>     * Precision(정밀도)
>       * 해당 class라고 예측한 것 중에 맞춘 비율
>       * TP / ( TP + FP )
>     * Recall(재현율) / Sensitivity(민감도)
>       * 실제 해당 class 중 맞춘 비율
>       * TP / ( TP + FN )
>     * F1-score
>       * Precision과 Recall의 조화평균
>       * (2 * Precision * Recall) / (Precision + Recall)
> * Cut-off에 따른 모델의 성능 변화 그래프
>   * Precision-Recall Curve
>   * AUC - ROC Curve
>     * ROC(Receiver Operating Characteristic) : 모든 임계깞에서 분류 모델의 성능을 보여주는 그래프
>     * AUC(Area Under Curve) : ROC Curve 아래 영역   
>     * x축은 FPR(False Positive Rate), y축은 TPR(True Positive Rate)
<img src="https://user-images.githubusercontent.com/110445149/186140929-03a5dc17-213a-44e3-902a-32857ede5de4.PNG" height="300" width="700"></img>


## 참고
> ### 1. Data Understading & Preparation
> * 과정
>   1. 데이터 수집*
>   2. test 데이터 분할*
>   3. EDA & CDA
>   4. 불필요한 변수 제거
>   5. NaN 조치 : 행 drop**
>   6. X, y 데이터 분할**
>   7. feature engineering
>   8. 가변수화
>   9. train & val 데이터 분할***
>   10. NaN 조치 : 추정해서 채우기
>   11. 스케일링***
> ### 2. 회귀와 분류
> | | 선형회귀 | KNN | 로지스틱 | DT | SVM |
> |:---:|:---:|:---:|:---:|:---:|:---:|
> | 회귀 | O | O | X | O | O |
> | 분류 | X | O | O(이진분류) | O  | O |
