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
1 - (SSE/SST)
```
> * 오차의 양과 비율로 평가하기
<img src="https://user-images.githubusercontent.com/110445149/186132150-483ea03b-b8e3-4ff4-8dd1-16080e0fd572.png" height="300" width="500"></img>
```
#R2 SCORE
r2_score(y_val, pred)
#MSE
mean_squared_error(y_val, pred) 
#RMSE
mean_squared_error(y_val, pred, squared = False)
#MAE
mean_absolute_error(y_val, pred)
# MAPE
mean_absolute_percentage_error(y_val, pred)
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
