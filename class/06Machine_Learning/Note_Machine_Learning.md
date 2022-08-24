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

## 5. Decision Tree
> ### 1) Decision Tree
> * Tree 기반 알고리즘으로, 특정 항목(변수)에 대한 의사 결정(분류) 규칙을 나무의 가지가 뻗는 형태로 분류해 나가는 분석기법
> ### 2) 속성의 불순도
> * 해당 범주에 다양한 class들의 개체들이 얼마나 포함되어 있는가를 의미하는 복잡성으로, 분류 후 얼마나 잘 분류했는지 평가하는 지표
> * 지니 계수   
> <img src="https://user-images.githubusercontent.com/110445149/186391923-27960022-ee63-4278-8aff-3d0b5511de4b.png" height="60" width="200"></img>
> * 엔트로피   
> <img src="https://user-images.githubusercontent.com/110445149/186392052-da292c2f-21dc-4049-be78-dc33ecd8ec47.png" height="60" width="200"></img>
> ### 3) Information Gain(정보 증가량)
> * 지니계수나 엔트로피는 단지 속성의 불순도를 표현
> * 이를 활용하여 어떤 변수가 많은 정복를 제공하는가를 확인하기 위한 지표   
> <img src="https://user-images.githubusercontent.com/110445149/186392128-46838839-9959-4aeb-9d9d-c0e038b48140.png" height="60" width="400"></img>
> * 정보증가량이 가장 높은 속성을 분할 기준으로 삼음
> ### 4) 성능
> * Hyperparameter에 따라 달라짐
>   * max_depth : 클수록 모델이 복잡
>   * min_samples_leaf : 작을수록 모델이 복잡
```
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth = ' ', min_samples_leaf = ' ')
```
> ### 5) 의사 결정 나무 시각화
```
from sklearn.tree import plot_tree

plot_tree(model,                                    #만든 모델 이름
               feature_names = list(x_train),       #Feature 이름, list(x_train)
               class_names = ['class1', 'class2'],  #Target(Class) 이름 
               filled = True);

plt.show()
```
> ### 6) 변수중요도 그래프 그리기 함수
```
def plot_feature_importance(importance, names):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df.reset_index(drop=True, inplace = True)

    plt.figure(figsize=(10,8))
    sns.barplot(x='feature_importance', y='feature_names', data = fi_df)

    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.grid()

    return fi_df

result = plot_feature_importance(model.feature_importances_, list(x_train))
result
```

## 6. Support Vector Machine(SVM)
> ### 0) 전제조건
> * 스케일링
> ### 1) 용어
> * 결정 경계(Decision Boundary)
>   * 클래스를 구분하는 경계선으로, 바로 모델임!
>   * Hyperplane이라고도 함
> * 벡터(Vector)
>   * 모든 데이터 포인트
> * 서포트 벡터(Support Vector)
>   * 결정 경계와 가까운 데이터 포인트(벡터)
>   * 마진의 크기와 결정경계에 영향을 끼침
> * 마진(Margin)
>   * 서포트벡터와 결정경계 사이의 거리
>   * 마진이 클수록 새로운 데이터에 대한 안정적인 분류가 가능해짐
> ### 2) 마진과 오류
> * SVM의 최적화 : 마진은 크게, 오류는 작게
> * 마진의 크기와 오류에 대한 허용 정도는 Trade-off 관계
> * 비용(C : 오류를 허용하지 않으려는 비용)를 조절함으로써 최적의 모델을 만들어감
>   * C가 클수록 오류를 허용하지 않으려는 마진과 결정경계를 찾음(Overfitting)
>   * C가 작을수록 오류를 허용해도 되는 마진과 결정경계를 찾음(Underfitting)
> ### 3) 커널 트릭
> * SVM은 직선 or 초평면으로 분류하는 선형 분류기이지만, 선형적으로 분류할 수 없는 데이터셋이 더 많음
> * 매핑함수
>   * 비선형 데이터 문제를 해결하기 위해 고차원 데이터로 변환하는 함수
>   * But, feature 수가 과도하게 증가하여 연산 시간이 길어짐 
> * 커널 트릭
>   * 실제 고차원 feature를 생성하지 않고, 추가한 것과 같은 효과를 얻음
>   * 종류 : poly / rbf(Radial Basis Function) / sigmoid
> ### 4) 비선형 SVM에서 Hyperparameter
> * Cost(C)
>   * C가 증가할수록 마진의 폭이 줄어들고 오차를 줄이기 위한 복잡한 모델 생성	
> * Gamma
>   * 곡률(반경)의 크기
>   * 값이 클수록 곡률(반경)이 작아지며 복잡한 모델 생성
```
from sklearn.svm import SVC, SVR # SVC : classification & SVR : Regression 

model = SVC(C=' ', kernel=' ', gamma=' ')
```

## 7. 성능 튜닝
> ### 1) 변수 선택법
> * 선형 모델(선형회귀, 로지스틱회귀)의 성능은 변수 선택에 따라 차이가 발생함
> * 전진 선택법(or 후진 소거법)
>   * AIC 값이 가장 작은 모델을 단계별, 순차적으로 탐색
>     1. feature 별로 각각 단순회귀모델을 생성하고 AIC 값을 비교하여 제일 작은 변수 선정
>     2. 과정 1에서 선정된 변수+나머지 변수 하나씩 추가해서 AIC 값이 가장 작은 모델으리 변수 선정(단, 과정 1보다 AIC 값이 더 낮아져야 함)
>     3. 더이상 AIC 값이 낮아지지 않을 때까지 반복
>   * 후진 소거법은 전진 선택법의 반대로 진행
> * AIC(Akaike Information Criterion)
>   * 모델은 train set을 얼마나 잘 설명하지는지가 중요함
>   * 모델이 과적합 되지 않도록, 선형모델에서의 적합도와 feature가 과도하게 늘어나는 것을 방지하도록 설계된 통계량이 AIC이다
>   * AIC 값이 작을수록 좋은 모델 
>   * 변수의 개수(적절한 복잡도) - 모델의 적합도(적절한 성능)
> ### 2) Hyperparameter Tuning
> * Parameter vs Hyperparameter
>   * Parameter : 모델 내부에서 결정되는 변수로, 그 값은 데이터로부터 결정됨
>   * Hyperparameter : 사용자가 직접 세팅해주는 값으로, 모델링할 때 최적하기 위한 파라미터
> * Random Search
> * Grid Search


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
