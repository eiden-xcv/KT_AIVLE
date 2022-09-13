# Model Evaluation  - 9/5~9/6

## 1. 모델 해석하기
> ### 1) 비즈니스를 위한 인공지능
>
> ### 2) 모델 성명에 대한 요구
> * Interpretability vs Explainability
>   * Interpretability(해석) : Input에 대해 모델이 왜 그런 Output을 예측했는지를 말함
>     * Whitebox model : 본질적으로 해석가능한 모델
>   * Explainability(설명) : Interpretability를 포함하며 추가적으로 투명성에 대한 요구
>     * Model Transparency : 모델이 어떻게 학습되는지 단계별로 설명 가능해야함
>  * 혼용해서 사용함
>     
>  * Whitebox vs Blackbox 
>  
>  * Interpretability-Accuracy Trade-off
>   * 설명이 잘되는 알고리즘은 대체로 성능이 낮다  
>   
> ### 3) 모델에 대한 설명
> | 구분 | |  |
> |------|-----|-----|
> |전체데이터 |모델 전체에서 어떤 feature가 중요할까|Feature Importance - Tree Based Model & Permutation Feature Importance |
> | |특정 feature 값의 변화에 따라 예측값은 어덯게 달라질까|Partial Dependence Plot(PDP)|
> |개별데이터|이 데이터(분석단위)는 왜 그러한 결과로 예측되었을까|Shapley Additive Explanation(SHAP)|
> * [1] 변수 중요도(Feature Importance)
>   * 알고리즘 별 내부 규칙에 의해, 예측에 대한 변수 별 영향도 측정 값
>   * 성능이 낮은 모델에서의 변수중요도는 의미 없음
>   * [1-1] 모델에서 Feature Importance를 제공하는 알고리즘 : Tree 기반 모델
>     * Decision Tree / Random Forest / XGB 등
>       * Decision Tree & Random Forest
>         * MDI(Mean Decrease Impurity)
>           * Imformation Gain : 지니 불순도가 감소하는 정도
>           * Tree 전체에서 feature별로 imformation gain의 평균을 계산
>       * Random Forest에서는 각 Tree의 feature importance의 평균
>     * XGB
>       * 변수중요도를 계산하는 3가지 방법
>         * 1. weight
>           * 모델 전체에서 해당 feature가 split 될대 사용된 횟수의 합
>           * plot_importance에서의 default
>         * 2. gain
>           * feature별 평균 imformation gain
>           * model.feature_importances_의 default
>           * cf) total_gain : feature별 importance gain의 총 합
>         * 3. cover
>           * feature가 split할 때 샘플 수의 평균
>           * cf) total_cover : 샘플 수의 총 합
>   * [1-2] Permutation Feature Importance
>     * 알고리즘과 상관없이 변수 중요도 파악하는 방법
>       * Feature 하나이 데이터를 무작위로 섞였을 때, model의 score가 얼마나 감소되는지로 계산
>     * 단점 : 다중공성선이 있는 변수가 존재할 때, 특정 변수가 하나 섞이고 관련 변수는 그대로 있으기에 score가 크게 줄어들지 않을 수 있음
>     * ```permutation_importance(model, x_val, y_val, n_repeats=10, scoring='r2')```
> * [2] PDP(Partail Dependence Plot)
>   * 관심 feature의 값이 변할 때, 모델에 미치는 영향을 시각화
>   * ```plot_partial_dependence(model, feature=[], X=x_train, kind='both')```
> * [3] SHAP(SHapley Additie exPlanations)
>   * Shapley Value : 모든 가능한 조합에서 하나의 feature에 대한 평균 기여도로, 특정 변수가 예측력에 얼마나 기여하는지 파악하기 위해 이 특정 변수와 관련된 모든 변수 조합들을 입력시켰을 때 나온 결과값과 비교를 하면서 변수의 평균 기여도를 계산한 값
>   * TreeExplainer / DeepExplainer / KernelExplainer / Explainer
```
import shap

explainer = shap.TreeExplainer(model)
shap_values1 = explainer.shap_values(x_train)
```

## 2. 모델 평가하기
> * 평가 전 질문
>   * 어떤 문제를 해결하기 위한 모델인가?
>   * 실제 목적에 맞게 모델의 결과를 평가하고 있는가?
>   * 모델의 예측 결과에 대한 비즈니스 프로세스는 어떻게 정의되어 있는가?
> * **비즈니스 가치로 평가!!!**
> ### 1) Classification
> * ML Metric
>   * Confusion Matrix / accuracy 
> ### 2) Regresson
> * ML Metric
>   * SSE / MSE / RMSE/ MAE / MAPE
> * 회귀모델에 대한 비즈니스 평가는 일반화가 쉽지 않음...
> * 예측결과를 가지고 어떤 action이 있는지 파악하고, 그에 맞는 평가체계를 만들어야함
> ### 3) 비지도학습
> *
