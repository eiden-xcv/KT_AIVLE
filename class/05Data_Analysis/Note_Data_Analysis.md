# Data Analysis - 8/10~8/12

* **EDA 1단계 단변량 분석 - matplotlib / 분석 프로세스 EDA & CDA / 단변량분석 : 숫자형 & 범주형 변수**
* **EDA 2단계 이변량 분석 - seaborn / 이변량 분석 : 숫자->숫자 & 범주->숫자 & 범주->범주  & 숫자->범주**

## 0. CRISP-DM ★★
<img src="https://user-images.githubusercontent.com/110445149/185815282-dd1ed38a-60bd-4ba9-b214-b27dcc43a3cb.PNG" width="200" height="200"></img>
> ### 1) Business Understanding
> * 비즈니스 문제 정의
> * 데이터분석 방향 및 목표 결정
> * 초기 가설 수립(X -> Y)
> ### 2) Data Understanding 
> * 원본식별 및 취득 : 가설에서 도출된 데이터의 원본을 확인  
>   * 있는 데이터 : 그대로 사용 가능 & 가공해야 사용가능
>   * 없는 데이터 : 취득 가능 & 취득 불가능
> * 데이터 탐색하는 두 가지 방법 : 데이터 통계량 & 데이터 시각화
>   * EDA(Exploratory Data Analysis) : 개별 데이터의 분포, 가설이 맞는지 파악 & NA, 이상치 파악
>   * CDA(Confirmatory Data Analysis) : 탐색으로 애매한 정보는 통계적 분석 도구(가설 검정) 사용
>     * 개별 변수 분포 (단변량분석)
>     * X -> Y확인 (이변량분석)
>     * 전처리전략 수립 
>       * why? : 모든 셀은 숫자인 값을 가져야 함 (+ 필요에 따라 값의 범위를 일치)
>       * NaN 처리 (삭제 : 행/열 or 채우기 : 비즈니스 관점으로 결정)
>       * 가변수화
>       * 스케일링
>       * feature engineering 
> ### 3) Data preparation
> * 모델링을 위한 데이터 구조 만들기
>   * 추가변수 도출 / 결측치 조치 / 가변수화 / 스케일링 / 데이터 분할
> ### 4) Modeling
> * 학습
> * 검증 : 예측 및 평가
> ### 5) Evaluation
> ### 6) Deployment

## 1. matploltlib 
> ### 1) 데이터의 시각화 (그래프 & 통계량)
> * 우리가 다루는 데이터는 "비즈니스"를 담고 있다.
> * 한 눈에 파악해야할 정보는 "비즈니스 인사이트"이다
> * `import matplotlib.pyplot as plt`

## 2. 단변량 분석 :  EDA & CDA
> | |기초통계량|그래프|
> |:---:|:---:|:---:|
> |숫자형|mean|히스토그램|
> | |mode|밀도함수그래프|
> | |4분위수|Boxplot|
> |범주형|범주별 빈도수|Barplot|
> | |범주별 비율|Piechar|
> ### 1) 단변량분석_숫자형변수
> * 수치화
>   * 대푯값( 평균 / 중앙값 / 최빈값 / 4분위수 )
>   * 기초통계량 ( series.describe() / df.describe() )
> * 시각화 
>   * 히스토그램 : plt.hist()
>		* 밀도함수그래프 : sns.kdeplot()
>		* Boxplot : plt.boxplot()
>		* 시계열 데이터의 시각화
> ### 2) 단변량분석_범주형변수
> * 기초통계량
> * 시각화
>   * Bar chart : plt.bar() / plt.hbar() / sns.countplot()
>   * Pie chart : plt.pie()
> ### 3) 단변량분석 시 유의할 점
> * 값의 범위 확인
> * 데이터가 모여 있는 구간(범주)와 희박한 구간(범주) 확인
> * 이상치(Outlier) 확인
> * 결측치(NaN) 확인 및 조치 방안
> * 가변수화, 스케일링 대상 선별

## 3. seaborn 라이브러리
> * seaborn은 Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
> * histplot / kdeplot / boxplot
> * distplot : histogram + density plot -> `sns.distplot(데이터, bins , hist_kws=dict())`
> * jointplot : scatter + histogram(density plot) -> `sns.jointplot(x, y, data, hue)`
> * pairplot : scatter + histogram(density plot) 확장, 모든 숫자형 변수들에 대해서 서로 비교하는 산점도 표시 -> `sns.pairplot(data, hue)`
> * countplot : 집계 + bar plot -> `sns.countplot(x, data, hue)`
> * barplot : 평균비교 bar plot + error bar -> `sns.barplot(x, y, data)`
> * heatmap :  두 범주 집계 시각화, 집계(groupby)와 피봇(pivot)을 먼저 만들어줘야함.

## 4. 이변량 분석 :  EDA & CDA
> | | | | Y | | |
> |:---:|:---:|:---:|:---:|:---: |:---: |
> | | | **숫자** | | **범주** | |
> | | **숫자**| Scatter(산점도)  | 상관분석(상관계수: p-value) | Histogram | (대체)로지스틱 회귀 |
> | | | |  | Density plot | |
> | **X** | **범주** | 평균비교(barplot) | T 검정 | 100% stacked barchart(plot.bar) | 카이제곱검정 |
> | | |  | 분산분석(ANOVA) | | |
> | | | *시각화* | *수치화* | *시각화* | *수치화* |
> 1) 두 변수의 관계 분석하기 : 숫자 -> 숫자
- 시각화 : Scatter(산점도)
	- 숫자vs숫자를 비교할 때 중요한 관점은 "직선"이다.
	- plt.scatter( x, y )
- 수치화 : 상관계수와 상관분석
	- import scipy.stats as spst / spst.pearsonr( X, Y )
	- df.corr()
	- 한계점 존재!!!
- 가설과 가설 검정 이야기
	- 귀무가설 H1(차이가 없다) vs 대립가설 H0(차이가 있다)
	- 대립가설이 맞다고 받아들일 때, 틀릴확률 = p-vlalue, 유의확률
	- 검정 통계량 : t 통계량(T-test) / 카이제곱 통계량 / f 통계량(ANOVA)
	- 이러한 절차를 가설 검정이라 한다.

2) 두 변수의 관계 분석하기 : 범주 -> 숫자
- 범주 -> 숫자 관계를 살펴볼 때 중요한 관점은 "평균비교"이다.
- 표준편차 & 표준오차
	- 표준오차는 모평균과 표본평균의 오차를 추정한 것으로, 공식은 s/sqrt(n) 이다.
	- 이 표준오차로분터 신뢰구간을 계산한다.
	- 표집을 여러차례하여 얻은 표본들로 부터 얻은 표본평균의 분포(표집분포)를 살펴보면 정규분포가 된다. By 중심극한 정리 - ☆
	- 이를 통해 95%의 신뢰구간, 모집단의 평균이 속하리라 간주되는 값들의 범위를 계산할 수 있다.
- 시각화 : 평균비교 barplot(sns.barplot)
	- sns.barplot(x, y, data)
- 수치화 
	- 범주가 2개인 경우 : t-검정(t-test)
		- 두 평균의 차이를 표준오차로 나눈 값으로, 보통 -2미만 혹은 2이상인 경우 차이가 있다고 봄
		- spst.ttest_ind(B, A, equal_var=T/F)

	- 범주가 3개 이상인 경우 : 분산분석(: ANOVA / f 통계량)
		- 여러 집단 간의 차이를 비교하는 방법으로, 기준은 "전체 평균"이다
		- f 통계량 = (집단간 분산 / 집단내 분산)으로 대략 2~3이사이면 차이가 있다고 판단
		- 이 값이 의미하는 것은 전체 평균대비 각 그룹간 차이가 있는지 여부만 알려줌		
		- spst.f_oneway(A, B, C)
		
3) 두 변수의 관계 분석하기 : 범주 -> 범주
- 시각화 
	-100% stacked bar chart(plot.bar)
		- 우선 교차표(crosstab)으로 집계
		- 비율만 비교하므로 양에 대한 비교는 할 수 없다
		- "pd.crosstab".plot.bar()
	- mosaic 
		- mosaic(data, ["col", "col"])
- 수치화
	- 카이제곱검정
		- "기대빈도로 부터 차이"가 중요한 관점이다
		- 우선 교차표(crosstab)으로 집계
		- 범주형 변수들 사이에 어떤 관계가 있는지, 수치화 하는 방법
		- 보통 자유도((각 범주의수 -1)의 곱)의 2~3배보다 크면 차이가 있다고 봄
4) 두 변수의 관계 분석하기 : 숫자 -> 범주
- 시각화	
	- "범주별로 비교"
	- Histogram을 범주별로 겹쳐 그리기
		- multiple 옵션
	- Density plot을 범주별로 겹쳐 그린 후 비교
		- multiple 옵션
- 로지스틱 회귀 모형으로 대체
	- 숫자 -> 범주 분석을 위한 가설 검정 도구가 없다.
	- 그래서 로지스틱 회귀 모델로 부터 p-value를 구해보는 것으로 대체
