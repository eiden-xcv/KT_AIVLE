# Data Preprocessing - 8/8~8/9

## 1. Pandas : 결합 및 집계
> ### 1) 결합
> * pd.merge(left, right, how, on)
>   * 조인(join)하는 방법으로 자동으로 key를 잡아 결합시킴
>   * how = 'inner'(default), 'outer', 'left', 'right'
>   * groupby와 자주 같이 사용됨
> * pd.concat([f1, df2], axis) 
>   * axis=0 : 위아래로 붙임 / axis=1 : 좌우로 붙임
> ### 2) 추가
>   * rolling() & shift() : 주로 시계열 데이터에 대해서 연산 및 행 이동
>     * rolling(n, min_periods=m) : n개씩 묶어서 연산하지만, 최소 m개 이상이어도 연산함

## 2. Pandas : 교차표 & Heatmap
> ### 1) 교차표
> * 범주 대 범주를 비교하고 분석하기 위함
>   * pd.crosstab(row, col, normalize='')
>   * normalize = 'columns' (열!!!) / 'index' (행) / 'all' (전체)
> ### 2) pivot() & heatmap()
> * ex)
```
    tmp1=titanic.groupby(['Embarked', 'Pclass'], as_index=False)['PassengerId'].count()
    pvt=tmp1.pivot('Embarked', 'Pclass', 'PassengerId') # pivot(index, column, value) : 행, 열, 값 순서로!
    sns.heat(pvt, annot=True)
```

## 3. 계열 데이터
> * datetime
>   * pd.to_datetime(df['date'])
>   * 메서드 참고 : df['date'].dt."methods"
> * shift() : 행 이동
> * diff() :  행 간 증감 

## 4. 데이터 준비
> <img src="https://user-images.githubusercontent.com/110445149/186397958-a4b5bf57-60a5-430d-ac05-072203a36dd9.PNG" width="900" height="300"></img>
> ### 1) 변수 확인 및 정리
> * data.dtypes / data.info() / data.describe()
> ### 2) NaN 처리
> * NaN 확인 : df.isnull().sum() & data.loc[data[col].isnull()]
> * 행 제거 : df.dropna(axis=0)
> * 단일 값으로 채움 : df.fillna(value)
> * 이전 값 or 이후 값으로 채움 : df.fillna(method='ffill') or df.fillna(method='bfill')
> * 앞뒤값의 중간값으로 채움 : df.interpolate(method='linear')
> ### 3) Feature Engineering
> * 데이터에 대한 이해를 바탕으로 새로운 feature를 만들어내는 과정
> ### 4) 가변수화 (Dummy Varable)
> * 범주형 변수를 숫자로 만드는 방법
> * 과정
```
    tmp =pd.get_dummies(df['col'], drop_first=T/F)
	  df=pd.concat([df, tmp], axis=1)
	  df.drop('col', axis=1, inplace=True)
```
> ### 5) Data Split
> * sklearn의 데이터 분할 함수 이용
> * 요인, x, feature, 조작변수, 통제변수, 리스크벡터, Input -> 독립변수
> * 결과, y, target, label, Output -> 종속변수
> * ex)
```
      from sklearn.model_selection import train_test_split
      train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)
```
> ### 6) Scaling features
> * 최소값, 최대값을 이용하여 각 feature의 값을  0~1 이 되도록 함
>   * ex)
```
      max_n, min_n = train_x.max(), train_x.min()
      train_x_scale=(train_x - min_n)/(max_n - min_n)
```
```
      from sklearn.preprocessing import MinMaxScaler
      scaler = MinMaxScaler()
      train_x=scaler.fit_transform(train_x)   # 함수 만들고 변환시키는 과정을 합친 함수 이용 [ 1. scaler.fit(train_x) 2. train_x = scaler.transform(train_x) ]
      test_x = scaler.transform(test_x)
```
> * 정규화
>   * ex)
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x2 = scaler.fit_transform(x) 
```
> ### 7) DataFrame to Numpy array
