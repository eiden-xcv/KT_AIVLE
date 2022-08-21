# Python 라이브러리 활용 : Numpy & Pandas - 8/1~8/2

## 0. 분석을 위한 데이터 구조
> * 데이터 분석 큰 그림 : CRISP-DM(Cross-Industry Standard Process for Data Mining)
> <img src="https://user-images.githubusercontent.com/110445149/185781416-4e07c61e-de9c-4b12-8180-891d0d6f572f.PNG" width="200" height="200"></img>   
> * 분석 대상 : 수치형 & 범주형 데이터 -> 2차원 구조(행렬)
>   - 행 : 분석단위
>   - 열 : 변수(Target, Feature)   
> -> Numpy & Pandas

## 1. Numpy ☆
> ### 1) 개요
>   * 리스트의 한계 -> Numpy(Numerical Python)
> ### 2) 배열 만들기
>   * a = np.array([1, 2, 3, 4])   
>     - type(a) / a.ndim / a.shape / a.dtype / a.reshape()
>   * np.zeros() / np.ones() / np.full() / np.eye() / np.random.random()
>   * np.mean() / a.mean()
> ### 3) 배열 데이터 조회
>  * 불리안 방식 배열 인덱싱 : a[조건]
>  * 정수 방식 배열 인덱싱
>  * 배열 인덱싱과 슬라이싱
>  ### 4) 배열 변환과 연산
>  * 기본연산(사칙연사)
>  * 배열 내 집계함수 : 옵션 없으면 전체, axis=0이면 열기준 axis=1이면 행기준
>   - np.sum() / np.mean() / np.std()
>  * np.where(조건문, 참일 때 값, 거짓일 때 값)
>  * np.argmax() / np.argmin()

## 2. Pandas ★
> ### 1) 개요
> * DataFrame(2차원)
>   - 열 : 정보(변수) == Series(1차원)
>   - 행 : 분석단위(관측치, 샘플)
>  * pd.DataFramme() 함수를 통해 만듦
>  * csv 파일에서 데이터 읽기 : pd.read_csv(path or url)
>  * csv 파일로 만들기 : pd.to_csv(path)
>  ### 2) 데이터프레임 정보 확인
>  * 데이터 탐색
>   - df.head() / df.tail() / df.shape / df.index / df.values / df.columns : 열 이름 / df.dtypes : 열 데이터 타입
>   - df.info() : 인덱스, 열, 값 개수, 데이터 형식 정보 등 확인 / df.describe() : 기초 통계 정보 확인
>  * 데이터 정렬
>   - df.sort_index(ascending=T/F) / df.sort_values(by=['열'], asceding=[T/F])
>  * 고유값 확인
>   - df['열'].unique() / df['열'].value_counts()
>  * 기본 집계 메소드
>   - df['열'].sum() / df['열'].mean() / df['열'].max() / df['열'].min()
>   
