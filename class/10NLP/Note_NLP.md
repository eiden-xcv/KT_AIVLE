# NLP - 9/30

## 1. 자연어처리 요소기술
> ### 0) NLP
> * NLP
>   * NLP = Computer Science + Linguistics + Artificial Intellignece
>   * SW를 이용하여 인간의 언어를 처리하고 이용하려는 연구 분야
> * Goal
>   * 인간의 언어로 디지털 디바이스(SW/HW)와 상호작용을하여 원하는 task를 수행하는 것
> * 결과물
>   * 정보검색 / 기계번역 / 챗봇 / 텍스트 마이닝
> ### 1) 형태소분석과 품사태깅
> * NLP 요소기술
>   * 텍스트 처리 / 형태소 분석 & 품사태깅 / 개채명 인식 / 패턴 매칭 / 구문 분석 / 의미 분석 
>   * 텍스트 분류 / 감성 분석 / 자연어 생성 / 문서 요약
> * 한국어의 품사
>   * 품사 : 단어를 그 문법적 성질에 따라 분류한 갈래
>   * 체언 / 용언 / 수식언 / 독립언 / 관계언
> * 형태소 분석
>   * 한 어절 내에 있는 모든 형태소를 분리
>   * 용언의 원형을 복원
>   * 형태소들 간의 결합관계가 적합한지 검사
>   * 복합어 분석, 미등록어 추정
>   * 형태소 분석의 모호성
>     * 동일한 표층형(surface form) 어절이 여러가지 형태소 결합으로 분석 가능한 문제
> * 품사 태깅(Parts-of-Speech Tagging)
>   * 품사태그 : 형태소 분석의 기준이 되는 세분화된 품사 체계
>   * 형태소 분석의 모호성을 해결 & 해당 문맥에 맞는 품사태그를 선택하는 문제
>   * Hidden Markov Model(HMM) 기반 품사 태깅
>     * W_i : i번째 단어, P_i는 W_i의 품사태그
>     * Pr(W_i | P_i) = freq(W_i, P_i) / freq(P_i)
>     * Pr(P_i | P_i-1) = freq(P_i-1, P_i) / freq(P_i)
>   * 세종말뭉치 품사태그셋 & 영어 품사태그(Penn Treebank)
> * 형태소 분석의 이슈
>   * 전처리 / 형태소 분석의 모호성 / 미등록어 처리 / 지속적인 사전 업데이트
> ### 2) 규칙 & 패턴 기반 자연어처리
> * 개채명 인식(Named Entity Recognition, NER)
>   * 누가, 무엇(누구)을, 언제, 어디서, 얼마나 등과 같은 정보
> * 개체명 카테고리
>   * OntoNotes
> * 개체명 태깅 기법(BIO 태깅)
>   * BIO(Beginning-Insdie-Outside) 태깅 - 개채명 태깅 방법 중 하나
> * 구문 분석(Syntax Analysis)
>   * 언어별 grammar와 lexicon(어휘의 품사/속성 정보를 담은 사전)에 기반하여 문장의 구문 구조를 분석
>   * 구문 분석의 모호성 : 하나의 입력문장이 여러가지 구조로 분석 가능한 문제
> * 패턴 매칭
>   * 패턴 매칭 기반 의도 분석 : 사용자 입력 문장을 분석하여 사용자 의도를 분석\
>   * 형태소 분석 -> 패턴 매칭(Intent 분석) -> Intent+Context(Context 분석) -> 응답문 조회&생성
> * 기존 자연어처리
>   * 언어자원(형태소 사전, 기분석 삿전, 감성어 사전, 개채명 사전) + 규칙&패턴(패턴, 형태소 결합 규칙)
> * 장단점
>   * 장점 : 좋은 성능을 보여줌, 즉각 반영이 가능함
>   * 단점 : 리소스 구축 비용, 새로운 도메인에 적용이 힘듦, 패턴 유지관리 이슈
> * 21세기 세종계획, 모두의 말뭉치, 공공 인공지능 오픈 API/DATA, AI 허브

## 2. 기계학습 기반 자연어처리
> * 자연어처리 기술 및 응용 문제
>   * 자동 띄어쓰기, 형태소분석, 개채명인식, 굼누분석, 의미분석 
>   * 문서분류, 감성분석, 언어모델, 키워드 추출, 요약, 기계번역, 질의응답, 챗봇
> * 자연어처리와 기계학습
>   * 대부분의 자연어처리 문제들은 분류문제로 해결 가능
> ### 1) 문서 벡터화 & 문서 유사성
> * 문서의 표현
>   * Bag of Words : 문서를 단어의 집합으로 간주, 문서에 나타나는 각 단어는 feature로 간주되고 단어의 출현 빈도에 따른 가중치를 얻음
>   * Feature Selection : 학습 문서에 출현한 term의 부분집합을 선택하는 것, 사전의 크기를 줄여서 학습에 더 효율적인 분류기를 만듦
>   * From Text To Weight Vector
> * **Term Extraction**
>   * 추출 단위 : 어절, 형태소, N-gram
> * **Vocabulary Generation**
>   * Document 집합에 있는 Term들을 사전화
>   * Filtering, Document Frequency Count(DF), Ordering, Term ID 부여
>   * Stop Word List : 너무 자주 출현되기에 문서를 변별하는 feature로서 쓸모없는 단어 제외
> * **Document Transformation**
>   * Term Frequency Vector
>     * Term -> ID / Out of Vocabulary Term 제거 / 각 Term의 문서 내 Frequency를 Count(TF)
> * **Document Weighting**
>   * 가중치 벡터로 문서를 표현
>     * Weighting 기법 : TF or TF x IDF / Probabiliyt
>   * IDF(Inverse Document Frequency)
>     * idf_t = log(N/df_t)   [N:문서집합의 총 문서수, df:문서빈도, tf:용어빈도]
>   * TF-IDF 값
>     * TF와 IDF를 결합하여 각 용어의 가중치를 계산
>     * 문서 D에 나타난 용어 t에 부여되는 가중치
>     * TF-IDF(t,D) = TF x IDF
>     * 적은 수의 문서에서 용어t가 많이 나타날 때 가장 높은 값을 가짐(높은 식별렬)
>     * 한 문서나 많은 문서들에서 그 용어가 적게 나타나면 낮은 값을 가짐(뚜렷하지 않은 적합성)
> * 문서 유사성
>   * Term Vector Model
>     * Document Vector간 유사도를 계산하여 유사성 비교
> ### 2) 문서 분류
> * 대량의 문서를 자동 분류, 컨텐츠 필터링, 의도분석, 감성 분류, 이메일 분류 등
> * 문서 분류 알고리즘
>   * KNN / Naive Bayes Classifier/ Support Vector Machine / CNN, RNN, BERT 등 딥러닝 기반 알고리즘

