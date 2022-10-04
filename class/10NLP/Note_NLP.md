# NLP - 9/30~10/7

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
>   * 자동 띄어쓰기, 형태소분석, 개채명인식, 구문분석, 의미분석 
>   * 문서분류, 감성 분석, 언어모델, 키워드 추출, 요약, 기계번역, 질의응답, 챗봇
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
>     * Cosine Similarity
>     
> ### 2) 문서 분류
> * 대량의 문서를 자동 분류, 컨텐츠 필터링, 의도분석, 감성 분류, 이메일 분류 등
> * 문서 분류 알고리즘
>   * KNN / Naive Bayes Classifier/ Support Vector Machine / CNN, RNN, BERT 등 딥러닝 기반 알고리즘

## 3. 텍스트 마이닝
> ### 1) 상용 텍스트마이닝 서비스
> * Text Mining : 대규모 텍스트 자료를 분석하여 "가치 있는" 새로운 정보를 찾아내는 것
> * 소셜미디어 분석 서비스 : pulseK & 바이브 컴퍼니(썸트렌드, 에이셉 뷰티)
> ### 2) 문서 클러스터링
> * 문서 분류 vs 문서 클러스터링
>   * 문서 분류
>     * NLP에서 가장 중요한 분야 중 하나로 다양한 NLP 응용 시스템에서 텍스트 분류 기술을 사용
>     * 스팸  메일 분류 / 문서 카테고리 분류 / 감성 분석 / 의도 분석
>   * **문서 클러스터링**
>     * 문서 분류와는 다르게 비지도학습으로, K-means clustering, DBSCAN 등 클러스터링 알고리즘 사용
> * K-means clustering
> * DBSCAN(Density-Based Spatial Clustering of Application with Noise)
>   * 노이즈가 있는 대규모 데이터에 적용할 수 있는 밀도 기반의 클러스터링 알고리즘
>   * 데이터 포인트 P를 중심으로 eps 반경 내에 min_samples 이상의 데이터 포인트가 존재하면 클러스터로 인식하고, P는 중심점이 됨
>   * 클러스터의 개수를 미리 지정할 필요가 없으며, noise를 효과적으로 제외할 수 있다는 장점
>   * 밀도가 다른 양상을 보일 때 군집 분석을 잘 못함
> ### 3) 키워드 추출
> * 문서 내용을 요약하는 기술
> * 추출 요약(Extractive Summarization)
>   * 주어진 문서 내에서 이를 대표할 수 있는 키워드들이나 핵심 문장들을 선택하여 문서를 요약하는 기술
>   * 통계 기반으로 작동하므로 학습데이터 불필요
> * 추상 요약(Abstractvie Summarization)
>   * 같은 의미의 다른 표현(paraphrasing)을 사용하거나 새로운 단어를 사용함으로써 새로운 문장으로 된 요약문을 생성하는 기술
>   * 학습데이터를 기반으로 한 supervised learning이라는 것이 단점
> * PageRank 알고리즘
>   * 많은 유입 링크(backlinks)를 지니는 페이지가 중요한 페이지라 가정
>   * 각 웹페이지는 다른 웹페이지에게 자신의 점수를 1/n을 분배(n:outbound 링크수)
>   * backlinks가 많은 페이지는 점수가 높아짐
> * TextRank 기반 키워드 추출
>   * 그래프 기반의 text summarization 기법
>   * PageRank를 사용하여 문서 내의 키워드 또는 핵심 문장을 추출
>   * 문서 집합의 핵심 단어를 선택하기 위해 단어 그래프(co-occurrence graph)를 생성
>   * 생성된 그래프에 PageRank를 학습하여 각 노드(단어 or 문장)의 랭킹을 계산하고, 랭킹이 높은 순서대로 문서를 대표하는 키워드 또는 핵심문장으로 선택
>   * 단어 그래프 생성
>     * Vertex 생성
>       * 주어진 문서 집합을 품사태깅한 후, 최소 빈도수 이상 등장한 단어를 대상으로 명사, 고유명사, 동사, 형용사 등을 vertex로 생성
>     * Edge 생성
>       * 두 단어가 co-occurrence(window size 내에 두 단어가 동시에 출현) 관계가 있을 경우, vertex간 edge를 생성
>     * vertex의 초기 중요도 1로 설정, 수렴할 때까지 알고리즘 반복 실행
>     * <img src="https://user-images.githubusercontent.com/110445149/193722760-fe370ff1-ea87-400d-bc85-a4415a7a9c75.JPG" height="60" width="400"></img>
> ### 4) 감성 분석(Sentiment Analysis)
> * 사전기반 감성 분석 툴
>   * VADER(Valence Aware Dictionary and sEntiment Reasoner)
>     * 사전과 규칙 기반의 감성 분석 툴로써, 소셜미디어 텍스트 분석에 강점
> * Sentiment(감성) vs Emotion(감정)
>   * Emotion - complex psychological state such as happiness, anger, jealousy, grief, etc.
>   * Sentiment - mental attitude that is created through the existence of the emotion.
