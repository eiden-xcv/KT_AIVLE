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
>   * **Hidden Markov Model(HMM) 기반 품사 태깅**
>     * W_i : i번째 단어, P_i는 W_i의 품사태그
>     * Pr(W_i | P_i) = freq(W_i, P_i) / freq(P_i)
>     * Pr(P_i | P_i-1) = freq(P_i-1, P_i) / freq(P_i)
>   * 세종말뭉치 품사태그셋 & 영어 품사태그(Penn Treebank)
> * 형태소 분석의 이슈
>   * 전처리 / 형태소 분석의 모호성 / 미등록어 처리 / 지속적인 사전 업데이트
> ### 2) 규칙 & 패턴 기반 자연어처리
> * 개체명 인식(Named Entity Recognition, NER)
>   * 누가, 무엇(누구)을, 언제, 어디서, 얼마나 등과 같은 정보
> * 개체명 카테고리
>   * OntoNotes
> * 개체명 태깅 기법(BIO 태깅)
>   * BIO(Beginning-Insdie-Outside) 태깅 - 개채명 태깅 방법 중 하나
> * 구문 분석(Syntax Analysis)
>   * 언어별 grammar와 lexicon(어휘의 품사/속성 정보를 담은 사전)에 기반하여 문장의 구문 구조를 분석
>   * 구문 분석의 모호성 : 하나의 입력문장이 여러가지 구조로 분석 가능한 문제
> * 패턴 매칭
>   * 패턴 매칭 기반 의도 분석 : 사용자 입력 문장을 분석하여 사용자 의도를 분석
>   * 형태소 분석 -> 패턴 매칭(Intent 분석) -> Intent+Context(Context 분석) -> 응답문 조회&생성
> * 기존 자연어처리
>   * 언어자원(형태소 사전, 기분석 사전, 감성어 사전, 개채명 사전) + 규칙&패턴(패턴, 형태소 결합 규칙)
> * 장단점
>   * 장점 : 좋은 성능을 보여줌, 즉각 반영이 가능함
>   * 단점 : 리소스 구축 비용, 새로운 도메인에 적용이 힘듦, 패턴 유지관리 이슈
> * 21세기 세종계획, 모두의 말뭉치, 공공 인공지능 오픈 API/DATA, AI 허브
> ### 3) MeCab
> * MeCab은 형태소 분석기로, 한국어 형태소 분석에도 유용하게 쓰임
> * 간단한 사용법
> ```
> import MeCab
> 
> tagger = MeCab.Tagger()
> sentence = '안녕하세요. 오늘은 수요일입니다.'
> print (tagger.parse(sentence))
> ### output ###
> 오늘	NNG,*,T,오늘,*,*,*,*
> 은	JX,*,T,은,*,*,*,*
> 수요일	NNG,*,T,수요일,Compound,*,*,수/NNG/*+요일/NNG/*
> 입니다	VCP+EC,*,F,입니다,Inflect,VCP,EC,이/VCP/*+ᄇ니다/EC/*
> EOS
> ##############
> * 사용자 사전 추가하기
>   * user-dic에서 nnp.csv(고유명사), nng.csv(일반명사) 등 파일 수정
> ### 4) Hannanum
> * 한국어 형태소 분석기
> ```
> from konlpy.tag import Hannanum
> hannanum = Hannanum()
> hannanum.nouns('sentence')
> ```

## 2. 기계학습 기반 자연어처리
> * 자연어처리 기술 및 응용 문제
>   * 자동 띄어쓰기, 형태소분석, 개채명인식, 구문분석, 의미분석 
>   * 문서분류, 감성 분석, 언어모델, 키워드 추출, 요약, 기계번역, 질의응답, 챗봇
> * 자연어처리와 기계학습
>   * 대부분의 자연어처리 문제들은 **분류문제**로 해결 가능
> ### 1) 문서 벡터화 & 문서 유사성
> * 문서의 표현
>   * Bag of Words : 문서를 단어의 집합으로 간주, 문서에 나타나는 각 단어는 feature로 간주되고 단어의 출현 빈도에 따른 가중치를 얻음
>   * Feature Selection
>     * 학습 문서에 출현한 term의 부분집합을 선택하는 것
>     * 사전의 크기를 줄여서 학습에 더 효율적인 분류기를 만듦
>     * Noise feature를 제거하여 분류의 정확도를 높임
>     * WordNet 등 어휘 리소스를 활용하여 동의어, 상위어로 단어를 확장
>   * From Text To Weight Vector(가중치 벡터)
> * **Term Extraction**
>   * 추출 단위 : 어절, 형태소, N-gram
> * **Vocabulary Generation**
>   * Document 집합에 있는 Term들을 사전화
>   * Filtering, Document Frequency Count(DF), Ordering, Term ID 부여
>   * Stop Word List : 너무 자주 출현되기에 문서를 변별하는 feature로서 쓸모없는 단어 제외
> * **Term Vocabulary**
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
> ### 2) 문서 분류
> * 대량의 문서를 자동 분류, 컨텐츠 필터링, 의도분석, 감성 분류, 이메일 분류 등
> * 문서 분류 알고리즘
>   * KNN / Naive Bayes Classifier/ Support Vector Machine / CNN, RNN, BERT 등 딥러닝 기반 알고리즘
> * Naive Bayes Classifier
> ```
> C = argmax P(c|x) = argmax (P(x|c) * P(c))/P(x) = argmax P(x|c) * P(c)
> { x:분류될 문장, C:분류 클래스, 사후확률 P(c|x)는 계산하기 어렵기에 Bayes' Rule를 적용 } 
> ``` 
> * CountVectorizer
>   * 문서를 토큰 리스트로 변환하여, 각 문서의 토큰의 빈도를 세어 BoW 벡터로 변환 
> ```
> from sklearn.feature_extraction.text import CountVectorizer
> vec = CountVectorizer(max_features = 1000).fit(train_docs_X) # .fit() : Term Vocabulary 생성
> train_X = vec.transform(train_docs_X).toarray() # .transform() : BoW로 변환
> ```
> * KNN & Naive Bayes Classifier
> ```
> from sklearn.naive_bayes import GaussianNB
> from sklearn.neighbors import KNeighborsClassifier
> 
> gnb = GaussianNB()
> gnb.fit(train_X, train_Y)
> knn = KNeighborsClassifier()
> knn.fit(train_X, train_Y)
> ```

## 3. 텍스트 마이닝
> ### 1) 상용 텍스트마이닝 서비스
> * Text Mining : 대규모 텍스트 자료를 분석하여 "가치 있는" 새로운 정보를 찾아내는 것
> * 소셜미디어 분석 서비스 : pulseK & 바이브 컴퍼니(썸트렌드, 에이셉 뷰티)
> ### 2) 문서 클러스터링
> * 문서 분류 vs 문서 클러스터링
>   * 문서 분류
>     * NLP에서 가장 중요한 분야 중 하나로 다양한 NLP 응용 시스템에서 텍스트 분류 기술을 사용
>     * 스팸 메일 분류 / 문서 카테고리 분류 / 감성 분석 / 의도 분석
>   * **문서 클러스터링**
>     * 문서 분류와는 다르게 비지도학습으로, K-means clustering, DBSCAN 등 클러스터링 알고리즘 사용
> * K-means clustering
>   * 주어진 데이터를 k개의 클러스터로 분할하는 알고리즘
> * DBSCAN(Density-Based Spatial Clustering of Application with Noise)
>   * 노이즈가 있는 대규모 데이터에 적용할 수 있는 밀도 기반의 클러스터링 알고리즘
>   * 데이터 포인트 P를 중심으로 eps 반경 내에 min_samples 이상의 데이터 포인트가 존재하면 클러스터로 인식하고, P는 중심점이 됨
>   * 클러스터의 개수를 미리 지정할 필요가 없으며, noise를 효과적으로 제외할 수 있다는 장점
>   * 밀도가 다른 양상을 보일 때 군집 분석을 잘 못함
> * TfidfVectorizer
>   * ngram을 사용하여 tfidf vector로 변환
> ```
> from sklearn.feature_extraction.text import TfidfVectorizer
> from sklearn.cluster import DBSCAN
>
> tfidf_vectorizer = TfidfVectorizer(min_df = 3, ngram_range=(1,5))
> tfidf_vectorizer.fit(docs)
> vector = tfidf_vectorizer.transform(docs).toarray()
> vector = np.array(vector)
> model = DBSCAN(eps=0.5, min_samples=3, metric = "cosine") 
> result = model.fit_predict(vector) # Computes clusters from a data and predict labels.
> ``` 
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
> ```
> from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
> 
> analyser = SentimentIntensityAnalyzer()
> sentence= 'Every bit of Top Gun Maverick is realistic, moving, and thrilling!'
> score=analyser.polarity_scores(sentence) 
> # output : {'neg': 0.154, 'neu': 0.513, 'pos': 0.333, 'compound': 0.4199}
> # compound결과가 0.05보다 크면 긍정, -0.05보다 작으면 부정
> ```

## 4. 워드 임베딩
> ### 0) 개요
> * Word Representation
>   * Bag of Word : A vector with one 1 and a lot of zeors, 전통적인 방법
>   * Distributed Semantics
>     * Distributional Hypothesis(분포가설) : 같은 문맥의 단어, 즉 비슷한 위치에 나오는 단어는 비슷한 의미를 가짐
>   * 문맥정보(Context)를 사용한 단어 표현
>     * co-occurrence matrix
>     * 단어-문서 행렬(Term-Document matrix)
>       * 벡터가 비슷하면 두 단어가 유사 & 두 문서가 유사
>     * 단어-단어 행렬(Term-Term matrix, Word-Word co-occurrence matrix)
>       * 단어가 늘어날수록 차원이 커지기에 저장공간이 많이 필요하며, 희소한 벡터이다.
> * **Word Embedding**
>   * 중요한 정보만 남기고 적은 차원에서 단어를 표현
>   * 단어를 d차원의 실수벡터로 표현(usually 50<=d<=300)
>   * Predictive-Based
> ### 1) **word2vec**
> * Continuous Bag-Of-Word - Predict a word given its bag-of-words context
>   * 주변 단어(context word)를 입력으로 받아 중심 단어(center word)를 예측하는 방법
>   * 슬라이딩 윈도우 방식으로 학습을 위한 데이터셋 구축
>   * 구조
>     * Input layer(V:단어 집합 크기), Projection layer(M:각 단어의 임베딩 벡터 차원, 여러 입력벡터 결과의 평균), Output layer(V, one-hot vector)
>     * CBOW는 주변 단어로 중심 단어를 더 정확하게 예측하기 위해 계속해서 가중치 벡터인 W(VxM), W'(MxV)를 학습해가는 구조
>     * lookup table
> * Skip-gram - Predict a context word (position-independent) from the center word
>   * 중심 단어에서 주변 단어를 예측
>   * Hidden Layer Weight Matrix(가중치 행렬) == Word Vector Lookup Table(단어 벡터의 lookup table)
> * gensim
>   * 텍스트를 벡터로 변환하는 데 필요한 함수 제공
>  ```
>  import gensim
>
> model = gensim.models.Word2Vec.load(~)
> model.wv['sentence'] # output
> model.wv.most_similar(positive=["word 1-1", "word 2-1"], negative=["word 2-2"], topn=1) # output : "word 1-2"
>  ```
> ### 2) Word Analogy
> * Distributed Representation
> * word2vec 평가
>   * Task-based evaluation - 좋은 단어 벡터를 사용하면 NLP task 성능을 개선
>   * Intrinsic evaluation - 단어 간 유사성에 대한 모델의 판단과 human 판단을 비교
> * word2vec for Data Augmentation
> * word2vec for recommendation

## 5. CNN 기반 자연어 처리
> ### 1) CNN 기반 텍스트 분류
> * 과정
>   * 1. 문장의 지역 정보를 보존하면서 각 문장 성분의 등장 정보를 학습에 반영하는 구조
>   * 2. 학습할 때 각 필터 크기를 조절하면서 언어의 특징 값을 추출하게 되는데, 기존의 N-gram 방식과 유사
>   * 3. max pooling 후 fully connected layer을 이용해 classification
> * 감성 분석(Sentiment Analysis) or 극성 분석(Polarity Detection)
>   * NSMC(Naver Sentiment Movie Corpus) Dataset
> ### 2) pytorch
> 

## 6. RNN 기반 자연어 처리
> ### 1) RNN(Recurrent Nerual Network) 개요
> * 특징
>   * 히든 노드가 방향을 가진 엣지로 연결되어 순환구조를 이루는 신경망 모델
>   * 음성, 텍스트 등 순차적으로 등장하는 데이터 처리에 적합한 모델
>   * 하나의 파라미터 쌍(weights, bias)을 각 시간대 데이터 처리에 반복 사용
>   * 시퀀스 길이에 관계없이 input과 output을 받아들일 수 있는 네트워크 구조여서 다양한 문제에 적용가능하다는 장점
>   * h_t = f_W(h_t-1, x_t) { h_t:new state, h_t-1:old state, x_t:input vector at some time step, f_W:some function with parameters W }
> * 구조
>   * input x_t -> hidden state h_t with activation function (<-h_t-1) -> ouptut y_t
>   * 활성화 함수
>     * 하이퍼볼릭 탄젠트 함수
>       * 실수 범위의 입력값 -> (-1, 1) 사이의 출력값
>       * 기울기가 양수, 음수 모두 나올 수 있기에 시그모이드 함수보다 학습 효율성이 뛰어남
>       * 시그모이드 함수보다 출력값의 범위가 넓기에 출력값의 변화폭이 큼. 따라서 기울기 소멸 현상이 더 적음
> * 품사태깅(POS Tagging)
>   * 같은 단어여도 문장 내 순서에 따라 품사가 달라짐
>   * 학습과정
>     * 정답값과 모델의 예측값을 비교하여 두 값의 차이를 줄여나가는 과정
>     * W_hh, W_xh, b의 값을 최적화
> * 기본 RNN 문제점
>   * 긴 시퀀스를 가진 입력이 들어올 경우 성능이 저조해짐
>   * 특정 시간대에 형성된 정보를 먼 시간대로 전달하기 어려움
>   * gradient vanishing & exploding이 발생하여 모델이 최적화되지 않음
> * **LSTM(Long Short-Term Memory)**
>   * 특징
>     * gradient vanishing & exploding 현상 해소
>     * 정보의 장거리 전달이 가능하여 기본 RNN에 비해 우수한 문제 처리 능력
>     * hidden state에 cell state를 추가, forget gate, input gate를 이용하여 이전 정보를 버리거나 유지
> * Gates
>   * 생성된 시그모이드 값이 0에 가까울수록 입력값을 무시, 1에 가까울수록 입력값을 활용
> * 연산기호
>   * Plus junction & Times junction  
> * 구성
>   * LSTM의 기본 구성요소는 은닉층을 의미하는 Memory Cell
>   * 각 메모리 셀에 적절한 가중치를 유지하는 순환 에지가 있는데, 이 순환 에지의 출력을 cell state라고 함.
>   * t-1의 cell state C_t-1은 어떤 가중치와도 직접 곱해지지 않고 변경되어 t의 cell state C_t를 얻음
>   * LSTM에는 3종류의 gate
>     * forget gate
>       * 메모리 셀이 무한정 성ㅈ아하지 않도록 셀 상태를 다시 설정함
>       * 통과할 정보와 억제할 정보를 결정
>     * input gate
>       * 입력게이트 i_t와 입력 노드 g_t는 셀 상태를 업데이트하는 역할을 담당
>     * output gate
>       * 은닉 유닛의 출력값을 업데이트함
>   * <img src="https://user-images.githubusercontent.com/110445149/194232989-fbed7f92-f4ae-4f7c-8171-36f551bc43d7.JPG" height="300" width="500"></img>
> * Bi-directional LSTM
>   * 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용하며, 첫번째 셀은 앞 시점의 은닉 상태를 계산하고 두번째 셀은 뒤 시점의 은닉 상태를 계산함
> ### 2) 언어 모델
> * 앞 단어(문장의 일부)를 보고 다음에 출현할 단어를 예측하는 모델
> * Statistical LM(N-gram LM) & Neural LM
>   * Statistical LM의 문제점
>     * Data Sparsity & Storage
>   * RNN LM
> ### 3) Sequence-to-Sequence
> * seq2seq
>   * 시퀀스 입력 데이터에 대해 적절한 시퀀스 출력을 학습하기 위한 모델
>   * 두개의 RNN을 이용해 모델링(Encoder-Decoder 모델)
>   * 인코더는 입력 문장의 모든 단어들을 순차적으로 입력받은 뒤에 모든 단어 정보들을 압축하여 하나의 벡터로 만듦 -> Context Vector
>   * 디코더는 Context Vector를 받아 단어를 하나씩 순차적으로 출력
>   * 단점 : 하나의 고정된 크기의 벡터에 모든 정보를 압축하니 정보 손실과 RNN의 고질적 문제인 기울기 소실이 발생
> * 신경망 기계번역(Neural Machine Translation, NMT)
>   * seq2seq의 단점은 context vector가 일종의 bottlenet이 되어 긴 문장에서 long-term dependency가 문제 발생하기에 긴 문장을 번역할 경우 성능 하락
>   * 입력문장 내 특정 단어의 정보를 더 참조할 수 있도록 처리하기 위해, 입력문장 내에 현재 출력될 단어와 관련된 부분에 가중치를 부여하는 기법인 Attention 기법으로 성능 개선
> ### 4) **Attention**
> * 기법
>   * 시퀀스 데이터 모델에서 단어 간 거리에 무관하게 입력, 출력 간의 의존성을 보존해주는 기법
>   * 디코더에서 t번째 단어를 예측하기 위한 Attention value를 계산
>   * Attetion 값은 Query와 Key, Value에 의해 Attention(Q, K, V) 계산
> * 개요
>   * Attention score / Attention distribution / Attetion Value / Attention concatenate / tanh
> * 기법
>   * Bahnadau Attention & Luong Attention

## 7. 챗봇
> * Retrieval-based vs Generative bots
> * 챗봇 시스템의 요소 기술
>   * Natural Language Understanding, Natural Language Generation, Dialog Management, Context Management, Common Sense Reasoning, Offensive Speech Filtering
> * 의도 분석(Intent Analysis)
>   * 기계학습 및 딥러닝 기반 의도 분석
>     * CNN, RNN, BERT 등 활용, 결국 document classification임
>   * 패턴매칭 기반 의도 분석
>     * 정규 표현식으로 구문을 기술, 입력 텍스트에 나타나는 특정한 규칙을 가진 시퀀스 매칭
> * Dialog Manager
>   * 대화 정책 운영, 대화 상태 트랙킹





perplexity
