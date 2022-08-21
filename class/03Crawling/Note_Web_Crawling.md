# Web Crawling - 8/3~8/5

## 0. 개념
> ### 1) URL((Uniform Resource Locator)
> * 구성
>   * protocol + domain(sub do + do) + port + path + page + query + fragment
>   * ex) https:// + news.naver.com + :80 + /main/ : + read.nhn + ?mode=LSD&mid=shm&sid1=105&oid=001&aid=0009847211 + #da_727145
> ### 2) GET & POST 
> * GET 방식 : URL에 데이터가 포함되어 있으며 길이제한이 있다. 데이터가 노출되어 있다는 단점!
> * POST 방식 : body에 데이터가 포함되어 있어 데이터가 숨겨져 있음
> ### 3) OSI 7계층
> * Application + Presentation + Session + Transport + Network + Data Link + Physical
> ### 4) Cookie & Session & Cache
> * Cookie : Client에 저장하는 문자열 데이터로, 도메인 별로 따로 저장
> * Session : Server에 저장하는 객체 데이터로, 브라우저와 연결시 Session ID 생성
> * Cache : Client나 Server의 메모리에 저장하여 빠르게 데이터를 가져오는 목적의 저장소
> ### 5) HTTP Status Code
> * 서버와 클라이언트가 데이터를 주고 받으면 주고 받은 결과를 상태코드를 통해 확인할 수 있다.
>   * 2xx : success
>   * 3xx : redirection(brower cache)
>   * 4xx : request error
>   * 5xx : server error
> ### 6) Web Language & Framwork
> * Client
>   * HTML / CSS / JavaScript(vue.js / react.js / angelar.js / backborn.js)
> * Server
>   * Python(Django/Flask) / Java(Spring) / Ruby(Rails) / Javascript(Node.js) / Scala(Play)
> ### 7) Scraping & Crawling & Spider or Web crawler & Bot

## 1. Requests_JSON
> * 웹 페이지의 종류
>   * 정적 페이지 : 페이지의 데이터가 변경될 때 URL이 변경 -> HTML
>   * 동적 페이지 : 페이지의 데이터가 변경될 때 URL이 고정 -> JSON
> * requests package
>   * 브라우저에서 URL 입력하면 서버에서 데이터를 다운받아 화면에 출력하듯이, requests 패키지 또한 같은 역할을 함 (URL -> DATA)
> * 과정
>   1) 웹서비스를 분석 : chrome 개발자 도구 사용 -> URL 알아내기
>   2) request(url) -> response : JSON(str)
>   3) JSON(str) -> list, dict -> DataFrame
>   ex) 
>     url = '...'
>     response = requests.get(url)
>     data = response.json()
>     df = pd.DataFrame(data)
>   +) 데이터 분석
>     * 상관관계분석 : 두 데이터 집합 사이에 어떤 관계가 있는지 확인하느 분석 방법
>      * 피어슨 상관계수 : df.corr()
>       * 1과 가까울수록 강한 양의 상관관계 / -1과 가까울수록 강한 음의 상관관계

## 2. Requests_API
> * API(Application Programming Interface)를 사용하여 데이터를 수집하는 것으로, 서비스에 데이터를 제공하는 공식적인 방법
> * 과정
>   1) App 등록 -> app_key(==request token) 얻기
>   2) Document 확인하여 요청 URL과 POST방식 parameters이 확인(문서 확인 or payload 확인)
>   3) requests(url, app_key) -> response(json) : JSON(str)
>   4) JSON(str) -> list, dict -> DataFrame
>   * ex)
```
     CLIENT_ID, CLIENT_SECRET = "id", "key"
     url = '...'
     param = {}
     header = {}
     response = requests.post(url, json.dumps(params), headers=headers)
     data = response.json()
     df = pd.DataFrame(data)
```
> + json.dump() : 인터넷 트래픽에서는 영문, 숫자, 특수문자만 사용가능하지만 한글과 같은 문자를 인코딩(영문,숫자,특수문자)해줌
> + requests 요청 시, 403 error가 난다면 headers(user-agent, referer, cookie 등...)가 필요할 수 있다!!
 
## 3. HTML & CSS_Selector
> * HTML
>   * 웹문서를 작성하는 언어
>   * 구성요소
>     1) Document : 한 페이지를 나타내는 코드
>     2) Element : 하나의 레이아웃 ( Element가 모여 Document를 만듦), 계층적 구조를 가짐
>     3) Tag : Element의 종류를 정의 ( 시작태그 + 끝태그 = Element)
>     4) Attribute : 속성값 ( 시작태그에서 태그의 기능 정의)
>       * id : element를 지정하는 페이지 내에서 유일한 값
>       * class : element를 지정하는 값으로 페이지 내에서 여러 개 사용 가능
>       * attr :  id와 class를 제외한 나머지 속성 값
>     5) Text : 시작태그와 끝태그 사이의 문자열
>   * HTML 태그 종류
>     * p : 한줄의 문자열 출력
>     * span : 한 블럭의 문자열 출력
>     * ul, li : 리스트 문자열 출력 
>     * a: 링크를 나타내는 태그
>     * img : 이미지를 출력하는 태그
>     * iframe : 외부 url 링크에 해당하는 웹페이지를 출력
>     * div : 레이아웃을 나타내는 태그
>     * table : 행열 데이터를 출력
>   * CSS Selector
>     * CSS 스타일을 적용시킬 HTML Element를 선택하는 방법 (code 참고)
>       1) Element tag 이름으로 선택 ex) span
>       2) tag의 id 값으로 선택	ex) #"id"
>       3) tag의 class 값으로 선택	ex) ."class"
>       4) tag의 attr 값으로 선택	ex) [value="name"]
>     * 여러 개의 Element를 선택
>       1) not selector ex) .c1:not(.c2) : c1 class 중 c2 class 빼고 모두 선택
>       2) n번째 element 선택 ex) .c1:nth-child(2) : 2번째 element 중에서 c1 class 모두 선택 (주의!!! c1 class 중 2번째가 아님)
>       3) 계층적으로 Element 선택 ex) .c1 > p : c1 class 한 계층 아래 모든 p tag 선택 vs .c1 p : c1 clss 아래 모든 p tag 선택
>       4) 여러개 Element 선택 ex) .c1 .c2 : c1, c2 class 선택

## 4. 정적 페이지 데이터 수집
> * from bs4 import BeautifulSoup : HTML(str)을 CSS Selector를 이용하여 Element 선택!
> * 과정
>   1) 웹서비스 분석 : URL
>   2) requests(url) -> response(html) : HTML(str)
>   3) HTML(str) -> BeautifulSoup Object 
>     - dom = BeatutifulSoup(response.txt, 'html.parser')
>   4) BeautifulSoup Object  - (CSS Selector) -> Data or DataFrame
>     - dom.select() & dom.select_one()

## 5. Selenium
> * 브라우저의 자동화 목적으로 만들어진 다양한 브라우저와 언어를 지원하는 라이브러리
> * 브라우저를 파이썬 코드로 컨트롤하여 브라우저에 있는 데이터 수집
> * ex)
```   
   driver = driver.Chrome()
   driver.get(url)
   elements = driver.find_elements(By.CSS_SELECTOR, selector)
```
> * Headless : 브라우저를 화면에 띄우지 않고 메모리 상에서만 브라우저를 실행하여 크롤링하는 방법

## 6. Scrapy
> * 웹사이트에서 데이터 수집을 위한 오픈소스 파이썬 프레임워크로, 멀티스레딩으로 데이터 수집
> * xpath : HTML에서 element를 선택하는 방법으로, scrapy에서 기본적으로 사용하는 selector
> * from scrapy.http import TextResponse 
> * 과정
>    1) 스크래피 프로젝트 생성 (!scrapy startproject gmarket)
>    2) xpath 찾기
>    3) items.py : 코드 작성 -> model (데이터 구조 : 수집할 데이터의 칼럼을 정의)
>    4) spider.py : 코드 작성 -> 크롤링 절차 정의
>    5) 스크래피 프로젝트 실행   
>    │  scrapy.cfg   
>    │     
>    └─gmarket   
>    │ - items.py : 수집할 데이터의 구조 정의   
>    │ - middlewares.py : 데이터를 수집할 때 headers 정보와 같은 내용 설정   
>    │ - pipelines.py : 데이터를 수집한 후에 코드 실행 정의   
>    │ - settings.py : 크롤링 설정 (크롤링 시간 간격, robots.txt 규칙 등)   
>    │ - __init__.py   
>    │     
>    └─spiders (크롤링 절차 정의)   
>    | - __init__.py   
>    | - spdier.py (크롤링 절차 정의)   
 
## 참고
> ### 1. docstring : 함수를 사용하는 방법을 문자열로 작성
>  * 함수 설명 / parameters / return 에 대한 설명을 포함
>  * hlep() or shift+tab 사용
> ### 2. apply() & lambda
> ### 3. encoding 
>   * ascii : 영문 숫자 특수문자 
>   * euc-kr : + 한글 
>   * utf-8 : + 모든 언어
> ### 4. 크롤링 정책
> * robots.txt : 웹사이트에 크롤러와 같은 로봇이 접근하는 것을 방지하기 위한 규약 ex) https://www.ted.com/robots.txt
> * 크롤링에 대한 법적 제재는 없으나 과도한 크롤링으로 서비스에 영향을 주었을 때는 영업방해, 지적재산권 침해 등 문제가 될 수 있다.
> * 따라서, api를 통해 크롤링을 하도록....
> ### 5. with 문법
>    * with {Experssion} as {Variable} : Coed Block
>    * 자원을 획득하고 사용 후 반납해야 하는 경우 주로 사용함 (자원 획득 -> 사용 -> 반납 프로세스)   
>    ex) `with open(f'{path}/test.png', 'wb') as file : file.write(response.content)`
>    * 컨텍스트 매니저
>	     * __enter__(self) : with문 진입 시 자동으로 호출
>	     * __exit__(self, type, value, traceback ) : with문 끝나기 직전에 자동으로 호출
> ### 6. yield
>    * iterator : next()를 호출할 때 다음 값을 생성해내는 상태를 가진 헬퍼 객체
>    * genartor : iterator를 간단하게 구현한 문법 ( ex. def fib() : )
>    * yield : 일반 함수를 generator로 만들어주는 명령어로, next 함수를 실행해서 yield를 만나면 코드 실행 중단
