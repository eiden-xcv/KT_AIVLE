# JavaScript - 11/7~11/8

## 1. JavaScript
> ### 1) JavaScript 개요
> * JavaScript
>   * 동적인 웹페이지를 만들기 위해 사용하는 프로그래밍 언어
>   * 기능
>     * 동적 컨텐츠 업데이트 / 웹, 모바일 애플리케이션 개발 / 서버 애플리케이션 개발
> * JavaScript 엔진
>   * JavaScript 코드를 실행하는 프로그램
>   * Google V8 - Chrome & Node.js
> * Client Side JavaScript
>   * 웹페이지 로드시 웹문서 DOM 구조로 변환
>   * JavaScript 코드를 바이트코드로 변환
>   * 마우스 클릭과 같은 이벤트와 연결도니 JavaScript 코드 블록 실행
>   * 브라우저에 새로운 DOM 표시
> * Server Side JavaScript
>   * 데이터베이스 엑세스
>   * 서비스 요청 처리
>   * 서비스 응답 처리
> * JavaScript 라이브러리
>   * 데이터 시각화
>   * DOM 조작
>   * 양식
>   * 수학 및 텍스트 함수
> * JavaScript 프레임워크
>   * 웹 및 모바일 애플리케이션 개발
>   * 반응형 웹 개발
>   * 서버 측 애플리케이션 개발

## 2. 데이터 타입과 변수
> ### 1) 데이터 타입
> * 데이터타입
>   * 기본타입 : number, string, boolean, undefined
>   * 참조타입 : Object(Array, Function!(function), Date, RegExp...)
> * Object 
>   * new ~ () or { ~ }
>   * Array는 [ ~ ]
> * var 키워드
> ### 2) 연산자와 형변환
> * 산술 연산자와 암시적 형변환
> * 비교 연산자

## 3. 함수와 실행 컨텍스트
> ### 1) 호이스팅
> * 함수의 정의
>   * 생성자 함수 방법 - var f = new Function()
>   * 선언적 함수 방법 - fuction f(){ }
>   * 리터럴 함수 방법 - var f = function(){ }
> * **호이스팅**
>   * 인터프리터가 변수와 함수의 메모리 공간을 선언 전에 미리 할당하는 것을 의미함
>     * [1] 선언적 함수 - 변수생성,초기화(호이스팅)
>     * [2] var - 변수생성(호이스팅), 초기화(실행시)
>   * 함수 호출 시 실행되는 순서
>     * [1] 함수 실행 영역(Execute Context Stack) 생성
>     * [2] 매개변수, arguments 변수 생성 및 초기화
>     * [3] 선언적 함수
>     * [4] var
>     * [5] this
>     * [6] 코드 실행
> ### 2) 함수의 파라미터
> ### 3) 함수의 리턴값
> ### 4) 스코프와 스코프 체인
> * 현재 실행중인 영역에서 변수 확인
> * 즉시 실행 함수
>   * (f(){})()
> ### 5) 클로저
> * 함수와 함수가 선언된 어휘적 환경(Lexical Environment)의 조합
> ```
> function outerFunc() {
>   var x = 10;
>   var innerFunc = function () { console.log(x); };
>   innerFunc();
> }
>  
> outerFunc(); // 10
> ```
> * Example 1
>   * 함수 outerFunc 내에서 내부함수 innerFunc가 선언되고 호출됨
>   * 이때 내부함수 innerFunc는 자신을 포함하고 있는 외부함수 outerFunc의 변수 x에 접근 가능
>   * 이는 innerFunc가 outerFunc의 내부에 선언되었기 때문.
>   * 렉시컬 스코핑(Lexical scoping)
>   * 스코프는 함수를 호출할 때가 아니라 함수를 어디에 선언하였는지에 따라 결정됨
>   * innerFunc는 함수 outerFunc의 내부에서 선언되었기 때문에 innerFunc의 상위 스코프는 outerFunc임.
> ```
> function outerFunc() {
>   var x = 10;
>   var innerFunc = function () { console.log(x); };
>   return innerFunc;
> }
>
> var inner = outerFunc();
> inner(); // 10
> ```
> * Example 2
>   * 함수 outerFunc는 내부함수 innerFunc를 반환하고 생을 마감 
>   * 즉, outerFunc는 실행된 이후 콜스택(실행 컨텍스트 스택)에서 제거
>   * outerFunc의 변수 x 또한 더이상 유효하지 않게 되어 변수 x에 접근할 수 있는 방법은 달리 없어 보이나, 코드의 실행 결과는 변수 x의 값인 10임 
>   * 이처럼 자신을 포함하고 있는 외부함수보다 내부함수가 더 오래 유지되는 경우, 
>   * 외부 함수 밖에서 내부함수가 호출되더라도 외부함수의 지역 변수에 접근할 수 있는데 이러한 함수를 **클로저(Closure)** 라고 부른다.
> * [참고]https://poiemaweb.com/js-closure
