# Django를 활용한 Web App - 11/11~11/17

## 1. Django 시작하기
> ### 1) 웹애플리케이션
> * 구조   
> <img src="https://user-images.githubusercontent.com/110445149/201236353-8bfda915-590b-4e4e-8233-bf89a8dad46c.JPG" height="250" width="500"></img>
> ### 2) HTTP 프로토콜의 이해
> 1. 웹 브라우저는 웹서버에 원하는 서비스 요청
> 2. 웹서버는 웹 브라우저가 요청한 파일을 찾아서 웹클라이언트에 응답하거나 요청한 파일을 수행시키고 그 결과를 브라우저에 응답
> 3. 웹 브라우저는 웹서버로부터의 응답 결과를 출력
> * HTTP 요청정보
>   * 헤더 - 요청방식(GET/POST/PUT/DELETE/PATCH) + URI + HTTP 버전
> * HTTP 응답정보
>   * 헤더 - HTTP 버전 + Status-Code + Reason-Phrase
> ### 3) Django
> * 파이썬 기반의 Full-stack 프레임워크
> * 특징
>   * 간결하고 쉬운 파이썬 언어를 사용하기에 배우기 쉬움
>   * 많은 라이브러리와 프레임워크 제공으로 쉽고 빠르게 개발 가능
>   * 확장성이 뛰어나 복잡한 요구사항과 통합이 필요한 개발에 적합
>   * 불필요한 중복을 없애고 많은 양의 코드를 줄여 유지보수가 쉽고 재사용하기 좋은 디자인 원칙과 패턴을 사용
>   * 리눅스, 윈도우, Mac OS 등 다양한 운영체제에서 작동
>   * 비밀번호, 세션, 크로스사이트 요청 위조 등의 보안 취약점을 보완할 방법을 기본적으로 제공
> * 요소 
>   * View : HTTP의 요청 처리
>   * Model : 데이터베이스 처리
>   * Template : 사용자의 인터페이스 처리
>   * Form : 사용자의 입력 데이터 처리
>   * Static 파일, Media 파일, Message framework, Send Email, Admin앱, Auth앱, Session앱 등
> ### 4) Django 서비스 구축
> * MVT 디자인 패턴 (Model - View - Template)   
> <img src="https://user-images.githubusercontent.com/110445149/201237503-df78dd76-efff-4c7e-bd08-66240fb2e29a.JPG" height="300" width="200"></img>
>   * View 
>     * 사용자의 요청에 대한 서비스 처리를 담당하며 view.py 파이썬 스크립트에 구현
>   * Model 
>     * DB 데이터 처리를 담당하며 model.py 파이썬 스크립트 구현
>   * Template
>     * 사용자들이 서비스를 요청할 수 있는 화면 또는 서비스가 처리된 결과 화면 처리를 담당하며 HTML 파일로 구현
> * Django 처리 구조   
>   * (Django HTTP Handler) 요청정보 -> (urls) URL 확인 -> (views) 서비스 처리 -> (models) DB처리 -> (templates) 탬플릿으로 응답정보 생성 or ()직접 응답정보 생성 -> (Django HTTP Handler) 응답정보    
> <img src="https://user-images.githubusercontent.com/110445149/201237751-bec0a1b9-99c2-43ab-9dd8-4e0c3ee61b63.JPG" height="300" width="300"></img>

## 2. Django Project
> ### 1) Django Project & APP
> * Project
>   * Django에서 웹 사이트 또는 웹 애플리케이션을 프로젝트라 부름
> * App
>   * 웹 사이트에는 사용자의 서비스를 처리하는데 제공하는 기능을 앱이라 부름
> * Django 프로젝트 생성
>   * ``` django-admin startproject <project_name>``` 을 통해 프로젝트를 생성
>   * 초기 프로젝트는 manage.py 와 환경설정 폴더(프로젝트명과 동일) 아래 asgi.py, settings.py, urls.py, wsgi.py, \_\_init\_\_.py 가 생성됨.
>   * 이후 앱 생성 및 앱 등록 절차를 진행
> * manage.py
>   * 현재 개발 중인 Django 프로젝트의 개발 과정에서 필요한 작업을 실행시켜주는 커멘트 유틸리티
>   * 사용법 :  ``` python manage.py <command> [options]```
> * Django 앱 생성 및 등록
>   * ``` python manage.py startapp <app_name> ``` 을 통해 앱 생성
>   * 앱 아래 생성되는 파일
>     * models.py : 현재 앱에서 사용하는 모델에 대해 구현하는 파일
>     * views.py : 현재 앱의 서비스를 기능을 구현하는 파일
>     * 그 외 admin.py, apps.py, tests.py, \_\_init\_\_.py 파일과 migrations 폴더가 생성
>   * 앱 등록은 프로젝트 환경설정 폴더에 INSTALLED_APPS에 등록하면 됨
> * URL과 View 매핑
>  * 프로젝트 URL 관리는 settings.py의 ROOT_URLCONF인 urls.py의 urlpatterns에서 함
>  * ```path(URL, View)```를 urlpatterns에 등록함으로써 URL과 View가 매핑됨
>  * 직접적으로 View를 등록해도 되지만, 각 앱 아래 urls.py를 통해 관리하기 위해 ```path(URL, include('app_name.urls')```을 함
>   * path 변수 선언
>     * URL 문자열 일부를 뷰함수의 인자로 전달하기 위해 선언하는 변수
>     * 선언된 변수는 default로 문자열 타입이기에 django.urls.converters 모듈을 활용하여 \<int:no\>와 같이 <DEFAULT_CONVERTERS의 키:변수명>을 통해 형변환 가능

## 3. Django Model
> * ORM
>   * ORM은 Object Relational Mapping의 약자로 객체와 데이터베이스의 관계를 매핑해주는 도구
>   * 즉, 프로그래밍언어의 객체와 관계형 데이터베이스의 데이터를 자동으로 매핑해주는 도구
>   * MVC 패턴에서 모델을 기술하는 도구이며, 객체와 모델 사이의 관계를 기술하는 도구
>   * 장점
>     * 직접적인 코드를 통해 가독성을 높이고, 비지니스 로직 집중함으로써 생산성을 높일 수 있음
>     * 재사용 및 유지보수 편리성 증가 
>     * DBMS에 대한 종속성 저하
>   * 단점
>     * ORM으로만 복잡한 서비스를 구현하기 어려움
> ### 1) Model 환경설정
> * DB 환경설정
>   * settings.py의 DATABASES
> ### 2) Model 생성
> * 앱 아래 models.py에 정의를 하며 django.db.models의 Model 클래스를 상속함
> * Migration
>   * 1단계(Model -> 파일) : 마이그레이션 파일 작성 == Class -> SQL
>     * ``` python manage.py makemigrations <app_name> ```
>   * 2단계(파일 -> DB) : 마이그레이션 파일의 내용을 DB에 반영 == SQL -> DB
>     * ``` python manage.py migrate <app_name> ```
> ### 3) Model 사용
> * 관리자 계정 만들기
>    * ``` python manage.py createsuperuser ```
> * admin 페이지 등록
>   * 앱 아래 admin.py에서 모델 등록 by ``` admin.site.register(<model_name>) ```
>   * 모델 인스턴스 작업 가능
> * \_\_str\_\_ 메서드
>   * 모델 인스턴스 출력시 특정 내용을 출력하고자 할 때 오버라이딩하는 메서드

## 4. Django Template
> * Template
> *   Client의 요청에 따라 View 함수에서 응답하는 HTML파일
> ### 1) Template 환경설정
> * settings.py의 TEMPLATES 
> * Template의 검색위치는 등록된 앱 순서대로 앱 아래 templates 폴더, 없으면 TEMPLATES의 'DIR'
> ### 2) Template 응답
> * render
>   * Template 페이지인 HTML파일을 응답할 때 render 함수 사용
>   * ``` render(HttpReqeust, Template, [context]) ```
> * context
>   * View 함수에서 Template으로 전달하는 데이터로, key:value 형식으로 여러 개의 데이터를 요소로 갖는 딕셔너리
> * path의 중복을 막기위해 앱 아래 templates\\<app_name> 아래 template 파일을 저장
> ### 3) Template 태그
> * for
>   * ``` {% for %}  {% endfor%}```
> * if
>   * ```{% if %}  {% elif %}  {% else %} {% endif %}``` 
> ### 4) Template 필터
> * 필터
>   * ``` {{ 값 | 필터 : 인자 | 필터 }}``` 
>   * Template에서 {{ }}문법은 {{ }}안의 값을 출력하는 명령문
>   * 이때 값을 출력할 때 그대로 출력하는 것이 아니라 가공을 해서 출력을 할 수 있도록 함
>   * 즉, 필터란 출력 전 전처리 작업을 하는 함수
>   * 파이프라인을 통해 다수의 필터 함수 적용 가능
>   * linebreaks, truncatechars/truncatechars_html, truncatewords/truncatewords_html, date&time, timesince, timeuntil
> ### 5) Template 상속
> * 부모 Template
>   * 공동 코드 구현 : ``` {% block "이름" %} {% endblock %} ```
> * 자식 Template
>   * 상단에 부모 Template을 상속받음 ``` {% extends "부모Template 경로" %} ```
>   * 부모 Template에게 전달할 블록 지정 ``` {% block "이름" %} {% endblock "이름" %} ```

## 5. Django Model 활용
> ### 1) Model Field
> * 문자열
>   * CharField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR로 지정
>     * 가변길이 문자열이 저장되기에 최대 길이 값인 max_length를 지정해야함
>   * TextField 
>     * SQL문으로 변환 시 데이터 타입이 TEXT로 지정
>     * 길이제한이 없는 문장열 저장 시 사용
>   * SlugField
>     * 친화적인 URL을 만들기 위한 문자,숫자,밑줄,하이픈으로 구성된 짧은 문자열
>   * EmailField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(254)로 지정
>     * 입력된 문자열은 Email 형식
>   * URLField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(200)로 지정
>     * 입력된 문자열은 IP4/IP6 또는 도메인 이름의 형식
>   * UUIDField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(32)로 지정
>     * 32개의 16진수를 (8)-(4)-(4)-(4)-(12)와 같이 하이픈으로 5개 그룹을 구분한 형식(UUID 형식)
>   * GenericIPAddressField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(39)로 지정
>     * IP4 혹은 IP6 형식
> * 날짜/시간
>   * DateField, DateTimeField, TimeField
> * Null/Boolean
>   * BooleanField, NullBooleanField
> * 숫자
>   * (Small/Big)AutoField, (Positive)(Small/Big)IntegerField, FloatField
>   * AutoField
>     * SQL문으로 변환 시 데이터 타입이 Integer로 지정
>     * 필드 값이 초기값 1부터 시작해서 새로운 레코드가 삽입될 때마다 1씩 증가된 값이 자동으로 저장
>     * 일반적으로 모델 클래스에서 선언해서 사용하기보다는 migration 작업 시 자동으로 추가되어 사용
>     * 모델 클래스 선언 시 primary key가 지정된 필드가 없는 경우 migration 작업 시 자동 생성
>     * SmallAutoField는 1~32767, BigAutoField는 1~1~ 9223372036854775807 범위의 값을 가짐
> * 파일  
>   * BinaryField
>     * 파일의 원본(binary) 데이터를 저장하는 필드
>     * bytes, bytearray, memorview 인스턴스로 표현
>     * DB에 파일을 저장하는 것임으로 사용시 주의
>   * FileField
>     * 사용자의 파일 업로드를 지원하기 위한 필드
>     * settings.MEDIA_ROOT 값으로 지정된 폴더에 업로드된 파일을 저장
>   * ImageField
>     * FileField를 상속받으며 업로드된 파일이 유효한 이미지 파일인지 유효성 체크함
>   * FilePathField
>     * 사용자가 업로드한 파일이 아니라 이미 파일 시스템에 있는 파일을 다루기 위해 사용
>     * FilePathField 생성 시 path 옵션에 사용할 파일이 있는 폴더를 반드시 지정해야 함
> ### 2) Model Field 옵션
> * Field 제약조건
>   * null, blank, default, unique, unique_for_date, primary_key, choices, validators
> * DB 정보
>   * db_column, db_index
> * Form 정보
>   * editable, error_message, help_text, verbose_name
> ### 3) 관계 설정
> ### 4) 관계 이름

## 6. Django ORM
> ### 1) Manager & Query
> ### 2) 조회
> ### 3) 추가
> ### 4) 수정
> ### 5) 삭제

## 7. Django admin App
> ### 1) admin App
> ### 2) Model 등록
> ### 3) 커스터마이징

## 8. Django Form
> ### 1) HTML Form
> ### 2) CSRF
> ### 3) HttpRequest
> ### 4) Django Form
> ### 5) URL Reverse
> ### 6) ModelForm
> ### 7) 유효성 검사

## 9. Django View
> ### 1) 기본 View
> ### 2) Generic View

## 10. Django File
> ### 1) Static 파일
> ### 2) Media 파일

## 11. Django RESTful API
> ### 1) RESTful API 개요
> ### 2) Django REST Framework
> ### 3) RESTful API 
