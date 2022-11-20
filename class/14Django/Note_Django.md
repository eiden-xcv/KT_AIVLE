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
>     * 선언된 변수는 default로 문자열 타입이기에 django.urls.converters 모듈을 활용하여 \<int:no\>와 같이 \<DEFAULT_CONVERTERS의 키:변수명\>을 통해 형변환 가능

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
>     * 입력된 문자열은 IPv4/IPv6 또는 도메인 이름의 형식
>   * UUIDField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(32)로 지정
>     * 32개의 16진수를 (8)-(4)-(4)-(4)-(12)와 같이 하이픈으로 5개 그룹을 구분한 형식(UUID 형식)
>   * GenericIPAddressField
>     * SQL문으로 변환 시 데이터 타입이 VARCHAR(39)로 지정
>     * IPv4 혹은 IPv6 형식
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
>     * SmallAutoField는 1 ~ 32767, BigAutoField는 1 ~ 9223372036854775807 범위의 값을 가짐
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
> * 데이터가 중복되지 않도록 분리된 모델들의 데이터를 사용할 때는 관계(Relationship) 설정을 통해 연결해서 사용하며 모델 간의 관계는 업무 규칙에 따라 설정
> * 종류
>   * 1:M 관계
>     * ForeignKey 클래스
>       * models.ForeignKey(to, on_delete, \*\*options)
>       * to에 1:M 관계에서 1측에 해당하는 모델명을 지정
>       * on_delete : 참조하는 인스턴스가 삭제되었을 때 처리하는 방식을 지정하는 변수로, 필수 옵션!!
>   * 1:1 관계
>     * OneToOneField 클래스
>       * models.OneToOneField(to, on_delete, \*\*options)
>       * to에 1:1 관계를 모델명을 지정
>       * on_delete : 참조하는 인스턴스가 삭제되었을 때 처리하는 방식을 지정하는 변수로, 필수 옵션!! 
>   * M:M 관계
>     * ManyToManyField
>       * models.ManyToManyField(to, \*\*options)
>       * to에 M:M 관계를 모델명을 지정
> ### 4) 관계 이름
> * 종류
>   * 1:M 관계
>     * **1측 인스턴스명.M측 모델명 소문자_set**
>   * 1:1 관계
>     * **1측 인스턴스명.1측 모델명 소문자**
>   * M:M 관계
>     * ManyToManyField가 설정된 모델의 인스턴스를 통해 접근할 때는 **인스턴스명.필드명**
>     * ManyToManyField가 설정되지 않은 모델의 인스턴스를 통해 접근할 때는 **인스턴스명.M측 모델명 소문자_set**
> * 사용자 정의 관계 이름
>   * 위와 같은 관계이름 대신 직접 related_name을 통해 이름 설정 가능
>   * ex) ForeignKey(Post, on_delete=models.CASCADE, **related_name**='comments')

## 6. Django ORM
> ### 1) Manager & Query
> * 모델명.objects.메서드()
> * models.Model 모듈 안에 objects=models.Manager()와 같이 정의되어 있으며, 이 Manager() 클래스는 ORM 처리하는 역할을 함
> * QuerySet
>   * QuerySet을 반환하는 함수
>     * all(), filter(), exclude(), annotate(), order_by(), reverse(), distinct(), values(), values_list(), date(), datetimes(), none(), union(), intersetcion(), difference(), select_related(), prefetch_related(), extra(), defer(), only(), using(), select_for_update(), raw()
>   * QuerySet을 반환하지 않는 함수
>     * get(), count(), create(), get_or_create(), update_or_create(), bulk_create(), bulk_update(), in_bulk(), iterator(), latest(), earliest(), first(), last(), aggregate(), exists(), update(), delete(), as_manager(), explain()
> ### 2) 조회
> * all() - 모델명.objects.all()
> * order_by() - 모델명.objects.order_by('필드명') or QuerySet객체명.order_by('필드명')
>   * Meta 내부 클래스의 ordering을 활용하면 자동적으로 정렬을 시킬 수 있음
>   ```
>     class 모델명(models.Model):
>       class Meta:
>         ordering=['필드명']
>   ```
> * filter() - 모델명.objects.filter(조건) or QuerySet객체명.filter(조건)
> * exclude() - 모델명.objects.exclude(조건) or QuerySet객체명.exclude(조건)
>   * 조건
>     * '필드명'\_\_'lookup명' = 값
>     * lookup명 - exact, contains, in, gt, gte, lt, lte, startswith, endswith, range, year, month, day
>   * 다중 조건
>     * AND(&)
>       * & 연산자 사용 / 인자로 지정 / QuerySet 대상으로 작업 / **Q 메소드 사용**(from django.db.models import Q)
>     * OR(|)
>       * | 연산자 사용 / **Q 메소드 사용**(from django.db.models import Q)
> * get() - 모델명.objects.get(조건) or QuerySet객체명.get(조건)
>   * 하나의 값만 return 가능하기에 주로 Primary Key 값으로 조회
> * first() / last() - 모델명.objects.first() / QuerySet객체명.last()
> * count() - 모델명.objects.count() or QuerySet객체명.count()
> * exist() - QuerySet객체명.exists()
> ### 3) 추가
> * Model 인스턴스의 save()
> * Manager의 create()
> ### 4) 수정
> * Model 인스턴스의 save()
> * QuerySet의 update()
> ### 5) 삭제
> * Model 인스턴스의 delete()
> * QuerySet의 delete()

## 7. Django admin App 
> ### 1) admin App
> ### 2) Model 등록
> ### 3) 커스터마이징

## 8. Django Form
> ### 1) HTML Form
> * 헤더(요청방식+URI+HTTP 버전) + 바디
> * GET 방식
>   * 요청정보 헤더에 담겨 전달되는 방식
>   * 전달되는 질의 문자열이 노출되며, 길이에 제한이 있음
> * POST 방식
>   * 요청정보 바디에 담겨 전달되는 방식
>   * 전달되는 질의 문자열이 노출되지 않으며, 길이에 제한이 없음
> ### 2) CSRF
> * CSRF(Cross Site Request Forgery)
>   * 웹사이트 취약점 공격의 하나로, 사용자가 자신의 의지와는 무관하게 공격자가 의도한 행위(수정, 삭제, 등록)를 특정 웹사이트에게 요청하게 하는 공격   
> <img src="https://user-images.githubusercontent.com/110445149/202876256-3ee5a8ee-bb8a-4f34-8a4b-b98e12da25a0.JPG" height="300" width="400"></img>   
>   * setting.py의 MIDDLEWARE에 'django.middleware.csrf.CsrfViewMiddelware'가 자동적으로 등록되어 있음
>   * template에 {% csrf_token %} token 등록
> ### 3) HttpRequest
> * 요청정보와 응답정보   
> <img src="https://user-images.githubusercontent.com/110445149/202876405-6bcb144e-4ab7-46f6-ade5-0e16bae978cf.JPG" height="400" width="300"></img>   
> * HttpRequest 속성
>   * headers - 요청 정보의 헤더에 포함된 정보
>   * body - 요청 정보의 바디에 포함된 정보
>   * path - 요청된 페이지의 경로 정보
>   * method - 요청방식정보 ex) GET, POST
>   * GET - GET 방식으로 전달된 질의 문자열 정보
>   * POST - POST 방식으로 전달된 질의 문자열 정보
>   * FILES - 업로드된 파일 정보
>     * 파일 업로드 시, \<form\>에 enctype="multipart/form-data" 속성 반드시 삽입
>   * scheme, session, site, user 등
> ### 4) Django Form
> * 장고 Form 기능   
> <img src="https://user-images.githubusercontent.com/110445149/202876812-cb68f8bd-c880-4bd5-ac0a-2d73c80cf83d.JPG" height="300" width="250"></img>      
>   * GET 방식의 입력 페이지 응답을 위한 구현
>     * 장고 Form 클래스를 선언하고 필드 속성으로 입력 폼을 처리
>   * POST 방식의 서비스 처리 응답을 위한 구현
>     1. HttpRequest.POST 값을 장고 Form 객체에 바인딩
>     2. 장고 Form의 is_valid() 메소드를 호출하여 유효성 검사
>     3-1. is_valid() 반환값이 False이면 유효성 검사 실패로 판단하고, 오류 메시지와 함께 입력 페이지를 응답
>     3-2. is_valid() 반환값이 True이면 유효성 검사 성공으로 판단하고, 입력값들을 cleaned_data 변수에 저장하며, 이를 기반으로 서비스 처리를 진행
> ```
> from django import forms
> 
> class 클래스명(forms.Form):
>   # Form 필드
>   # 유효성 검사
> ```
> * Django Form Field
>   * forms.Form 상속하여 사용
>   * 종류
>     * BooleanField, CharField, ChoiceField, DataField, EmailFiel, FileField, ImageField, IntegerField ...
>   * Django의 Form Field를 HTML Form 태그로 변환할 때, Django Form에서 제공하는 메소드를 사용
>     * as_p() : HTML Form 태그를 \<p\>태그로 분리
>     * as_ul() : HTML Form 태그를 \<li\>태그로 분리
>     * as_table() : HTML Form 태그를 \<tr\>태그로 분리
> ### 5) URL Reverse
> * url이 변경되더라도 변경된 url을 추적하기 위한 방법
> * app의 urls.py에서 path의 name='path_name' 속성 설정
> * 이 때 settings.py의 INSTALLED_APPS에서 path의 name 속성을 순서대로 찾는데, 앱 사이 중복된 path_name으로 잘못된 접근을 막고자 각 app의 urls.py에 app_name 변수를 등록
> * URL Reverse를 실행하는 함수
>   * reverse()
>     * 리턴값 : string
>     * path 변수가 지정된 경우 args or kwargs를 통해 인자 전달
>     ```
>     reverse('app_name:path_name')
>     reverse('app_name:path_name', args=[값1, 값2, ...])
>     reverse('app_name:path_name', kwargs={키1:값1, 키2:값2, ...})
>     ```
>   * resolve_url() 
>     * 리턴값 : string
>     * 내부적으로 reverse()를 사용하며, 사용이 편함
>     * path 변수가 지정된 경우 args or kwargs를 통해 인자 전달
>     ```
>     resolve_url('app_name:path_name')
>     resolve_url('app_name:path_name', 값1, 값2, ...)
>     resolve_url('app_name:path_name', 키1=값1, 키2=값2, ...)
>     ```
>   * redirect()
>     * 리턴값 : HttpResponseRedirect(다른 페이지로 이동시켜주는 응답객체)
>     * 내부적으로 reverse_url()을 사용
>     * view 함수 내에서 특정 URL로 이동하고자 할 때 사용(HttpResponse)
>     * 주소 직접 지정 가능
>     * 인수로 모델 인스턴스 가능 -> 모델 객체의 get_absolute_url 메소드가 자동으로 호출됨
>     ```
>     redirect('URL')
>     redirect('app_name:path_name')
>     redirect('app_name:path_name', 값1, 값2, ...)
>     redirect('app_name:path_name', 키1=값1, 키2=값2, ...)
>     ```
>   * url template tag
>     * 내부적으로 reverse() 사용
>     ``` 
>     {% url 'app_name:path_name' %}
>     {% url 'app_name:path_name' 값1 값2 %}
>     ```
> * 모델 클래스 내 get_absolute_url 멤버 함수
>   * 어떠한 모델에 대해 detail View를 만들게 되면 get_absolute_url() 멤버함수를 무조건 선언!
>   * resolve_url(모델 인스턴스), redirect(모델 인스턴스)를 통해 모델 인스턴스의 get_absolute_url() 함수를 자동으로 호출
>   * 먼저 get_absolute_url 함수의 존재 여부를 확인하여, 존재하면 호출하여 그 리턴값으로 URL사용
>   * 활용법
>     * url template tag로 활용
>     * **resolve_url, redirect를 통한 활용**
>     * CBV에서 활용(?)
> ### 6) ModelForm   
> <img src="https://user-images.githubusercontent.com/110445149/202879317-c8370241-7615-45b3-b42e-a1f0e9368201.JPG" height="300" width="500"></img>   
> * ModelForm 선언
> ```
> class 클래스명(forms.ModelForm):
>   class Meta:
>     model=모델명
>     field=[필드명1, 필드명2, ...] 또는 '__all__'
> ```
> * DB에 새로운 레코드를 추가하는 방법
>   1. 모델 인스턴스를 생성 후 save() 메소드 호출
>   2. 모델명.objects.create(필드값) 메소드 호출
>   3. **ModelForm의 save() 메소드 호출**
> * 인스턴스를 바탕으로 ModelForm 호출하는 방법
>   * instance='인스턴스명' 속성 설정
>   * ``` form=PostModelForm(request.POST, instance=post) ```

## 9. Django View
> ### 1) 기본 View
> * 함수 기반 View
>   * views.py에 함수로 정의
> * 클래스 기반 View
>   * views.py에 클래스로 정의
> ### 2) Generic View
> * Classy Class-Based View [참고](https://ccbv.co.uk/)
>   * Display View
>     * ListView - 지정된 모델의 모든 인스턴스 목록을 보여주는 View
>     * DetailView - 선택한 모델 인스턴스의 자세한 내용을 보여주는 View
>   * Edit View
>     * CreateView - 지정된 Form을 출력하고 값을 입력받아 DB에 추가하는 View
>     * UpdateView - 지정된 Form을 출력하고 값을 입력받아 DB에 수정하는 View
>     * DeleteView - 특정 데이터를 DB에서 삭제하는 View
> * Generic View의 Class View 수정하는 방법 
>   * as_view() 메소드의 키워드 인자로 지정하는 방법
>   * Generic View를 상속받은 클래스에서 지정하는 방법

## 10. Django File
> ### 1) Static 파일
> ### 2) Media 파일

## 11. Django RESTful API
> ### 1) RESTful API 개요
> ### 2) Django REST Framework
> ### 3) RESTful API
>
