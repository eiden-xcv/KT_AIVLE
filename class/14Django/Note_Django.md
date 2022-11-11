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
> * 장고 처리 구조   
> <img src="https://user-images.githubusercontent.com/110445149/201237751-bec0a1b9-99c2-43ab-9dd8-4e0c3ee61b63.JPG" height="300" width="300"></img>

## 2. Django Project
> ### 1) Django Project & APP
> * Project
>   * Django에서 웹 사이트 또는 웹 애플리케이션 프로젝트라 부름
> * App
>   * 웹 사이트에는 사용자의 서비스를 처리하는데 제공하는 기능을 앱이라 부름
> * manage.py
>   * 현재 개발 중인 Django 프로젝트의 개발 과정에서 필요한 작업을 실행시켜주는 커멘트 유틸리티

절차 : 프로젝트 생성 -> 앱생성 -> 앱등록

## 3. Django Model
절차 : DB 설정 -> Model
ORM
migrate :
  makemigrate : class -> sql & migrate : sql -> db
models.Model == Table
p1=Post() -> p1.save() == INSERT
