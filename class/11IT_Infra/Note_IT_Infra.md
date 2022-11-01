# IT Infra - 10/31~11/1

## 0. IT Infra
> ### 1) Storage
> * Magnetic Tape Drive
>   * 장점 : 용량당 단가 매우 낮음
>   * 단점 : 입출력 속도 매우 느림 & 동작 속도 느림
> * Tape Storage Automatic System
>   * 저장 및 보관하기 위해 라벨링을 해야함
>   * Barcode 방식을 활용해서 긴 자리수 기록 가능
> * HDD(기계식) vs SSD(전자식)
> * Main Board
> * Network Devices
>   * Routers, Firewalls, Hub, Layer 2 Swtiches, Multi-Layer Switches, Bridges, Repeater, Modems
> * DataBase

## 1. Web
> ### 1) Web, WAS, DB 이해
> * Client vs Server   
>
|Client|Server|DB|
|:---:|:---:|:---:|
|웹 브라우저|Apache2|MySQL|
|Putty|Nignix|오라클DB|
|AL FTP|IIS|PostgreSQL|
|-|-|MS-SQL|
> * 2 Tier
>   * Client - ( Server + DB )
>   * 장점
>     * 소규모 네트워크에 적합
>     * 응답이 빠르며, 구조가 간단하고, 관리가 편함
> * 3 Tier
>   * Client - Server(WAS) - DB
>   * 장점
>     * 오류 발생에 대한 대응 용이
>     * 부하의 분산
>     * 웹서버와 DB의 다른 보안 적용
>   * 인터넷 뱅킹 구조
>     * 인터넷 뱅킹 웹서버에 접속해서 로그인하고 계좌를 조회하면 계좌 정보는 웹서버가 DB서버에게 조회하는 구조
>     * 클라이언트는 DB서버에 직접 접근이 불가능

## 2. Web Server
> ### 1) Linux, MySQL-Server 설치
> * Web Server 구축
>   * 3 Tier
>   * Client : Windows 10 - Web Browser
>   * Web Application Server : Ubuntu - Apache+PHP - Application(Gnuboard Soruce, php) 
>   * DataBase Server : Ubuntu - MySQL Server

## 3. Tomcat
> ### 1) Apache & Tomcat
> * Apache
>   * 1995년 처음 발표된 www 서버용 소프트웨어
>   * 대부분의 운영체제에서 운용이 가능하며 오픈소스 라이선스 자유롭게 사용 가능
>   * 가정 널리 쓰이는 웹 서버로, 현재는 Apache2를 일반적으로 사용
> * Tomcat
>   * 아파치 소프트웨어 재단에서 개발한 섭르릿 컨테이너만 있는 웹 애플리케이션 서버
>   * WAS라고 말하는데, 이는 웹 서버와 웹 컨테이너의 결합으로 다양한 역할을 수행할 수 있는 서버 ( WebLogic, Jeus, Tomcat 등)
>   * 클라이언트의 요청이 들어오면 내부의 실행 결과를 만들어내고 이를 다시 전달해주는 역할(상대적으로 느림)
> * Apache와 Tomcat의 연동
>   * Client <-> [ WAS (Webserver(정적 데이터 처리) <-> Web Container(동적 데이터 처리) ]
>   * 만일 웹 서버 없이 WAS만 사용하는 경우
>     * 웹페이지에는 정적 리소스와 동적 리소스가 함께 존재하는데, 정적 데이터는 빠르게 응답가능하지만 동적 데이터는 처리시간이 오래 걸림
>     * WAS의 정적 데이터 처리로 인해 동적 데이터에 대한 처리는 늦어지게 되고, 클라이언트의 요청에 대한 응답 시간은 전반적으로 늘어나게 됨
>   * HTML 파일이나 이미지 파일과 같은 정적 컨텐츠들은 WAS까지 거치는 것보다 웹 서버를 바로 통한느 것이 빠름
>   * 하나의 웹 서버에 여러 개의 톰캣을 연결해서 분산시킬 수 있는 Load Balancing 구현 가능
