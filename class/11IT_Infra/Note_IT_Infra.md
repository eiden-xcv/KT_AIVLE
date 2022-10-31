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
>   * Web Application Server : Ubuntu - Apache+PHP - Application(Gnuboard Soruce) 
>   * DataBase Server : Ubuntu - MySQL Server
>   
