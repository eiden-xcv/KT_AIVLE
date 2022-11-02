# IT Virtualization & Cloud - 11/2~11/4

## 1. 가상화 및 클라우드 개요
> ### 1) 기존 환경의 문제점들
> * 원인
>   * 극도의 복잡성 & 빈약한 인프라에 의존
> * 결과
>   * 70% 이상의 IT 예산이 현상유지에 사용
>   * 30% 이하의 IT 예산만이 개선과 경쟁력 강화를 위해 사용
> ### 2) 가상화
> * 정의
>   * 운영체제에서 물리적 하드웨어를 분리하여 IT 담당자가 직면한 많은 문제에 대한 해결책을 제공하는 기술
> * 유형
>   * 서버 가상화 / 네트워크 가상화 / 스토리지 가상화 / 데스크톱 가상화
> * 가상화 기술을 통해 해결
>   * 모든 IT 자산의 가상화, 데이터 센터의 모든 리소스를 가상화
>   * 가상화 플랫폼을 이용하여 동적이고 유연한 업무 인프랄르 구축
> * 가상화 도입 효과
>   * 서비스를 위한 물리적인 서버의 대수를 감소
>   *  전체적인 상면/전력/관리 비용 절감 및 그린 IT 구현을 위한 탄소배출 절감
> * 발전
>   * 클라이언트 하이퍼바이저 > 서버 하이퍼바이저 > 가상 인프라 > 클라우드
> * 하이퍼바이저
>   * 시스템에서 다수의 운영체제를 동시에 실행할 수 있게 해주는 논리적 플랫폼
>   * Type-1(Navtive or Bare-metal) or Type-2(Hosted)
> * 물리적 리소스 공유
>   * CPU 리소스 공유 / 메모리 리소스 공유
>     * 가상 환경에서 운영체제는 시스템의 모든 물리적 자원 중 할당받은 자원만을 소유한것으로 인식
>   * 가상 네트워킹
>     * 가상 이더넷 어댑터와 가상 스위치는 하이퍼바이저가 소프트웨어적으로 구현하여 제공
> ### 3) 클라우드
> * 개념
>   * PC 데이터를 PC에 보관하는 것이 아니라 인터넷을 통해 중앙 PC 또는 서버에 저장하는 공간
>   * 클라우드 컴퓨팅 - 인터넷을 통해 IT 리소스를 원할 때 언제든지 사용하고, 사용한 만큼 비용을 지불하는 서비스
> * 유형
>   * 퍼블릭 클라우드
>     * 클라우드 컴퓨팅 서비스를 제공해주는 업체(CSP : Cloud Service Provider)에게 인프라에 필요한 자원들을 대여하여 사용하는 방식
>       * AWS, Azure, GCP, KT 클라우드, 네이버 클라우드
>   * 프라이빗 클라우드
>     * 기업이 직접 클라우드 환경을 구축, 이를 기업내부에서 활용 및 계열사에 공개
>     * 특정 기업, 사용자만 사용하는 방식
>     * 서비스 자원과 데이터는 기업의 데이터센터에 저장
>   * 하이브리드 유형
>     * 기존 On-premise(프라이빗 클라우드) 구성되어 있는 인프라와 Public Cloud를 혼용하여 함께 사용하는 방식
>   * 멀티 클라우드
>     * 2개 이상의 서로 다른 클라우드를 함께 사용하는 방식
>     * AWS + Azure / AWS + KT
>     * 하나의 CSP에 종속되지 않기 위해 사용
> * 이점
>   * 초기 선 투자 불필요 : 서비스 규모를 예측하고 미리 서버를 구매하고 관리할 필요가 없음
>   * 저렴한 종량제 가격 : 사용한 만큼 지불하는 종량제와 함께 규모의 경제로 인한 지속적인 비용 절감 가능
>   * 탄력적인 운영 및 확장 가능 : 필요한 용량을 예측할 필요없이 트래픽 만큼만 사용하거나 손쉽게 확장 가능
>   * 속도와 민첩성 : 시장 상황에 빠르게 대응할 수 있는 민첩성을 통해 비즈니스를 혁신 가능
>   * 비즈니스에만 집중 가능 : 차별화된 서비스를 개발할 수 있는 다양하고 많은 실험 시도 가능
>   * 손 쉬운 글로벌 진출 : 빠른 시간내에 손쉽게 글로벌 고객을 위한 서비스를 시작할 수 있음
> * AWS(Amazon Web Service)

## 2. AWS 기본 서비스(EC2, VPC, EBS, S3)
> ### 1) EC2 서비스(Elastic Compute Cloud)
> * Amazon EC2 : 가상 서버 서비스
>   * Virtual Machine, 재구성이 가능한 컴퓨팅 리소스
>   * 쉽게 확장/축소되는 컴퓨팅 용량
>   * 고객 업무 영역에 따른 다양한 인스턴스 타입 제공
>   * 사용한 만큼만 과금(초단위) 
> * EC2 지원 OS
>   * Windows / Amazon Linux / Debian / Suse / CentOS / Red Hat Enterprise Linux / Ubuntu / macOS / iOS / iPadOS / etc...
> * 폭 넓은 컴퓨팅 인스턴스 타입 제공
>   * 범용 / 컴퓨팅 최적화 / 스토리지&IO 최적화 / GPU 사용 / 메모리 최적화
> * 인스턴스 읽는 법 
>   * c5.large (인스턴스 종류/인스턴스 세대.인스턴스 사이즈)
> * EC2 구매 옵션
>   * On-Demand 인스턴스 / Reserved 인스턴스 / Spot 인스턴스
> * EC2 Security Group
>   * 보안그룹 규칙
>     * Name / Description / Protocol / Port range / IP address, IP range, Security Group name
>   * 특징
>     * In/Out bound 지정
>     * 모든 인터넷 프로토콜 지원
>     * 인스턴스 동작 중에도 규칙 변경 가능  
>   * 계층적인 보안그룹
>     * IP Range 대신 어느 SG로부터의 트래픽을 허용할지 지정가능
>     * 계층적인 네트워크 구조 생성 가능
> * EC2 접속 (인증)암호
>   * 표준 SSH RSA key pair
>   * EC2 key pairs
>   * AWS가 제공하는 초기 OS 접속 방법
> ### 2) VPC 서비스(Virtual Private Cloud)
> * VPC
>   * 사용자가 정의한 가상의 네트워크 환경으로 통신을 위한 기본 네트워크
>   * 보안 강화 및 부족한 IP 자원의 효율적인 관리 목적
> * VPC 생성 과정
>   * Region, IP 대역 결정 -> 가용영역(AZ)에 Subnet 생성 -> Routing 설정 -> Traffic 통제(In/Out)
>   * 1. IP Range 결정
>     * IP address group : VPC를 구성하는 가장 중요한 요소
>     * CIDR(Classless Inter-Domain Routing) : 클래스 없는 도메인 간 IP 할당 기법
>     * IP Class : Network(할당하고자하는 IP 대역 지정) + HOST(지정한 Network 내 할당 가능한 IP)
>     * Subnet Mask
>     * 특징 
>       * CIDR 숫자가 높을수록 Network 내 할당가능한 Host 개수가 줄어듦
>       * 타이트하게 IP를 관리하고 싶다면, CIDR 숫자를 높여 Network를 촘촘하게 관리
>       * 타이트하게 관리할 경우, 추후 동일 Network 대역에 IP 부족 현상 발생할 수 있음
>     * VPC CIDR 설정
>       * VPC 내 위치한 서버들이 사용할 Private IP의 범위를 지정하는 것
>       * CIDR은 VPC 생성이후 변경 불가 & VPC CIDR은 16~28 bit 사이로 설정 가능 
>     * CIDR 결정 시 고려사항
>       * 구축할 서비스 규모 / 시스템의 IP 소모량 / 추후 서비스의 확장 가능성 / 타 시스템과 연계 가능성
>   * 2. Subnet
>     * VPC의 IP 대역을 적절한 단위로 분할 사용
>     * 각 Subnet도 VPC와 마찬가지로 CIDR을 이용해 IP 범위를 지정
>     * 각 Subnet의 대역은 VPC의 대역에 존재해야하며, 중복 불가
>     * 본래 Subnetting의 주요 목적 중 하나는 Broadcasting 영역 분리이지만, AWS VPC는 Broadcast/Multicast 지원하지 않음
>     * Subnet 별로 경로를 제어하고, 원하는 트래픽만 Subent 별로 받을 수 있도록 네트워크 레벨에서 격리시키는 것이 목적
>     * Internet Gateway
>       * VPC는 기본 외부 통신 단절이기에, 외부 통신하려면 Internet Gateway를 통해야만 함
>   * 3. Routing 설정
>     * Subnet의 트래픽 경로 설정, Route 설정을 통해 Subnet의 통신 방향을 결정할 수 있음
>     * Subnet 생성시 고려사항
>       * Subnet의 CIDR은 생성 후 변경 불가
>       * Subnet의 IP 대역 중 예약된 IP 존재
>         * Subnet CIDR 영역 내 모든 IP를 사용 가능한 것이 아님
>     * Routing Table 특징
>       * VPC 생성 시, 자동으로 Main Routing Table 생성
>       * Subnet은 하나의 Routing Table과 연결될 수 있음
>       * Main Routing Table은 삭제 
>   * AWS Storage Service
>     * Block Storage
>       * 사용자의 데이터가 Local Disk 또는 SAN Storage 상의 Volume에 Block 단위로 저장 및 Access하는 스토리지 유형
>       * Amazon EBS(Elasctic Block Store)
>     * File Storage
>       * 파일 시스템으로 구성된 저장소를 Network 기반 Protocol을 사용하여 파일 단위로 Access하는 스토리지 유형(NAS)
>       * Amazon EFS(Elastic File System), FSx 
>     * Object Storage
>       * Encapsulate된 데이터 및 속성, 메타데이터, 오브젝트 ID를 저장하는 가상의 컨테이너
>       * API 기반의 데이터 접근 & 메타데이터 또는 정책에 기반한 운영 
>       * Amazon S3(Simple Storage Service), Glacier
>   * EBS(Elastic Block Store)
>     * 개념
>       * AWS에서 제공하는 Block Storage 서비스
>       * 사용이 쉽고 확장 가능한 고성능 블록 스토리지 서비스로 EC2용으로 설계
>     * 특징
>       * EC2 인스턴스를 위한 비휘발성 블록 스토리지
>       * 가상디스크 = Volume(볼륨)
>       * API 기반 볼륨 생성, 연결, 삭제
>       * 다양한 타입 지원
>       * 네트워크를 통한 연결
>         * 인스턴스 간 연결 및 해제 언제든 가능
>         * 특수한 경우를 제외하고, EBS Volume은 동시에 하나의 Instance 연결 가능
>       * 온라인 변경
>         * 디스크 추가 및 Scale up
>       * EBS 불륨과 인스턴스는 같은 Availability Zone에 있는 경우 연결 가능
>       * 인스턴스와 볼륨 연결 시 데이터 전송 속도가 중요하므로, 동일 네트워크상의 Availability Zone에 있어야 데이터 처리 속도 보장
>     * Volume Type 선택 시 중요 고려 지표
>       * Size - 데이터 저장 용량
>       * IOPS(Input/Output Per Seconds) - 데이터를 얼마나 빠르게 읽고 쓸 수 있는지에 대한 대표적인 성능 지표
>       * Throughput - 초당 얼마만큼의 데이터를 처리 가능한지에 대한 성능 지표
>       * Cost - 클라우드 사용 시 가장 중요하게 고려되야 하는 점이 바로 비용
>     * Volume Type
>       * SSD 기반 볼륨 : io2 / io2 Block Express / io1 / gp3 / gp2
>       * HDD 기반 볼륨 : st1 / sc1
>     * EBS Volume Snapshot
>       * EBS Volume을 특정 시점 기준으로 복사하여 백업하는 기능
>       * 스냅샷은 실제로는 S3에 저장
>       * 스냅샷은 마지막 스냅샷 이후 변경분만 저장되는 증분식 백업
>       * 활용
>         * EBS Volume을 Availabiliy Zone을 넘어서 복사 가능
>         * 스냅샷을 다른 Region으로 복제하면, 동일 Volume을 Region 단위로 복사하여 넘기는 것도 가능
>   * AMI(Amazon Machine Image)
>     * 인스턴스를 배포 가능한 탬플릿
>     * OS + System 서버 + Application 와 같이 묶여있는 형태
>   * S3(Simple Storage Service)
>     * 개념
>       * AW에서 제공하는 Object Storage 서비스
>       * 언제 어디서나 원하는 양의 데이터를 저장, 검색할 수 있는 객체 기반 스토리지 서비스
>     * 특징
>       * Object 스토리지 서비스
>       * 웹 서비스 기반 인터페이스 제공(REST API 기반 데이터 생성/수정/삭제)
>       * 고가용성, 무제한 용량 제곧
>       * 초기 저장 용량 확보 불필요, 강력한 보안 기능
>     * S3 Bucket
>       * Object를 저장하는 컨테이너로, Object는 하나의 Bucket에 속해야 함
>       * Bucket에 저장할 수 있는 Object는 무제한
>     * S3 Object & Key
>       * Object는 S3에 저장되는 기본 개체로, 하나 최대 크기는 5TB
>       * Object는 데이터와 메타데이터로 구성되어 있음
>         * 메타데이터 : Object를 설명하는 이름-값 쌍
>         * 기본 메타데이터 및 Content-Type 같은 HTTP 메타데이터 포함
>       * Key 및 Version ID를 통해 버킷 내 고유 식별
>       * Key는 Bucket 내 Object에 대한 고유한 식별자







