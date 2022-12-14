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
>   * 가상화 플랫폼을 이용하여 동적이고 유연한 업무 인프 구축
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
>     * Subnet 생성 시 고려사항
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
>       * AWS에서 제공하는 Object Storage 서비스
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

## 3. AWS 고가용성 구현
> * 가용성
>   * 워크로드를 사용할 수 있는 시간의 비율로, 서비스 가용성이라고도 함
> * 고가용성(High Availability)
>   * 높은 가용성으로, 지속적으로 구현한 시스템이 정상적으로 운영이 되는 성질
>   * 장애 또는 고장이 나더라도 복구를 해서 서비스를 지속할 수 있는 능력
> ### 1) Region & Availability Zone
> * AWS는 Region과 Availability Zone으로 이루어져있음(Region안에 여러 AZ가 있음)
> * Region
>   * 전 세계에서 데이터센털르 클러스터링 하는 물리적 위치
>   * Resource는 Region내 AZ 단위로 배포
> * Availability Zone
>   * Region 내 물리적으로 분리된 전력 네트워킹 장치가 분리된 영역
>   * 보통 AZ 별 데이터센터 분리된 구조
> * 구성
>   * Region은 보통 2~3개의 AZ로 구성
>   * 동일 Region 내 AZ는 전용 광 네트워크로 구성되어 매우 낮은 지연 속도와 높은 처리 처리량 보장
>   * AZ간 모든 데이터 트래픽은 기본 암호화
> * AZ 분산 배치
>   * 동일 역할을 수행하는 인스턴스의 경우, AZ를 분산 배치하여 서비스 가용성을 높이는 것이 좋음
> * AZ & VPC
>   * Region - VPC와 맵핑
>   * AZ - Subnet과 맵핑
>   * Instance 생성 시 VPC와 Subnet을 선택하여 배포
> * VPC 구성
>   * Public Subnet & Private Subnet
>     * VPC 구성 시, 목적에 따라 Subnet을 구분하여 생성
>       * Public(외부 통신용) & Private(Public과 Private간 연동용)
>     * 외부 통신 시 NAT Gateway를 통한 단방향 허용
>   * AZ별 Subnet 구성
>     * VPC 구성 시, AZ에 따라 Subnet 구성
>       * 각각의 Subnet을 AZ 수 만큼 생성
>       * 총 Subnet 수 = AZ Count * 용도별 Subnet
> * ELB(Elastic Load Balancer)
>   * Load Balancer
>     * 인입되는 트래픽을 특정 알고리즘 기반으로 다수의 서버로 분산 시켜주는 장비
>   * 특징
>     * Region 내 인스턴스 및 다양한 서비스로 트래픽 분배 서비스
>     * 다수의 AZ로 트래픽 분배
>     * HTTP/S 웹 기반 트래픽, TCP/S 프로토콜 기반
>     * Backend 인스턴스에 대한 Health Check 수행
>     * 고가용성 기반 L4/L7 서비스
>     * AZ 분산 및 Traffic 증가 시 자동 Scale-out 기능 지원
>   * ELB 4 Type
>     * ALB / NLB / GLB / CLB
>       * NLB - L4 Load Balancer, TCP/UDP
>       * ALB - L7 Load Balancer, HTTP/S
>   * Scale-out
>     * 트래픽 증가 시, 서비스에 투입되는 서버를 증설하여 각 서버가 처리하는 부하를 낮추는 방식
>     * Web basedt 서비스의 경우 많이 사용하는 구성으로 Session이나 Data 처리 영역 없이 Stateless한 서버에서 주로 사용
>   * Scale-in
>     * 트래픽 감소 시, 배포된 서버를 제거하는 방식
>     * 낭비되는 리소스를 줄임으로 비용 최적화 목적
>   * ELB 알고리즘
>     * 어떤 규칙으로 트래픽을 인스턴스로 분배할 것인가
>     * Round Robin / Hashing / Weighted RR / Least Connection / Weighted LC
>   * ELB 헬스체크 기능
>     * 주기적으로 서버가 정상 상태인지 확인하고 정상상태가 아닌 서버에게는 트래픽을 전달하지 않게하는 기능
>   * ELB AZ 분산배치
>     * 활성화된 AZ에는 LB node가 자동으로 생성되어 배치
>     * 기본적으로 해당 AZ에 배치된 타겟(Instance)는 해당 AZ의 LB node가 트래픽을 처리
> * ASG(Auto Scaling Group)
>   * Scaling을 자동으로 해줌
>   * Auto Scaling 대상
>     * Launch Template
>       * 인스턴스를 배포하기 위한 정보들의 묶음
>       * AMI, Instance Type, Keypair, Security Group, Network와 같은 Instance에 대한 정보
>       * IAM Role, Userdata, Tags 등 추가 정보를 미리 Template로 정의 가능
>       * 사용자는 해당 Template을 그대로 인스턴스로 배포하는데 
>   * 자동 설정 정책 설정
>     * ASG : Desired Capacity, Min/Max Size, Target Group 등 자동 확장에 대한 정의

## 4. 쿠버네티스 개요 및 주요 아키텍쳐
> * 컨테이너
>   * OS 가상화 기술
>   * 프로세스 격리
>   * 리눅스 커널 공유
> * 가상머신과의 차이 - Guest OS의 유무
>   * 가상머신 - Infrastructure - Hypervisor - (Guest OS - APP)s
>   * 컨테이너 - Infrastructure - OS - Container Engine - APPs

|구분|가상머신|컨테이너|
|:---:|:---:|:---:|
|Guest OS|Windows, Linux ... | x |
|시작시간|길다|짧다|
|이미지 사이즈|크다|작다|
|환경 관리|각 VM마다 OS 패치 필요|호스트 OS만 패치|
|데이터 관리|VM 내부 또는 연결된 스토리지에 저장|컨테이너 내부의 데이터는 컨테이너 종료 시 소멸, 필요시 스토리지를 이용하여 저장|

> * Monolithic vs Micro Service
>   * Monolithic Architecture
>     * 고용량 고성능의 단일 서버로 구성
>   * MicroService Architecture
>     * Monolithic Architecture와 비교하여 작은 서버들의 집합체로 구성
> ### 1) Docker
> * Docker
>   * 컨테이너 엔진, 컨테이너 기반의 오픈소스 가상화 플랫폼
>   * 도커는 도커허브라는 공개된 저장소 서버를 통해, 컨테이너 자료들을 관리
>   * 컨테이너를 생성하고 실행하기 위해서는 Dockerfile과 Image가 필요
> * Dockerfile
>   * 컨테이너 이미지를 생성하기 위한 레시피 파일
>   * 이 파일에 이미지 생성과정을 무넙에 따라 작성하여 저장
> * Docker Image
>   * 서비스 운영에 필욯나 프로그램, 소스코드, 라이브러리 등을 묶는 형태
>   * 도커 이미지는 Dokcerfile을 사용하여 생성할 수 있음(Build)
>   * 도커 이미지를 사용하여 다수의 Container를 실행할 수 있음(Run)
>   * Dockerfile -build-> Image -run-> Container
>   * 경로
>     * url 방식으로 관리하고 태그를 붙일 수 있음
>     * 형식
>       * (Namespace)/(ImageName):(Tag) == 저장소/이름:Tag(version)
> * Docker HUB
>   * 수많은 컨테이너 이미지들을 서버에 저장하고 관리
>   * 공개 이미지를 무료로 관리
> * 컨테이너 오케스트레이터
>   * 컨테이너 오케스트레이션을 해주는 도구
>   * Kubernetes, Docker Swarm, AWS ECS...
>   * 컨테이너 오케스트레이션
>     * 다수의 컨테이너를, 다수의 시스템에서, 각각의 목적에 따라, 배포/복제/장애복구 등 총괄적으로 관리하는 것
>     * 기능 - 스케쥴링 / 자동확장 및 축소 / 장애복구 / 로깅 및 모니터링 / 검색 및 통신 / 업데이트 및 롤백
>     * 배포위치 - 베어 메탈 / 가상머신/ 온프레미스 / 클라우드
> ### 2) Kubernetes
> * Kubernetes
>   * 컨테이너형 애플리케이션의 배포, 확장, 관리를 자동화하는 오픈 소스 시스템
>   * 장점
>     * 높은 확장성, 원활한 이동성(이식성)
>     * 퍼블릭/프라이빗/하이브리드/멀티 클라우드, 로컬 또는 원격 가상머신, 베어메탈과 같은 여러 환경에 구축 가능
>     * 오픈 소스 도구의 장점, 플러그가 가능한 모듈 형식
> * 아키텍처   
> <img src="https://user-images.githubusercontent.com/110445149/199664785-da3a8bce-c7c9-419b-82ed-bfbc2b4589c9.JPG" height="300" width="500"></img>   
>   * Cluster = Master Node(control plane) + Worker Node
>   * Master Node
>     * API Server(api) - API를 사용할 수 있게 해주는 프로세스, 각 구성요소 간 통신
>     * Scheduler(sched) - Pod의 생성 명령이 있을 경우 어떤 Node에 배포할 지 결정
>     * Controller Managers(c-m) - 클러스터의 상태를 조절하는 컨트롤러들의 생성 및 배포
>     * etcd - 모든 클러스터의 구성 데이터를 저장하는 저장소
>   * Worker Node
>     * Container Runtime - 컨테이너(Pod)를 실행하고 노드에서 컨테이너 이미지를 관리
>     * kubelet - 각 Node의 에이전트
>     * kube-proxy(k-proxy) - 쿠버네티스 클러스터의 각 노드마다 실행되고 있으면서, 각 노드 간의 통신을 담당
>   * Addons
>     * Kubernetes에서 추가적으로 설치하여 Kubernetes의 기능을 확장시킬수 있는 도구
>     * DNS, Dashboard, Monitoring, Logging ...

## 5. 쿠버네티스 클러스터 배포
> * 배포유형
>   * All-in-One Single-Node Installation
>   * Single-Node etcd, Single-Master and Multi-Worker Installation
>   * Single-Node etcd, Multi-Master and Multi-Worker Installation
>   * Mulit-Node etcd, Multi-Master and Multi-Worker Installation
> * 설치도구
>   * kubeadm / kubespray / kops
> * Kubernetes 배포 순서
>   1. Container Runtime 설치
>   2. Kubernetes 설치
>   3. Master와 Worker 연동
>   * 환경 안내
>     * AWS의 Cloud9 서비스 & AWS의 EC2 인스턴스 서비스

## 6. 쿠버네티스 클러스터 배포, 통신, 볼륨관리
> * Kubernetes Object
>   * 가장 기본적인 구성단위로, 상태를 관리하는 역할
>   * 가장 기본적인 오브젝트
>     * Pod, Service, Volume, Namespace
>   * 오브젝트의 Spec, Status 필드
>     * Spec : 정의된 상태 / Status : 현재 상태
> * Kubernetes Controller
>   * 클러스터의 상태를 관찰하고, 필요한 경우 오브젝트를 생성, 변경을 요청하는 역할
>   * 각 컨트롤러는 현재 상태를 정의된 상태에 가깝게 유지하려는 특징
>     * Deployment, Replicaset, Daemonset, Job, CronJob ...
>   * Controller Cycle
>     * 관찰(Current State) -> 상태 변동(Current State != Desired State) -> 조치 (Current State <- Desired State)
>   * Auto Healing & Auto Scaling & Update & Rollback & Job
> * YAML 구조
>   * apiVersion : 연결할 API server의 버전
>   * kind : 리소스의 유형
>   * metadata : 리소스가 기본 정보를 갖고 있는 필드로, name, label, namespace 등
>   * spec : 배포되는 리소스의 원하는 상태
> * kubectl
>   * Kubernetes에 명령을 내리는 CLI
>   * 오브젝와 컨트롤러를 생성, 수정, 삭제
>   * 명령 구조
>     * kubectl [COMMAND] [TYPE] [NAME] [FLAGS]]
> * Pod
>   * Kubernetes의 가장 작은, 최소 단위 Object
>   * 하나 이상의 컨테이너 그룹, 네트워크와 볼륨을 공유
> * Template
>   * Pod를 생성하기 위한 명세
>   * Deployment, ReplicaSet과 같은 Controller의 yaml 내용에 포함
>   * Template에는 Pod 세부사항을 경정 
> * Kubernetes Object - Namespace
>   * 단일 클러스트 내 리소스 그룹 격리를 위한 오브젝트
>   * 사용자가 여러 팀으로 구성하는 경우, 프로젝트를 진행함에 있어 환경을 분리해야하는 경우 사용
> * Kubernetes Controller - ReplicaSet
>   * ReplicaSet은 Pod의 개수를 유지
>   * yaml을 작성할 때 replica 개수를 지정하면 그 개수에 따라 유지
> * Kubernetes Controller - Deployment
>   * ReplicaSet을 관리하며 애플리케이션의 배포를 더욱 세밀하게 관리(Pod의 개수도 유지)
>   * 초기 배포 이후에 버전 업데이트, 이전 버전으로도 Rollback도 가능
> * Deployment Update
>   * 운영중인 서비스의 업데이트 시 재배포를 관리
>   * 2가지 재배포 방식
>     * Recreate : 현재 운영중인 Pod들을 삭제하고, 업데이트 된 Pod들을 생성, Downtime이 발생하기에 실시간으로 사용해야한다면 권장되지 않음
>     * Rolling Update : 먼저 업데이트된 Pod를 하나 생성하고 구버전의 Pod를 삭제하여, Downtime없이 업데이트 가능
> * Deployment Rollback
>   * Deployment 이전버전의 ReplicaSet을 10개까지 저장
>   * 저장된 이전 버전의 ReplicaSet을 활용하여 Rollback
> * Kubernetes Object - Service
>   * Pod에 접근하기 위해 사용하는 Object, 고정된 주소를 이용하여 접근 가능
>   * Kubernetes 외부 또는 내부에서 Pod에 접근할 때 필요
>   * Pod에 실행중인 애플리케이션을 네트워크 서비스로 노출시키는 Object
>   * 유형
>     * ClusterIP(default) : Service가 기본적으로 갖고있는 ClusterIP를 활용하는 방식
>     * NodePort : 모든 Node에 Port를 할당하여 접근하는 방식
>     * Load Balancer : Load Balancer Plugin을 설치하여 접근하는 방식
> * Label : Pod와 같은 Object에 첨부된 키와 값 쌍
> * Selector : 특정 Label값을 찾아 해당하는 Object만 관리할 수 있게 연결
> * annotation : Object를 식별하고 선택하는 데에는 사용되지 않으나 참조할 만한 내용들을 Laebl처럼 첨부
> * Kubernetes DNS
>   * Kubernetes는 Pod와 Service에 DNS 레코드를 생성
>   * IP대신, 이 DNS를 활용하여 접근 가능
> * Volume
>   * Pod 컨테이너에서 접근할 수 있는 디렉터리
>   * 유형
>     * EmptyDir : Pod 생성될때 함께 생성되고, 삭제될때 함께 삭제되는 임시 Volume
>     * HostPath : 호스트 노드의 경로를 Pod에 마운트하여 함께 사용하는 유형의 Volume
>     * PV/PVC
>       * PV(Persistent Volume) 
>         * Volume 자체를 의미, 클러스터 내부에서 Object처럼 관리 가능, Pod와는 별도로 관리
>         * 클라우드 서비스에서 제공해주는 Volume 서비스를 이용할 수도 있고, 사설에 직접 구축되어있는 스토리지를 사용가능
>         * Pod에 직접 연결하지 않고 PVC를 통해 사용
>       * PVC(Persistent Volume Claim)
>         * 사용자가 PV에 하는 요청, Pod와 PV의 중간 다리역할
