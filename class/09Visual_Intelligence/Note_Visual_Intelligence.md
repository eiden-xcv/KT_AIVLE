# Visual Intellgence - 9/19~9/23

## 1. Review
> ### 1) Batch Normalization
> * Internal Covariate Shift
>   * Covariate(공변량) : 독립변수 이외에 종속변수에 영향을 줄 수 있는 잡음인자(변수)
>   * Covariate Shift : train data와 test data의 distribution이 다른 현상
>   * Internal Covariate Shift : 가중치를 학습 시 이전 레이어의 값에 따라 다음 레이어의 가중치가 영향을 받는다. 이때 규제없이 가중치를 학습하다보면 매개변수 값의 범위가 넓어질 수 있는데, 이렇게 매개변수의 값들의 변동이 심해지는 현상
> * 2015년 Sergey loffe & Christian Szegedy의 논문 Batch Normalization : Accelerating Deep Network Training by Reducing Internal Covariate Shift에서 그레디언트 소실과 폭주 문제를 해결하기 위한 배치 정규화(Batch Normalization)기법 제안
> * 단순히 입력을 원점에 맞추고 정규화한 다음, 각 층에서 두개의 새로운 파라미터로 결과값의 스케일을 조정하고 이동시킴
> * 배치 정규화 알고리즘
>   * <img src="https://user-images.githubusercontent.com/110445149/191153342-925dba2e-0f31-43d2-9d21-638287896d9d.JPG" height="250" width="300"></img>
>   * 미니배치B에 대해 평가한 입력의 평균벡터, 표준편차벡터
>   * 3번은 평균이 0이고 정규화된 샘플i의 입력
>   * gamma는 층의 출력 스케일 파라미터 벡터, beta는 층의 출력 이동 파라미터 벡터
>   * 4번은 배치 정규화 연산의 출력, 즉 입력의 스케일을 조정하고 이동시킨 것
> * 장점 : 배치 정규화는 규제와 같은 역할을 하여 다른 규제 기법의 필요성을 줄여줌
>   * 배치 정규화는 전체 데이터셋이 아니고 미니배치마다 평균과 표준편차를 계산하므로 훈련 데이터에 일종의 잡음을 넣는 것으로 볼 수 있고, 이 잡음은 훈련 세트에 과대적합되는 것을 방지하는 규제의 효과를 가짐
> * 단점 : 모델의 복잡도를 키우며, 층마다 추가되는 계산이 신경망의 예측을 느리게 하기에 실행 시간 면에서도 손해

> ### 2) 규제를 통한 과적합 방지
> * L1 규제 & L2 규제
>   *
>    ``` 
>   layer = keras.layers.Dense(256, activation='elu', kernel_initailizer='he_normal', 
>                                 kernel_regularizer=keras.regularizer.l2(0.01))
>   ```
>   * keras.regularizer.l1(), keras.regularizer.l2(), keras.regularizer.l1_l2()
> * Dropout
>   * 매 훈련 스텝에서 각 뉴런은 임시적으로 드롭아웃될 학률 p를 가지는데, 이번 훈련 스텝에는 완전히 무시되지만 다음 스텝에서는 활성화 될 수 있음
>   * 각 훈련 스텝에서 고유한 네트워크가 생성된다고 생각할 수 있으며, 결과적으로 만들어진 신경망은 이 모든 신경망을 평균한 앙상블로 볼 수 있음
>   * Dropout은 훈련하는 동안에만 활성화 됨
>   * ``` layer=keras.layers.Dropout(rate=0.2) ```
> * Max-Norm 규제
>   *
>    ``` 
>   layer = keras.layers.Dense(256, activation='elu', kernel_initailizer='he_normal', 
>                                 kernel_constraint=keras.constraints.max_nomr(1.))
>   ```

## 2. CNN Basics
> ### 1) Convolution Layer
> * 입력이미지에 필터를 적용하여 새로운 feature map을 만드는 층
> * filter 수, 크기, stride, padding 설정
> ### 2) Pooling Layer
> * 계산량과 메모리 사용량, 과적합 위험을 줄여주기 위한 파라미터 수를 줄이기 위해 입력이미지의 부표본을 만드는 층
> * MaxPool2D, AvgPool2D

## 3. Object Detection
> ### 1) Object Detection
> * Classification(분류) + Localization(위치) = Multi-Labeled Classification + Bounding Box Regression
> * Segmentation
>   * Object Detection 보다 발전된 형태로, pixel 단위로 detection 수행
>     * Semantic Segmentation : 같은 class의 object는 같은 영역/색으로 표현
>     * Instance Segmentation : 같은 class의 object여도 서로 다른 영역/색으로 표현
> * Dataset 
>   * Pascal VOC Dataset
>   * COCO Dataset
>   * Open Images Dataset
>   * Udacity Dataset
>   * KITTI-360 Dataset
> * History
>   * Two-stage Object Detection
>     * Faster R-CNN
>   * One-stage Object detection
>     * YOLO v1(15.06)
>     * SSD(15.12)
>     * YOLO v2(16.12)
>     * RetinaNet(17.08)
>     * YOLO v3(18.04)
>     * YOLO v4(20.04)
>     * YOLO v5(20.06)
> * 구성요소
>   * Bounding Box : 하나의 object가 포함된 최소 크기의 box (x, y, w, h 포함)
>   * Class Classification
>   * Confidence Score : Object가 Bounding Box안에 있는지에 대한 확신의 정도
> * YOLO 논문정리
>   * Sketch에... :)
> * Metrics
>   * IoU(Intersection over Union)
>     * Ground-truth Bounding Box와 Prediction Bounding Box에 대하여 연산
>     * = Area of Intersection / Area of Union
>   * Counfusion Matrix with Object Detection
>     * TP : 실제 Object를 모델이 Object라 예측 -> IoU >= threshold
>     * FP : Object가 아닌데 모델이 Object라 예측 -> IoU < threshold
>     * FN : 실제 Object를 모델이 Object가 아니라 예측 -> 모델이 예측을 못함
>     * TN : Object가 아닌데 모델이 Object가 아니라 예측 -> 모델이 예측을 못함
>     * Precsion : TP / (TP+FP) -> 모델이 Object라 예측한 것 중 실제 Object의 비율
>     * Recall : TP / (TP+FN) -> 실제 Object 중 모델이 정확히 예측한 Object이 비율
>     * **Precision - Recall Curve**
>       * **AP(Average Precision)** : Precision-Recall Curve 그래프 면적
>       * **mAP(mean AP)** : 각 클래스 별 AP를 합산하여 평균을 낸 것
>     * **설정한 threshold 값에 따라 Precision과 Recall이 변함!!**

## 4. UltraLytics package
> * DarkNet Framework기반 YOLO v3를 pytorch로 변환 & YOLO v5 개발
> ### 1) Pretrained YOLO 모델 사용해보기
> * 과정
>  * 0. UltraLytics github에서 yolov3 다운받기
>  * 1. Pretrained weights 다운받기
>  * 2. detect.py 실행하기
>    ```
>    !cd yolov3; python detect.py \
>      --weights 'yolov3.pt 경로' \ # pretrained 가중치
>      --source 'detect할 이미지 경로' \
>      --project 'detect 후 이미지 저장할 경로' \
>      --name '저장폴더명 지정' \
>      --img 640 \ # 이미지 크기
>      --conf-thres 0.5 \ # confidence threshold
>      --iou-thres 0.5 \ # NMS IoU threshold
>      --line-thickness 2 \ 
>      --exist-ok \ # existing project/name ok, do not increment (덮어쓰기)
>      --device DEVICEE # cuda device, i.e. 0,1,2,3 or cpu
>    ```
>  * 3. detected image 확인하기
>    ```
>    from IPython.display import Image
>    from google.colab import files
>
>    Image(filename='경로', width=640)
>    ```
> ### 2) Custom Data를 사용하여 train하고 평가하기
> * 과정
>   * 0. UltraLytics github에서 yolov5 다운받기
>   * 1. Custom_Data.yaml 만들기 
>   * 2. Pretrained weights 다운받기
>   * 3. train.py 실행하기
>     ```
>     !cd yolov5; python train.py \
>       --data 'Custom_Data.yaml 경로' \
>       --cfg 'model.yaml 경로' \
>       --weights 'yolov5.pt 경로' \ # pretrained 가중치
>       --epochs 1000 \
>       --patinece 5 \
>       --img 640 \
>       --project 'train 저장 경로' \
>       --name '저장폴더명' \
>       --exist-ok \ # existing project/name ok, do not increment (덮어쓰기)
>       --device Device # cuda device, i.e. 0,1,2,3 or cpu
>     ```
>   * 4. detect.py 실행하기
>   * 5. detected image 확인하기
>  
