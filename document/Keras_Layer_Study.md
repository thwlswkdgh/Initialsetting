# Keras Layer

https://techvidvan.com/tutorials/keras-layers/



# Layer 소개 

### Dense, Flatten, Convolution, Max Pooling, LSTM

https://ssongnote.tistory.com/13?category=727139



# 신경망 구조

<img src="C:\Users\thwls\AppData\Roaming\Typora\typora-user-images\image-20210725175929883.png" alt="image-20210725175929883" style="zoom:50%;" />

앞의 layer에서 다음 layer로 값을 전달 할때는 **활성화 함수**를 거쳐 출력된 값을 전달한다.

* 활성화 함수: 입력 신호의 총합을 출력 신호로 변화하는 함수로,  입력 받은 신호를 얼마나 출력할지 결정, Network에 층을 쌓아 비선형성을 표현 할 수 있도록 해준다.

  > 1. Step function
  >
  > <img src="C:\Users\thwls\AppData\Roaming\Typora\typora-user-images\image-20210725180653599.png" alt="image-20210725180653599" style="zoom:70%;" />
  >
  > 2. Sigmoid function
  >
  >    <img src="C:\Users\thwls\AppData\Roaming\Typora\typora-user-images\image-20210725180811206.png" alt="image-20210725180811206" style="zoom:70%;" />
  >
  >    - 비교
  >
  >      <img src="C:\Users\thwls\AppData\Roaming\Typora\typora-user-images\image-20210725180851860.png" alt="image-20210725180851860" style="zoom:70%;" />
  >
  >      step function은 0,1을 출력했다면, sigmoid를 사용하는 모델은 연속적인 값을 전달 , sigmoid 함수가 계단 함수에 비해 더 많은 정보를 전달 할 수 있다.

출처: https://leedakyeong.tistory.com/entry/%EB%B0%91%EB%B0%94%EB%8B%A5%EB%B6%80%ED%84%B0-%EC%8B%9C%EC%9E%91%ED%95%98%EB%8A%94-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98%EB%9E%80-What-is-activation-function

- 활성화 함수 자세히 : https://yeomko.tistory.com/39



### Batch Normalization

학습하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속 시킬 수 있는 근본적인 방법

**신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정** 

즉, 각 레이어마다 정규화 하는 레이어를 두어, 변형된 분포가 나오지 않도록 조절하게 하는 것이 배치 정규화이다. 

출처: https://eehoeskrap.tistory.com/430 [Enough is not enough]



## Sequential Model 구현

```
model.add(Dense(5, input_dim=1, activation='relu'))
```

**Dense의 첫번째 인자** : 출력 뉴런(노드)의 수를 결정

**Dense의 두번째 인자** : input_dim은 입력 뉴런(노드)의 수를 결정, 맨 처음 입력층에서만 사용

**Dense의 세번째 인자** : activation 활성화 함수를 선택

- 활성화 함수 종류

| relu    | 은닉 층으로 학습'relu' 는 은닉층으로 역전파를 통해 좋은 성능이 나오기 때문에 마지막 층이 아니고서야 거의 relu 를 이용한다. |
| ------- | :----------------------------------------------------------- |
| sigmond | yes or no 와 같은 이진 분류 문제                             |
| softmax | 확률 값을 이용해 다양한 클래스를 분류하기 위한 문데          |

출처: https://ebbnflow.tistory.com/120 [Dev Log : 삶은 확률의 구름]





#### Keras 모델을 컴파일 하기 위해 필요한 두개의 매개변수

```
self.Discriminator.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.0002, decay=8e-9)
```

1. **loss**는 손실함수를 의미합니다. 얼마나 입력데이터가 출력데이터와 일치하는지 평가해주는 함수

2. **optimizer**는 학습 속도를 빠르고 안정적이게 하는 것

   <img src="https://user-images.githubusercontent.com/30134043/126892775-d42430cd-a46c-48a8-a468-ebc1430991b2.png" alt="image" style="zoom:70%;" />

   

출처:https://gomguard.tistory.com/187



### Adam optimizer

```
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

**인자**

- **lr**: 0보다 크거나 같은 float 값. 학습률. 
- **beta_1**: 0보다 크고 1보다 작은 float 값. 일반적으로 1에 가깝게 설정됩니다.
- **beta_2**: 0보다 크고 1보다 작은 float 값. 일반적으로 1에 가깝게 설정됩니다.
- **epsilon**: 0보다 크거나 같은 float형 fuzz factor. `None`인 경우 `K.epsilon()`이 사용됩니다.
- **decay**: 0보다 크거나 같은 float 값. 업데이트마다 적용되는 학습률의 감소율입니다.
- **amsgrad**: 불리언. Adam의 변형인 AMSGrad의 적용 여부를 설정합니다. AMSGrad는 "On the Convergence of Adam and Beyond" 논문에서 소개되었습니다.

출처:https://keras.io/ko/optimizers/

> - lr  (Learning Rate) 학습률
>
>   <img src="https://user-images.githubusercontent.com/30134043/126902985-61f498b0-8dac-446d-956d-bf7224c01db3.png" alt="image" style="zoom:80%;" />
>
>   * 최대한 틀리지 않게 > loss 를 최소화
>
>   * 세로축은 loss function을 의미
>
>   * 결국 Optimizer는 loss function의 최소값을 찾는 것을 목표로 함!
>
>     Optimizer 정리 site: https://ganghee-lee.tistory.com/24 

> <img src="https://user-images.githubusercontent.com/30134043/126903027-3a4e2852-d86c-4af2-88c1-c3f895af13a4.png" alt="image" style="zoom:50%;" />
>
> 출처: https://proggg.tistory.com/23

>  decay  : 학습률의 감소율
>
> <img src="https://user-images.githubusercontent.com/30134043/126903169-f715fe4b-f335-4c18-8954-583a91c4381d.png" alt="image" style="zoom:50%;" />
>
> 학습률이 일정한 첫 번째 이미지에서 최소값을 향해 반복하는 동안 알고리즘이 수행하는 단계는 너무 시끄러워서 특정 반복 후에는 최소값 주위를 돌아 다니는 것처럼 보이며 실제로 수렴하지 않습니다.
>
> 그러나 시간이 지남에 따라 학습률이 감소하는 두 번째 이미지 (녹색 선으로 표시됨)에서 처음에는 학습률이 크므로 학습 속도가 상대적으로 빠르지 만 최소 학습률을 향한 경향이 점점 더 작아지면서 더 타이트하게 진동하게됩니다. 멀리 방황하지 않고 최소 주변 지역.
>
> 출처: https://ichi.pro/ko/dib-leoning-ui-hagseublyul-gamso-mich-bangbeob-49402161893643

** **



### **1) 에포크(Epoch)**

에포크란 인공 신경망에서 전체 데이터에 대해서 순전파와 역전파가 끝난 상태를 말합니다. 전체 데이터를 하나의 문제지에 비유한다면 문제지의 모든 문제를 끝까지 다 풀고, 정답지로 채점을 하여 문제지에 대한 공부를 한 번 끝낸 상태를 말합니다.

만약 에포크가 50이라고 하면, 전체 데이터 단위로는 총 50번 학습합니다. 문제지에 비유하면 문제지를 50번 푼 셈입니다. 이 에포크 횟수가 지나치거나 너무 적으면 앞서 배운 과적합과 과소적합이 발생할 수 있습니다.

### **2) 배치 크기(Batch size)**

배치 크기는 몇 개의 데이터 단위로 매개변수를 업데이트 하는지를 말합니다. 현실에 비유하면 문제지에서 몇 개씩 문제를 풀고나서 정답지를 확인하느냐의 문제입니다. 사람은 문제를 풀고 정답을 보는 순간 부족했던 점을 깨달으며 지식이 업데이트 된다고 하였습니다. 기계 입장에서는 실제값과 예측값으로부터 오차를 계산하고 옵티마이저가 매개변수를 업데이트합니다. 여기서 중요한 포인트는 업데이트가 시작되는 시점이 정답지/실제값을 확인하는 시점이라는 겁니다.

사람이 2,000 문제가 수록되어있는 문제지의 문제를 200개 단위로 풀고 채점한다고 하면 이때 배치 크기는 200입니다. 기계는 배치 크기가 200이면 200개의 샘플 단위로 가중치를 업데이트 합니다.

여기서 주의할 점은 배치 크기와 배치의 수는 다른 개념이라는 점입니다. 전체 데이터가 2,000일때 배치 크기를 200으로 준다면 배치의 수는 10입니다. 이는 에포크에서 배치 크기를 나눠준 값(2,000/200 = 10)이기도 합니다. 이때 배치의 수를 이터레이션이라고 합니다.

### **3) 이터레이션(Iteration)**

이터레이션이란 한 번의 에포크를 끝내기 위해서 필요한 배치의 수를 말합니다. 또는 한 번의 에포크 내에서 이루어지는 매개변수의 업데이트 횟수이기도 합니다. 전체 데이터가 2,000일 때 배치 크기를 200으로 한다면 이터레이션의 수는 총 10개입니다. 이는 한 번의 에포크 당 매개변수 업데이트가 10번 이루어진다는 것을 의미합니다. SGD를 이 개념을 가지고 다시 설명하면, SGD는 배치 크기가 1이므로 모든 이터레이션마다 하나의 데이터를 선택하여 경사 하강법을 수행합니다.



출처: https://wikidocs.net/36033
