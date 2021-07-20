## Nvidia Driver 설치 (Ubuntu 18.04)

### cuda 버전 확인
```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
python
```
### nvidia-smi 테스트
> 지원 가능한 최대 CUDA 버전을 출력
```
$ docker run --runtime=nvidia --rm nvidia/cuda:10.0-base nvidia-smi
Tue Jul 20 12:39:34 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   42C    P8    29W / 149W |      0MiB / 11441MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
> CUDA 10 에 맞는 KERAS 및 tensorflow_gpu 버전
```
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ARG KERAS=2.2.4
ARG TENSORFLOW=1.13.1
```
