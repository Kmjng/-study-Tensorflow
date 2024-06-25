# -*- coding: utf-8 -*-
"""
합성곱과 폴링 : ppt. 25~26 참고

합성곱층(Convolution layer) = 합성곱 + 폴링   
 합성곱(Convolution) : 이미지 특징 추출
 폴링(Pooling ) : 이미지 픽셀 축소(다운 샘플링)
"""
import tensorflow as tf # 합성곱, 폴링 연산 
from tensorflow.keras.datasets.mnist import load_data # 데이터셋 
import numpy as np # 이미지 축 변경 
import matplotlib.pyplot as plt # 이미지 시각화 
# 시각화 오류 해결방법 : Fatal Python error: Aborted 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. Input image 만들기  
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,) : 10진수 


# 1) 자료형 변환 : int -> float
x_train = x_train.astype('float32') # type 일치 
x_test = x_test.astype('float32')

# 2) 정규화 
x_train.min # 0.0 -> 흰색 
x_train.max # 255.0 -> 검정색
x_train /= 255 # x_train = x_train / 255
x_test /= 255
# 0 ~ 1

# 3) input image 선정 : 첫번째 image 
img = x_train[0]
plt.imshow(img, cmap='gray') # 숫자 5  
# plt.imshow() ; 이미지를 표시하는 데 사용
# img ; 이미지 데이터, 보통 Numpy 배열 형태 
plt.show() 
img.shape # (28, 28)

# 4) input image 모양변경  
inputImg = img.reshape(1,28,28,1) # [size, h, w, ch]


# 2. Filter 만들기 : image에서 특징 추출  
Filter = tf.Variable(tf.random.normal([3,3,1,5])) # [h, w, ch, fmap] 
'''
★ filter의 크기는 보통 짝수가 아닌 홀수가 온다. 
h=3, w=3 : 커널(kernel) 세로,가로 크기
c=1 : 이미지 채널(channel)수    
fmap=5 : 추출할 특징 맵 개수   

28*28 이 5개의 feature map으로 
'''


# 3. Convolution layer : 특징맵  추출  # .nn.conv2d ★★★   
conv2d = tf.nn.conv2d(inputImg, Filter, # Filter에 필터(커널)사이즈 들어감
                      strides=[1,1,1,1], 
                      # 2,3번째 원소가 중요 
                      # 2번째 원소:가로 이동 크기, 3: 세로이동 크기 
                      # 1,4번째 : 차원 맞추는 용도
                      padding='VALID') # 이미지와 필터 합성곱
'''
strides=[1,1,1,1] : kernel 가로/세로 1칸씩 이동 

padding = 'SAME' : 원본이미지와 동일한 크기로 이미지 특징 추출 
- output size = (Input_size)/(Stride)
- zero padding이 들어가서 사이즈 유지됨

padding='VALID' : 필터의 크기에 의해 사이즈 변동
- output = (Input_size - Kernel_size) /Stride + 1
'''

conv2d.shape
 # TensorShape([1, 28, 28, 5]) # padding = 'SAME'
# TensorShape([1, 26, 26, 5]) # padding='VALID'

# 합성곱(Convolution) 연산 결과  
# 1 <--> 5 ; 5장의 이미지를 한 개의 열로 나타내기 위해 
conv2d_img = np.swapaxes(conv2d, 0, 3) # 축 변환(첫번째와 네번째) 

# output = (Input_size - Kernel_size) /Stride + 1
output =  (28 - 3) / 1 + 1
output # 26

for i, img in enumerate(conv2d_img) : 
    plt.subplot(1, 5, i+1) # 1행 5열, 열index 
    plt.imshow(img, cmap='gray')  
plt.show()


###########################################
# 4. Pool layer : 특징맵 픽셀 축소  # .nn.max_pool ★★★
pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],
                      padding='SAME') 
'''
ksize=[1,2,2,1] : feature map 하나 안에 윈도우 크기 
 strides=[1,2,2,1] : pooling 크기 가로/세로 2칸씩 이동(이미지 1/2 축소) 
padding = 'SAME' : 원본이미지와 동일한 크기로 특징 이미지 추출 
'''  
pool.shape # TensorShape([1, 13, 13, 5])

# 폴링(Pool) 연산 결과 
pool_img = np.swapaxes(pool, 0, 3) # 축 변환(첫번째와 네번째)
pool_img.shape # (5, 13, 13, 1)

for i, img in enumerate(pool_img) :
    plt.subplot(1,5, i+1)
    plt.imshow(img, cmap='gray') 
plt.show()

    











