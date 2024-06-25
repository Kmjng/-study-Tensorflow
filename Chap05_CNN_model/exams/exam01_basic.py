'''
문1) 다음과 같은 조건으로 Convolution layer와 Max Pooling layer를 정의하고, 특징맵의 shape을 확인하시오.
  <조건1> input image : volcano.jpg 파일 대상    
  <조건2> Convolution layer 정의 
       -> Kernel size : 6x6
       -> featuremap : 16개
       -> strides= 1x1, padding='SAME'  
  <조건3> Max Pooling layer 정의 
       -> ksize= 3x3, strides= 2x2, padding='SAME' 
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('C:/ITWILL/7_Tensorflow/data/images/volcano.jpg') # 이미지 읽어오기
plt.imshow(img)
plt.show()
print(img.shape)

