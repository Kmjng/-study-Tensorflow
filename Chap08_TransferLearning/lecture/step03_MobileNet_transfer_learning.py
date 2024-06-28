# -*- coding: utf-8 -*-
"""
전이학습(transfer_learning)  : ppt.14~15 참고 
 - cifar10 이미지 분류(chapter06) vs MobileNet 전이학습 
"""

from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
#from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN layer 
import matplotlib.pyplot as plt 

from tensorflow.keras.applications.mobilenet import MobileNet 

import numpy as np # ndarray
import tensorflow as tf # seed 값
import random 

tf.random.set_seed(35) # seed값 지정
np.random.seed(35)
random.seed(35)


# 1. image dataset load 
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # image : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # label : (50000, 1)


x_val.shape # image : (10000, 32, 32, 3)
y_val.shape # label : (10000, 1)


# 2. image dataset 전처리

# 1) image pixel 실수형 변환 
x_train = x_train.astype(dtype ='float32')  
x_val = x_val.astype(dtype ='float32')

# 2) image 정규화 : 0~1
x_train = x_train / 255
x_val = x_val / 255


# 3) label 전처리 : 10진수 -> one hot encoding(2진수) 
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)


# 3. CNN model & layer 구축 
#input_shape = (32, 32, 3) # input images 

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
'''
# Conv layer1 
model.add(Conv2D(filters=32, kernel_size=(5,5), 
                 input_shape = input_shape, activation='relu')) # image[28x28]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[13x13]

# Conv layer2 
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # image[9x9]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[4x4]

# Conv layer3  
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))# image[2x2]
'''


###########################################
####  전이학습 
###########################################

### MobileNet 기본 model 생성 

# 1) input image
input_shape = (32, 32, 3) 

# 2) 기본 모델을 이용하여 객체 생성 : 학습된 가중치, 입력 이미지 정보, 출력층(x)    

base_model = MobileNet(weights='imagenet', # 기존 학습된 가중치 이용 
                   input_shape=input_shape,  # input image 
                   include_top = False) # 최상위 레이어(output) 사용안함

model.add(base_model) # 전이학습 모델 레이어 추가 


# 전결합층 : Flatten layer 
model.add(Flatten())

model.add(Dropout(0.7))

# DNN1 : hidden layer 
model.add(Dense(units=64, activation='relu'))


# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax'))
                  

# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(50000) vs val(10000) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=10, # 반복학습 
          batch_size = 100, # 1회 공급 image size 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. CNN model 평가 : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

 
# 7. CMM model history 
print(model_fit.history.keys()) # key 확인 
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# loss vs val_loss 
plt.plot(model_fit.history['loss'], 'y', label='train loss')
plt.plot(model_fit.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.legend(loc='best')
plt.show()


# accuracy vs val_accuracy 
plt.plot(model_fit.history['accuracy'], 'y', label='train acc')
plt.plot(model_fit.history['val_accuracy'], 'r', label='val acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.show()



