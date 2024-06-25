# -*- coding: utf-8 -*-
"""
step03_keras_cnn_cifar10.py

CNN model 생성 
 1. image dataset load 
 2. image dataset 전처리 
 3. CNN model 생성 : layer 구축 + 학습환경 + 학습 
 4. CNN model 평가
 5. CMM model history 
"""

from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten # DNN layer 
import matplotlib.pyplot as plt 

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
input_shape = (32, 32, 3) # input images 

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
# Conv layer1 : Conv + MaxPool
model.add(Conv2D(filters=32, kernel_size=(5, 5),                  
                 input_shape = input_shape, # (32, 32, 3)
                 activation='relu')) # padding default VALID 
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 
'''
filter = 32 : 특징맵 32장
kernel_size = (5,5) : 커널(필터)의 세로/가로 크기 
strides = (1,1) : 커널 세로/가로 이동 크기 
padding = 'VALID' : 합성곱으로 특징맵 크기 결정 
-----------------------------------------------
pool_size = (3, 3) : pooling window 크기 
strides = (2, 2) : (합성곱 이후 strides를 통해 차원 줄이기)
'''

# Conv layer2 : Conv + MaxPool 
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu')) 
model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2))) 

# Conv layer3 : Conv   
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

# 전결합층 : Flatten layer 
model.add(Flatten()) # 3d/2d -> 1d 
# (컬러이면 3d, 흑백이면 2d)

# DNN1 : hidden layer 
model.add(Dense(units=64, activation='relu'))

# DNN2 : output layer  
model.add(Dense(units = 10, activation='softmax')) 
                  

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        2432      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 9, 9, 64)          51264     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 4, 4, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 2, 2, 128)         73856     
         
 ---------------------
 flatten_1 전까지 3D                                                        
 ---------------------
 flatten (Flatten)           (None, 512)               0         
                                                                 
 dense (Dense)               (None, 64)                32832     
                                                                 
 dense_1 (Dense)             (None, 10)                650       
                                                                 
=================================================================
Total params: 161,034
Trainable params: 161,034
Non-trainable params: 0
_________________________________________________________________
'''

# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 (50,000장)
          epochs=10, # 반복학습  # 50,000 * 10 번
          batch_size = 100, # 1회 공급 image size 
                      # 100장 씩 500번(iteration) => 50000 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 

'''
val_accuracy: 0.6865 : 이미지의 노이즈에 의해 70% 언저리로 나옴 
epoch vs accuracy 그래프에서 train - val 간격이 클 수록 과적합 우려 
'''
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



