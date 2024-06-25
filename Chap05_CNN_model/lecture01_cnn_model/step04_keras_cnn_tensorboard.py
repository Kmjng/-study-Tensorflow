# -*- coding: utf-8 -*-
"""
step04_keras_cnn_tensorboard.py

 - Keras CNN layers tensorboard 시각화 
"""
import tensorflow as tf 
from tensorflow.keras.datasets.cifar10 import load_data # color image dataset 
from tensorflow.keras.utils import to_categorical # one-hot encoding 
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Conv2D, MaxPool2D # Conv layer 
from tensorflow.keras.layers import Dense, Flatten, Dropout # DNN layer 
import matplotlib.pyplot as plt 

# [추가] tensorboard 초기화 
tf.keras.backend.clear_session()


# 1. image dataset load 
(x_train, y_train), (x_val, y_val) = load_data()

x_train.shape # image : (50000, 32, 32, 3) - (size, h, w, c)
y_train.shape # label : (50000, 1)

first_img = x_train[0]
first_img.shape # (32, 32, 3)

plt.imshow(first_img)
plt.show()

print(y_train[0]) # [6]

x_val.shape # image : (10000, 32, 32, 3)
y_val.shape # label : (10000, 1)


# 2. image dataset 전처리

# 1) image pixel 실수형 변환 
x_train = x_train.astype(dtype ='float32') # type일치  
x_val = x_val.astype(dtype ='float32')

# 2) image 정규화 : 0~1
x_train = x_train / 255
x_val = x_val / 255

x_train[0]

# 3) label 전처리 : 10진수 -> one hot encoding(2진수) 
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)

y_train.shape # (50000, 10)
y_train[0] # [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]


# 3. CNN model & layer 구축 
input_shape = (32, 32, 3)

# 1) model 생성 
model = Sequential()

# 2) layer 구축 
# Conv layer1 : filter[5, 5, 3, 32]
model.add(Conv2D(filters=32, kernel_size=(5,5), 
                 input_shape = input_shape, activation='relu')) # image[28x28]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[13x13]
model.add(Dropout(0.3))

# Conv layer2 : filter[5, 5, 32, 64]
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu')) # image[9x9]
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2))) # image[4x4]
model.add(Dropout(0.1))

# Conv layer3 : filter[3, 3, 64, 128] - MaxPool2D 제외 
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))# image[2x2]
model.add(Dropout(0.1))


# 전결합층 : Flatten layer : 3d[h,w,c] -> 1d[n=h*w*c]
model.add(Flatten())


# DNN : hidden layer : 4층[n x 64] 
model.add(Dense(units=64, activation='relu'))


# DNN : output layer : 5층 
model.add(Dense(units = 10, activation='softmax'))
        
model.summary()  
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 28, 28, 32)        2432      
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
dropout_8 (Dropout)          (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 9, 9, 64)          51264     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 4, 4, 64)          0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 4, 4, 64)          0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 2, 128)         73856     
_________________________________________________________________
dropout_10 (Dropout)         (None, 2, 2, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 64)                32832     
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
'''
          

# 4. model compile : 학습과정 설정(다항분류기) 
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# [추가] Tensorboard 
from tensorflow.keras.callbacks import TensorBoard 
from datetime import datetime # 날짜/시간 자료 생성 

logdir = 'c:/graph/' + datetime.now().strftime('%Y%m%d-%H%M%S') # '년월일-시분초'
callback = TensorBoard(log_dir=logdir)


# 5. model training : train(105) vs val(45) 
model_fit = model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=2, # 반복학습 
          batch_size = 100, # 1회 공급 image size
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val), # 검증셋
          callbacks = [callback]) # [추가] 


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


'''
Tensorboard 실행순서
1. logdir : log 파일 확인 
2. (base) conda activate tensorflow 
3. (tensorflow) tensorboard --logdir=C:\graph\년월일-시분초\train
-> 주의 : '=' 앞과 뒤 공백 없음 
4. http://localhost/6006
'''








