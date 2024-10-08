# -*- coding: utf-8 -*-
"""
keras 모델에서 학습률 적용   
 optimizer=Adam(learning_rate = 0.01)
"""

from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # split 
from sklearn.preprocessing import minmax_scale # x변수 : 스케일링(0~1)
from sklearn.metrics import accuracy_score  # model 평가 

from tensorflow.keras.utils import to_categorical # Y변수 : encoding
from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Dense # DNN layer 구축 
from tensorflow.keras.models import load_model # model load 

import tensorflow as tf
import numpy as np 
import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(123)
np.random.seed(123)
rd.seed(123)


# 1. dataset load & 전처리 
X, y = load_iris(return_X_y=True)

X.shape # (150, 4)
y.shape # (150,)


# X변수 : 정규화(0~1)
X = minmax_scale(X) # 

# y변수 : one hot encoding
y_one = to_categorical(y)  
print(y_one)
y_one.shape #  (150, 3)
'''
[1, 0, 0] <- 0
[0, 1, 0] <- 1
[0, 0, 1] <- 2
'''

# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    X, y_one, test_size=0.3, random_state=123)


# 3. keras layer & model 생성 
model = Sequential()


# hidden layer1 
model.add(Dense(units=12, input_shape =(4, ), activation = 'relu')) # 1층 

# hidden layer2 
model.add(Dense(units=6, activation = 'relu')) # 2층 

# output layer 
model.add(Dense(units=3, activation = 'softmax')) # 3층 


# 4. model compile : 학습과정 설정(다항분류기) 
from tensorflow.keras import optimizers # 딥러닝 최적화 알고리즘 
dir(optimizers)
'''
Adam
RMSprop
SGD
'''
model.compile(optimizer=optimizers.Adam(learning_rate=0.01), 
              loss = 'categorical_crossentropy',  
              metrics=['accuracy'])


# 5. model training : train(105) vs val(45) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=200, # 반복학습 
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 6. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

'''
Adam(learning_rate=0.001)
loss: 0.1898 - accuracy: 0.9111

Adam(learning_rate=0.01)
loss: 0.0525 - accuracy: 0.9778
'''


# 7. model save & load : HDF5 파일 형식 
model.save('keras_model_iris.h5')

my_model = load_model('keras_model_iris.h5')
 