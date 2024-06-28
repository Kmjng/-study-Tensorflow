# -*- coding: utf-8 -*-
"""
step01_RNN_basic.py

RNN model 
 - 순환신경망 Many to One RNN 모델(PPT.8 참고)  
"""

import tensorflow as tf # seed value 
import numpy as np # ndarray
from tensorflow.keras import Sequential # model
from tensorflow.keras.layers import SimpleRNN, Dense # RNN layer 

import random as rd 

################################
### keras 내부 seed 적용 
################################
tf.random.set_seed(34)
np.random.seed(34)
rd.seed(34)


# many-to-one : word(4개) -> 출력(1개)
X = [[[0.0], [0.1], [0.2], [0.3]], 
     [[0.1], [0.2], [0.3], [0.4]],
     [[0.2], [0.3], [0.4], [0.5]],
     [[0.3], [0.4], [0.5], [0.6]],
     [[0.4], [0.5], [0.6], [0.7]],
     [[0.5], [0.6], [0.7], [0.8]]] 

y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

X.shape # (6, 4, 1) : RNN 3차원 입력(batch_size, time_steps, features)

model = Sequential() 

input_shape = (4, 1) # (timestep, feature)

# RNN layer 추가 
model.add(SimpleRNN(units=35, input_shape=input_shape, 
                    return_state=False, # Many to One
                    activation='tanh'))

# DNN layer 추가 
model.add(Dense(units=1)) # 출력 : 회귀모델 

# model 학습환경 
model.compile(optimizer='adam', 
              loss='mse', metrics=['mae'])

# model training 
model.fit(X, y, epochs=50, verbose=1)

# model prediction
y_pred = model.predict(X)
print(y_pred)

