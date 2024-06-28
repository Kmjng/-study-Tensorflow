# -*- coding: utf-8 -*-
"""
step02_timeSeries_RNN.py

 - 시계열분석 : 시계열데이터 + RNN model (PPT.14~15 참고)
"""
import pandas as pd # csv file read 
import matplotlib.pyplot as plt # 시계열 시각화 
import numpy as np # ndarray
import tensorflow as tf # seed 값 
from tensorflow.keras import Sequential # model 
from tensorflow.keras.layers import SimpleRNN, Dense # RNN layer 
import random 

tf.random.set_seed(35) # seed값 지정
np.random.seed(35)
random.seed(35)
 

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. csv file read  
path = r'C:/ITWILL/7_Tensorflow/Tensorflow/data'
timeSeries = pd.read_csv(path + '/timeSeries.csv')

data = timeSeries['data']
print(data) 


# 2. RNN 적합한 dataset 생성 : : (batch_size, time_steps, features)
x_data = [] # 학습데이터 : 1~10개 시점 데이터  
for i in range(len(data)-10) : # 90
    for j in range(10) : # 10
        x_data.append(data[i+j]) # 90 * 10 = 900

# list -> array 
x_data = np.array(x_data)


y_data = [] # 정답 : 11번째 시점 데이터  
for i in range(len(data)-10) : # 90
    y_data.append(data[i+10]) # 90


# list -> array
y_data = np.array(y_data)


# train(70)/val(20) split 
x_train = x_data[:700].reshape(70, 10, 1) 
x_val = x_data[700:].reshape(20, 10, 1) 


# train(70)/val(20) split 
y_train = y_data[:70].reshape(70,1) 
y_val = y_data[70:].reshape(20,1) 



# 3. model 생성 
model = Sequential()

input_shape = (10, 1)

# RNN layer 추가 
model.add(SimpleRNN(units=16, input_shape=input_shape, 
                    activation ='tanh'))

# DNN layer 추가 
model.add(Dense(units=1)) # # 회귀모델 : 활성함수 없음 

# model 학습환경 
model.compile(optimizer='sgd', 
              loss='mse', metrics=['mae'])

# model 학습 
model.fit(x=x_train, y=y_train, 
          epochs=400, verbose=1)


# model 예측 
y_pred = model.predict(x_val) 


# 학습데이터(70개) + 예측데이터(20개) 시각화 
y_pred = np.concatenate([y_train, y_pred])  


# 20개 주가 예측 
future = 20

threshold = np.ones_like(y_pred, dtype='bool') 
threshold[:-future] = False  

pred_x = np.arange(len(y_pred)).reshape(-1, 1) # x축 색인자료 
pred_y = y_pred # y축 시계열 예측치  

# 한글 & 음수부호 지원 
plt.rcParams['font.family'] = 'Malgun Gothic'
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# y_true vs y_pred 
plt.plot(y_data, color='lightblue', linestyle='--', marker='o', label='real value')
plt.plot(pred_x[threshold], pred_y[threshold], 'r--', marker='o', label='predicted value')
plt.legend(loc='best')
plt.title(f'{20}개의 시계열 예측결과')
plt.show()
