# -*- coding: utf-8 -*-
"""
기후 데이터 시계열 분석 : ppt.17 참고 

시계열 모델의 독립변수와 종속변수 
 독립변수 : 이전 시점 20개 온도 -> 종속변수 : 21번째 온도(1개)  
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True' 

# Matplotlib Setting
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
tf.random.set_seed(13) # Random Seed Setting


'''
에나 기후(jena climate) dataset
독일 에나 연구소에서 제공하는 기후(climate) 데이터셋으로 온도, 기압, 습도 등 14개의 
날씨 관련 변수를 제공한다. 8년(2009~2016) 동안 매일 10 단위로 기록한 데이터셋 
'''

# 1. csv file read : 에나 기후(jena climate) dataset 
path = r'C:\ITWILL\7_Tensorflow\data'
df = pd.read_csv(path+'/jena_climate_2009_2016.csv')
df.info() # 420551
'''
RangeIndex: 420551 entries, 0 to 420550
Data columns (total 15 columns):
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   Date Time        420551 non-null  object  : 날짜/시간 
 1   p (mbar)         420551 non-null  float64 : 대기압(밀리바 단위)
 2   T (degC)         420551 non-null  float64 : 온도(섭씨)
 3   Tpot (K)         420551 non-null  float64 : 온도(절대온도)
 4   Tdew (degC)      420551 non-null  float64 : 습도에 대한 온도
 5   rh (%)           420551 non-null  float64 : 상대 습도
 6   VPmax (mbar)     420551 non-null  float64 : 포화증기압
 7   VPact (mbar)     420551 non-null  float64 : 중기압 
 8   VPdef (mbar)     420551 non-null  float64 : 중기압부족 
 9   sh (g/kg)        420551 non-null  float64 : 습도 
 10  H2OC (mmol/mol)  420551 non-null  float64 : 수증기 농도 
 11  rho (g/m**3)     420551 non-null  float64 : 공기밀도 
 12  wv (m/s)         420551 non-null  float64 : 풍속 
 13  max. wv (m/s)    420551 non-null  float64 : 최대풍속
 14  wd (deg)         420551 non-null  float64 : 풍향 
''' 


#######################################
# LSTM을 이용한 기상예측: 단변량
#######################################

### 1. 변수 선택 및 탐색  
uni_data = df['T (degC)'] # 온도 칼럼 
uni_data.index = df['Date Time'] # 날짜 칼럼으로 index 지정 


# Visualization the univariate : 표준화 필요성 확인 
uni_data.plot(subplots=True)
plt.show() # -20 ~ 40

# 시계열 자료 추출 
uni_data = uni_data.values # 값 추출 

# 표준화(Z-Normalization)   
uni_train_mean = uni_data.mean()
uni_train_std = uni_data.std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

# 표준화 여부 확인 
plt.plot(uni_data)
plt.show() # -3 ~ 3



### 2. 단변량 데이터 생성 :  LSTM모델 공급에 적합한 자료 만들기 

# 1) 단변량 데이터 생성 함수 
def univariate_data(dataset, s_index, e_index, past_size) : 
    X = [] # x변수 
    y = [] # y변수 

    s_index = s_index + past_size
    if e_index is None: # val dataset 
        e_index = len(dataset) 
    
    for i in range(s_index, e_index): 
        indices = range(i-past_size, i) 
        X.append(np.reshape(dataset[indices], (past_size, 1))) # x변수(20, 1)  
        
        y.append(dataset[i]) # y변수(1,)  
        
    return np.array(X), np.array(y)


# 2) 단변량 데이터 생성 
TRAIN_SPLIT = 300000 # train vs val split 기준
past_data = 20 # x변수 : 과거 20개 자료[0~19, 1~20,..] 

# 훈련셋 
X_train, y_train = univariate_data(uni_data,0,TRAIN_SPLIT,past_data)
# 검증셋 
X_val, y_val = univariate_data(uni_data,TRAIN_SPLIT, None, past_data)

# Check the Data
print(X_train.shape) # (299980, 20, 1) 
print(y_train.shape) # (299980,) 



### 3. LSTM Model 학습 & 평가   
input_shape=(20, 1)

model = Sequential()
model.add(LSTM(16, input_shape = input_shape)) 
model.add(Dense(1)) # 회귀함수
model.summary()


# 학습환경 
model.compile(optimizer='adam', loss='mse')


# 모델 학습 
model_history = model.fit(X_train, y_train, epochs=10, # trainset
          batch_size = 256,
          validation_data=(X_val, y_val))#, # valset 

          
# model evaluation 
print('='*30)
print('model evaluation')
model.evaluate(x=X_val, y=y_val)



### 4. Model 손실(Loss) 시각화 
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Single Step Training and validation loss')
plt.legend(loc='best')
plt.show()


### 5. model prediction 

# 테스트셋 선택 : 검증셋 중에서 5개 관측치 선택 
X_test = X_val[:5] # (5, 20, 1)
y_test = y_val[:5] # (5,)


# 예측치 : 5개 관측치 -> 5개 예측치 
y_pred = model.predict(X_test)  





