# -*- coding: utf-8 -*-
"""
문1) boston 데이터셋을 이용하여 다음과 같이 Keras DNN model layer을 
    구축하고, model을 학습하고, 검증(evaluation)하시오. 
    <조건1> 4. DNN model layer 구축 
         1층(hidden layer1) : units = 64
         2층(hidden layer2) : units = 32
         3층(hidden layer3) : units = 16 
         4층(output layer) : units=1
    <조건2> 6. model training  : 훈련용 데이터셋 이용 
            epochs = 50
    <조건3> 7. model evaluation : 검증용 데이터셋 이용     
"""
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 

# keras model 관련 API
from tensorflow.keras.datasets import boston_housing # dataset
from tensorflow.keras import Sequential # model 생성 
from tensorflow.keras.layers import Dense # DNN layer


# 1. x,y data 생성 : keras datasests 이용 
(x_train, y_train), (x_val, y_val) = boston_housing.load_data()
x_train.shape # (404, 13)
x_val.shape # (102, 13)

# 2. X, y변수 정규화 
x_train = minmax_scale(x_train)
x_val = minmax_scale(x_val)

y_train = y_train / y_train.max()
y_val = y_val / y_val.max()
y_val.shape # (102,)

# 3. keras model
model = Sequential() 
print(model) # object info


# 4. DNN model layer 구축 # 독립변수 12개 
model.add(Dense(units=64, input_shape=(13,), activation='relu')) # 1층 
model.add(Dense(units=32, activation='relu')) # 2층 
model.add(Dense(units=16, activation='relu')) # 3층 
model.add(Dense(units=1)) # 4층 


# 5. model compile : 학습과정 설정(다항 분류기)
model.compile(optimizer = 'adam', 
         loss = 'mse', 
         metrics = ['mae'])


# model layer 확인 
model.summary()
'''
________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_25 (Dense)            (None, 64)                896       
                                                                 
 dense_26 (Dense)            (None, 32)                2080      
                                                                 
 dense_27 (Dense)            (None, 16)                528       
                                                                 
 dense_28 (Dense)            (None, 1)                 17        
                                                                 
=================================================================
Total params: 3,521
Trainable params: 3,521
Non-trainable params: 0
_________________________________________________________________
'''

# 6. model training 
model.fit(x_train, y_train, epochs = 50, 
          verbose =1, validation_data=(x_val, y_val)) 

# 7. model evaluation : test dataset

# 방법 1. 자체 평가 도구 (model.evaluate)
model.evaluate(x_val, y_val) 
'''
loss: 0.0147 - mae: 0.0893
'''

# 방법 2.
y_pred = model.predict(x_val)
y_true = y_val 

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print('rmse:', mse**0.5)
# rmse: 0.12141977232796272