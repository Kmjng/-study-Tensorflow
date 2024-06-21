# -*- coding: utf-8 -*-
"""
step01_linearRegression_dnn

Keras : High Level API  
"""

# dataset 
from sklearn.datasets import load_iris # dataset
from sklearn.model_selection import train_test_split # split 
from sklearn.metrics import mean_squared_error, r2_score # 평가 

# keras model 
import tensorflow as tf
from tensorflow.keras import Sequential # keara model 
from tensorflow.keras.layers import Dense # DNN layer 
# 실제 레이어 계층을 추가해주는 역할 

import numpy as np 
import random 

# 조절 변수를 포함한 난수들을 생성할 때
## karas 내부 weight seed 적용 
tf.random.set_seed(123) # global seed 
np.random.seed(123) # numpy seed
random.seed(123) # random seed 


# 1. dataset laod 
X, y = load_iris(return_X_y=True)


# 2. 공급 data 생성 : 훈련셋, 검증셋 
x_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=123)


# 3. DNN model 생성 
model = Sequential() 
dir(model)

'''
add() : 레이어 추가
compile(): 학습과정 설정
fit() : 모델학습
summary() : 레이어 요약
predict() : y 예측
'''

# 4. DNN model layer 구축 

# hidden layer1  : 1층 (w = [4,12]개, b = 12개)
model.add(Dense(units=12, input_shape=(4,), activation='relu'))# 1층 
# units = 노드 갯수 (hidden layer 하나에 들어가는 노드들)
# hidden layer2 : 2층 (w = [12, 6]개, b = 6개)
model.add(Dense(units=6, activation='relu'))# 2층

# output layer  # 하나의 노드 
model.add(Dense(units=1))# 3층 
# 선형회귀 DNN이므로, 활성화함수 사용하지 않음 ★

'''
add 메서드는 모델에 새로운 층을 추가
이때 추가하는 층은 Dense, Activation, Dropout, Conv2D 등 
다양한 층일 수 있음
<역할>
Dense: 완전 연결 층을 정의합니다.
add: 모델에 층을 추가합니다.
'''

# 레이어 확인 
model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 12)                60   (w, b 갯수 합; 4*12 + 12)     
                                                                 
 dense_1 (Dense)             (None, 6)                 78        
                                                                 
 dense_2 (Dense)             (None, 1)                 7         
                                                                 
=================================================================
Total params: 145
Trainable params: 145
Non-trainable params: 0
_________________________________________________________________
'''
# 5. model compile : 학습과정 설정 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
'''
optimizer : 딥러닝 최적화 알고리즘(SGD, Adam)
loss : 손실함수(mse, cross_entropy)
metrics : 평가방법
'''

# 6. model training 
model.fit(x=x_train, y=y_train,  
          epochs=100,  
          verbose=1,  
          validation_data=(x_val, y_val))   

''' verbose=1..

validation_data=(x_val, y_val))
Epoch 1/100
4/4 [==============================] - 1s 104ms/step - loss: 1.4317 - mae: 1.0020 - val_loss: 1.3476 - val_mae: 0.9088
Epoch 2/100
4/4 [==============================] - 0s 10ms/step - loss: 1.1849 - mae: 0.8870 - val_loss: 1.1194 - val_mae: 0.7974
Epoch 3/100


Epoch 98/100
4/4 [==============================] - 0s 16ms/step - loss: 0.0590 - mae: 0.1844 - val_loss: 0.0575 - val_mae: 0.1646
Epoch 99/100
4/4 [==============================] - 0s 16ms/step - loss: 0.0588 - mae: 0.1838 - val_loss: 0.0590 - val_mae: 0.1665
Epoch 100/100
4/4 [==============================] - 0s 16ms/step - loss: 0.0586 - mae: 0.1818 - val_loss: 0.0596 - val_mae: 0.1663
'''

# 7. model testing 
y_pred = model.predict(x_val)
y_true = y_val 
mse = mean_squared_error(y_true, y_pred)
y_pred.dtype
r2_score = r2_score(y_true, y_pred)
print('rmse:', mse**0.5, '\nR2score:',r2_score)

# rmse: 0.24412126038262874 
# R2score: 0.9233289331093542