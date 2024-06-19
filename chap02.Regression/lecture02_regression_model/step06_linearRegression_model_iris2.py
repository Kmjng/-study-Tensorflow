# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용 다중선형회귀모델 + csv file 
 - y변수 : 1칼럼, X변수 : 2~4칼럼 
 - 딥러닝 최적화 알고리즘 : Adam 적용 
 - 반복학습(step) 적용 
"""

import tensorflow as tf  # 딥러닝 최적화 알고리즘
from sklearn.datasets import load_iris # dataset 
from sklearn.model_selection import train_test_split # dataset split 
from sklearn.metrics import mean_squared_error # 평가 
from sklearn.preprocessing import minmax_scale # 정규화(0~1)
import numpy as np 
# 1. dataset load 
X, y = load_iris(return_X_y=True)


# X변수 정규화 
X_nor = minmax_scale(X)
type(X_nor) # numpy.ndarray

# Convert X_nor to float32 explicitly
X_nor = X_nor.astype(np.float32)

# y변수 : 1칼럼, X변수 : 2~4칼럼 
y_data = X_nor[:,0] # 1칼럼  - y변수 
x_data = X_nor[:,1:] # 2~4칼럼 - x변수 


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=123)



# 3. w, b 변수 정의 : update 대상 
tf.random.set_seed(123) # w,b 난수 seed값 지정 
w = tf.Variable(tf.random.normal(shape=[3, 1])) 
b = tf.Variable(tf.random.normal(shape=[1])) 


# 4. 회귀모델 정의 : 행렬곱 이용 
def linear_model(X) : # X:입력 -> y 예측치 : 출력 
    y_pred = tf.linalg.matmul(X, w) + b 
    return y_pred 


# 5. 손실/비용 함수 정의
def loss_fn() : # 인수 없음 
    y_pred = linear_model(x_train) # y 예측치
    err = tf.math.subtract(y_train, y_pred) # y - y_pred 
    loss = tf.reduce_mean(tf.square(err)) # MSE 
    return loss 


# 6. 최적화 객체 생성 
opt = tf.optimizers.Adam(learning_rate=0.01) # 학습률 

print('초기값 : w =', w.numpy(), ", b =", b.numpy())

loss_value =[] 
# 7. 반복학습
for step in range(100) : 
    opt.minimize(loss=loss_fn, var_list=[w, b]) 
    loss_value.append(loss_fn().numpy())
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ', loss value =', loss_fn().numpy())
    
'''
step = 10 , loss value = 0.10367453
step = 20 , loss value = 0.098841116
step = 30 , loss value = 0.09543637
step = 40 , loss value = 0.09259493
step = 50 , loss value = 0.08998571
step = 60 , loss value = 0.08750658
step = 70 , loss value = 0.085137114
step = 80 , loss value = 0.082879044
step = 90 , loss value = 0.08073658
step = 100 , loss value = 0.07871212
'''

# 모델링 및 평가 
y_pred = linear_model(x_test)
y_pred

rmse = mean_squared_error(y_test, y_pred)**0.5

print('rmse:', rmse)
# rmse: 0.5089280712870958



# loss vs step 
loss_value

import matplotlib.pyplot as plt 
plt.plot(loss_value)
plt.ylabel('Loss value')
plt.xlabel('epochs')
plt.show()




