# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용 단순선형회귀모델 
"""

import tensorflow as tf  # tensorflow 도구 
import matplotlib.pyplot as plt # 회귀선 시각화 

# 1. X, y변수 생성 
X = tf.constant([1, 2, 3], dtype=tf.float32) # X는 독립변수 (입력)  
# 1, 2, 3 각각은 관측치임 ★★★
y = tf.constant([2, 4, 6], dtype=tf.float32) # 종속변수(정답) 


# 2. a, b변수 정의 
tf.random.set_seed(123)
w  = tf.Variable(tf.random.normal([1])) # 가중치 : 난수 
b  = tf.Variable(tf.random.normal([1])) # 편향 : 난수 


# 3. 회귀모델 
def linear_model(X) :  
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 


# 4. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss


# 5. model 최적화 객체  
optimizer = tf.optimizers.SGD(learning_rate=0.01) # 딥러닝 최적화 알고리즘
# optimizers 모듈의 SGD() 함수 

# 6. 반복학습 
for step in range(100) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b]) # (최소화시킬 손실값, 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())

    # a, b 변수 update 
    print(f'가중치(w) = {w.numpy()}, 편향(b) = {b.numpy()}')



