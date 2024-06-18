# -*- coding: utf-8 -*-
"""
★★★ 정의할 때, 자료형 미지정 오류를 방지하기 위해 
x.0 소수점을 표기하여 실수 자료형으로 지정 할것  ★★★

다중선형회귀방정식 작성 
  예) 독립변수 2개 
   y_pred = (X1 * w1 + X2 * w2) + base
"""

import tensorflow as tf 

# X, y변수 정의 
X = tf.constant([[1.0, 2.0]]) # 독립변수
y = tf.constant(2.5)  # 종속변수

# w, b변수 정의 : 초기값 난수   
tf.random.set_seed(1) # 난수 seed값 
w = tf.Variable(tf.random.normal([2, 1])) # 2행1열의 2개 난수 
b = tf.Variable(tf.random.normal([1])) # 1개 난수 


# 선형회귀모델 
def linear_model(X) : 
    global w, b
    y_pred = tf.linalg.matmul(X, w) + b # 다중회귀방정식 (행렬내적) 
    return y_pred 
'''
행렬곱 
tf.linalg.matmul(X, w)
1. X, w 모두 행렬 
2. 수일치 : X(1,2) vs w(2,1)
3. 
'''

# 모델 오차(model error) 
def model_err(X, y) : 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 오차  
    return err 


# 손실 함수(loss function)
def loss_fn(X, y) :
    err = model_err(X, y) # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수   
    return loss


# 프로그램 시작점 
print('-'*30)
print('<<가중치, 편향 초기값>>')
print('가중치(w) : \n', w.numpy(), '\n편향(b) :', b.numpy())
print('-'*30)
'''
------------------------------
<<가중치, 편향 초기값>>
가중치(w) : 
 [[-1.1012203]
 [ 1.5457517]] 
편향(b) : [0.40308788]
------------------------------
'''



# 모델 오차 
err = model_err(X, y)
print('err =', err.numpy())
# err = [[0.10662889]]

# 손실/비용 함수
loss = loss_fn(X, y)
print('손실(loss) =', loss.numpy()) # seed값을 바꾸면 달라짐 