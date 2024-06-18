# -*- coding: utf-8 -*-
"""
단순선형회귀방정식(formula) 작성 : ppt.5 참고 
"""

import tensorflow as tf  # ver 2.x

# X, y 변수 
X = tf.constant(6.5) # 독립변수 1개
y = tf.constant(5.2) # 종속변수 1개

# w, b 변수 
# 조절변수이기 때문에 Variable 클래스로 지정해줘야 한다. 
w = tf.Variable(0.5) # 가중치(weight) 
b = tf.Variable(1.5) # 편향(base) 


# 회귀모델 
def linear_model(X) : 
    global w, b # 전역변수 가져와서 함수 내에서 사용할 수 있도록 
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식  
    return y_pred 

# model 오차 
def model_err(X, y) : 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred)  # err = y - y_pred 
    return err 


# 손실 함수 (MSE: 오차제곱평균)
def loss_fn(X, y) :
    err = model_err(X, y) 
    loss = tf.reduce_mean(tf.square(err)) # 손실함수 
    return loss

# >> 위에 지정한 함수를 바탕으로 
# 프로그램 시작점 
print('-'*40)
print('<<가중치, 편향 초기값>>')    
print('가중치(w) =', w.numpy(), '편향(b) =', b.numpy()) # X = 6.5
print('-'*40)

print('y_pred = ', linear_model(X).numpy()) # 0.45
print('y = ', y.numpy()) # y =  5.2
print('model error = %.5f'%(model_err(X, y)))    
print('loss value = %.5f'%(loss_fn(X, y)))

'''
----------------------------------------
<<가중치, 편향 초기값>>
가중치(w) = 0.5 편향(b) = 1.5
----------------------------------------
y_pred =  4.75
y =  5.2
model error = 0.45000
loss value = 0.20250
'''

# 변형 
# 가중치, 편향 수정 
w.assign(0.6) 
b.assign(1.2)

# 프로그램 시작점 
print('-'*40)
print('<<가중치, 편향 수정>>')    
print('가중치(w) =', w.numpy(), '편향(b) =', b.numpy()) # X = 6.5
print('-'*40)

print('y_pred = ', linear_model(X).numpy()) # 0.45
print('y = ', y.numpy()) # y =  5.2
print('model error = %.5f'%(model_err(X, y)))    
print('loss value = %.5f'%(loss_fn(X, y)))
'''
----------------------------------------
<<가중치, 편향 수정>>
가중치(w) = 0.6 편향(b) = 1.2
----------------------------------------
y_pred =  5.1000004
y =  5.2
model error = 0.10000
loss value = 0.01000
'''

