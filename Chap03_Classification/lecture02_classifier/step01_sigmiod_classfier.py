# -*- coding: utf-8 -*-
"""
이항분류기 : 테스트 데이터 적용
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score #  model 평가 

# 1. x, y 공급 data 
# x변수 : [hours, video]
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]] # [6, 2]
# 독립변수가 2개 , 관측치 6개 

# y변수 : [fail or pass] one-hot encoding 
y_data = [[1,0], [1,0], [1,0], [0,1], [0,1], [0,1]] # [6, 2] : 이항분류 
# 종속변수가 0,1 으로 이진분류

# 2. X, Y변수 정의 : type 일치 - float32
X = tf.constant(x_data, tf.float32) # shape=(6, 2)
y = tf.constant(y_data, tf.float32) # shape=(6, 2)


# 3. w, b변수 정의 : 초기값(난수)  
w = tf.Variable(tf.random.normal(shape=[2, 2]))  # ★ 가중치[입력수,출력수]
b = tf.Variable(tf.random.normal(shape=[2])) 


# 4. 회귀모델 
def linear_model(X) :
    model = tf.linalg.matmul(X, w) + b # 회귀방정식 
    return model 
    
# 5. sigmoid 함수  : 이항분류 활성함수 
def sigmoid_fn(X) :
    model = linear_model(X)
    y_pred = tf.nn.sigmoid(model) 
    return y_pred 
'''
tf.nn : neural network 관련 연산을 포함하는 모듈
'''    

# 6. 손실함수 : cross entropy 이용 
def loss_fn() : # 인수 없음 
    y_pred = sigmoid_fn(X)
    loss = -tf.reduce_mean(y * tf.math.log(y_pred) + (1-y) * tf.math.log(1-y_pred))
    return loss


# 7. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate=0.5)


# 8. 반복학습 
for step in range(100) :
    opt.minimize(loss=loss_fn, var_list=[w, b])
    
    # 10배수 단위 출력 
    if (step+1) % 10 == 0 :
        print('step =', (step+1), ", loss val = ", loss_fn().numpy())
    
'''
step = 10 , loss val =  0.7692073 ... 
step = 100 , loss val =  0.06145263
'''
# 9. 최적화된 model 검증 
# 확률값으로 반환된다. 
y_pred = sigmoid_fn(X) # sigmoid 함수 호출 
print(y_pred.numpy()) # 확률예측 (0~1) 
'''
[[0.9948767  0.00403662]
 [0.90155154 0.08727253]
 [0.88149256 0.113517  ]
 [0.11958554 0.88714075]
 [0.01627312 0.98616326]
 [0.00517596 0.9959896 ]]
'''

# 확률 0.5 이상을 1로 ★★★
# tf.argmax(, axis =1).numpy() 
# 가장 큰 값의 인덱스(위치)를 반환하는 함수
y_pred = tf.argmax(y_pred, axis =1).numpy()
y_true = tf.argmax(y, axis =1).numpy()

y_pred # array([0, 0, 0, 1, 1, 1], dtype=int64)
y_true # array([0, 0, 0, 1, 1, 1], dtype=int64)

# ★★★ 예측 결과 변환 방법 두 가지 
# 확률 -> 0, 1으로 변경 : tf.argmax()
# 확률 -> 원핫인코딩으로 변경 : tf.cast() 
y_pred = tf.cast(y_pred > 0.5, dtype=tf.float32)
y_pred = tf.argmax(y_pred, axis =1).numpy()




