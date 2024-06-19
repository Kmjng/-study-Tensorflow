# -*- coding: utf-8 -*-
"""
딥러닝 최적화 알고리즘 이용 단순선형회귀모델 + csv file 
"""

import tensorflow as tf # 최적화 알고리즘 
import pandas as pd  # csv file 
from sklearn.preprocessing import minmax_scale # 정규화 
from sklearn.metrics import mean_squared_error # model 평가 

iris = pd.read_csv('C:/ITWILL/7_Tensorflow/Tensorflow/data/iris.csv')
print(iris.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None
'''

# 1. X, y data 생성
x_data = iris['Sepal.Length'] 
y_data = iris['Petal.Length']

# 2. X, y변수 만들기     
# ★★ 주의: weight의 자료형이 보통 float32 이기 때문에 자료형을 맞춰준다.  
X = tf.constant(x_data, dtype=tf.float32) # dtype 지정 
y = tf.constant(y_data, dtype=tf.float32) # dtype 지정 


# 3. a,b 변수 정의 : 초기값 - 난수  
tf.random.set_seed(123)
w = tf.Variable(tf.random.normal([1])) # 가중치 
b = tf.Variable(tf.random.normal([1])) # 편향 


# 4. 회귀모델 
def linear_model(X) : # 입력 : X -> y예측치 
    y_pred = tf.math.multiply(X, w) + b # 회귀방정식 
    return y_pred 


# 5. 손실/비용 함수(loss/cost function) : 손실반환(MSE)
def loss_fn() : # 인수 없음 
    y_pred = linear_model(X) # 예측치 
    err = tf.math.subtract(y, y_pred) # 정답 - 예측치  
    loss = tf.reduce_mean(tf.square(err)) # MSE  
    return loss
'''
 loss_fn 함수 내에서 계산된 MSE는
 현재 가중치 w와 편향 b에 대해 계산된 손실 값
'''

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체 
optimizer = tf.optimizers.Adam(learning_rate=0.5) # lr : 0.9 ~ 0.0001
# learning rate Defaults to 0.001 
print(f'기울기(w) 초기값 = {w.numpy()}, 절편(b) 초기값 = {b.numpy()}')

# 7. 반복학습 : 100회
for step in range(300) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b])#(손실값, 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
    # a, b 변수 update 
    print(f'적용된 기울기(w) = {w.numpy()}, 절편(b) = {b.numpy()}')

'''
step = 100 , loss value = 1.4621323
기울기(w) = [0.8448554], 절편(b) = [-1.039748]

# 반복학습 300하면 ? 
step = 300 , loss value = 0.74557996
기울기(w) = [1.7980571], 절편(b) = [-6.742635]

'''
# 8. 최적화된 model 평가 
y_pred = linear_model(X.numpy())
# linear_model()에 최적화된 w,b 가 들어가 있음 
mse = mean_squared_error(y_true = y.numpy(), y_pred=y_pred)

print('rmse:', mse**0.5) # rmse: 9.002011392263693