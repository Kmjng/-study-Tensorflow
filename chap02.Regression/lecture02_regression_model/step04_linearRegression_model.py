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
dir(tf.optimizers)
'''
['Adadelta',
 'Adagrad',
 'Adam',
 'Adamax',
 'Ftrl',
 'Nadam',
 'Optimizer',
 'RMSprop',
 'SGD',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '_sys',
 'deserialize',
 'experimental',
 'get',
 'legacy',
 'schedules',
 'serialize']
'''

# 6. 반복학습 
for step in range(100) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b]) # (손실값, 적용된 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
        
    # a, b 변수 update 
    print(f'가중치(w) = {w.numpy()}, 편향(b) = {b.numpy()}')

'''
step = 1 , loss value = 0.1612598
가중치(w) = [1.5335835], 편향(b) = [1.0602307]
step = 2 , loss value = 0.16048528
가중치(w) = [1.5347065], 편향(b) = [1.0576828]
step = 3 , loss value = 0.15971468
가중치(w) = [1.5358266], 편향(b) = [1.0551409]
...
step = 100 , loss value = 0.1620378
가중치(w) = [1.5324576], 편향(b) = [1.0627847]
'''


# Leaning Rate = 0.1 로 할 경우 , 
# (learning rate가 높을 수록, 경사하강(optimize) step 폭이 넓어짐) ★★★
optimizer = tf.optimizers.SGD(learning_rate=0.1) # 딥러닝 최적화 알고리즘
for step in range(100) :
    optimizer.minimize(loss=loss_fn, var_list=[w, b]) # (손실값, 적용된 조절변수)
    
    # step 단위 -> 손실값 -> a,b 출력 
    print('step =', (step+1), ", loss value =", loss_fn().numpy())
        
    # a, b 변수 update 
    print(f'가중치(w) = {w.numpy()}, 편향(b) = {b.numpy()}')
'''
step = 100 , loss value = 0.0007709505
가중치(w) = [1.9677514], 편향(b) = [0.07330845]
'''

# 7. 최적화된 model 검증 
'''
위에서 optimize를 통해 조절된 w, b 파라미터를 토대로 검증
'''
X_test = [2.5] # test set

y_pred = linear_model(X_test)
print(y_pred.numpy()) # [4.9906974]
# X 에 2.5가 들어갔을 때 y가 4.9906로 최적화되어 출력됨 
'''
X = tf.constant([1, 2, 3], dtype=tf.float32) 
y = tf.constant([2, 4, 6], dtype=tf.float32) 
'''

# 전체 dataset 
y_pred = linear_model(X) # [1,2,3]
print('예측치:',y_pred.numpy())
print('실제값:', y.numpy())
'''
예측치: [2.052233  4.011209  5.9701853]
실제값: [2. 4. 6.]
'''

#################
### 회귀선 시각화 
#################
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

plt.plot(X.numpy(),y.numpy(), 'bo') # blue dot marker # 산점도
plt.plot(X.numpy(), y_pred.numpy(), 'r-') # red line # 회귀선
plt.show() 







