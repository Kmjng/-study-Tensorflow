'''
문3) women.csv 데이터 파일을 이용하여 선형회귀모델 생성하시오.
     <조건1> x변수 : height,  y변수 : weight
     <조건2> learning_rate=0.5
     <조건3> 최적화함수 : Adam
     <조건4> 반복학습 : 200회
     <조건5> 학습과정 출력 : step, loss_value
     <조건6> 최적화 모델 검증 : MSE, 회귀선 시각화  
'''
import tensorflow as tf # ver2.x 

import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

women = pd.read_csv(r'C:/ITWILL/7_Tensorflow/Tensorflow/data/women.csv')
print(women.info())

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 15 entries, 0 to 14
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   height  15 non-null     int64
 1   weight  15 non-null     int64
dtypes: int64(2)
memory usage: 368.0 bytes
None
'''
# 1. x,y data 생성 
x_data = women['height']
y_data = women['weight']

# 정규화 
print(x_data.max()) # 72
print(y_data.max()) # 164

# 2. 정규화(0~1)
X = x_data / 72
Y = y_data / 164

# 자료형 변환 
X = tf.constant(X, dtype=tf.float32)
Y = tf.constant(Y, dtype=tf.float32)
X.shape # TensorShape([15])
X.numpy() # height 독립변수에 대한 1차원 벡터

# 3. w,b변수 정의 - 난수 이용 
# 입력변수 height 1개 이므로 (1,) shape으로 생성
w = tf.Variable(tf.random.uniform([1], 0.1, 1.0)) 
# random.uniform : 균등분포 난수 shape, minval, maxval
b = tf.Variable(tf.random.uniform([1], 0.1, 1.0))


# 4. 회귀모델 
def linear_model(X) : # 입력 X
    y_pred = tf.multiply(X, w) + b # y_pred = X * a + b
    return y_pred

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 
def loss_fn() : #  인수 없음 
    y_pred = linear_model(X) # 예측치 : 회귀방정식  
    err = Y - y_pred # 오차 
    loss = tf.reduce_mean(tf.square(err)) # 오차제곱평균(MSE) 
    return loss

# 6. model 최적화 객체 : 오차의 최소점을 찾는 객체  
optimize = tf.optimizers.Adam(learning_rate = 0.5)

print(f'초기값들\nw:{w.numpy()},\nb:{b.numpy()}')
'''
초기값들
w:[0.969358],
b:[0.5540841]
'''
# 7. 반복학습 : 200회 
for step in range(200):
    optimize.minimize(loss =loss_fn, var_list = [w,b])
    print(f'step:{step+1}\n손실값:{loss_fn().numpy()}')
    print(f'적용된 가중치: {w.numpy()}, 편향: {b.numpy()}')
'''
손실값:0.00022850146342534572
적용된 가중치: [1.308229], 편향: [-0.34690493]
'''
# 8. 최적화된 model 검증
# 1) MSE 평가 
from sklearn.metrics import mean_squared_error
y_pred = linear_model(X.numpy())
mse = mean_squared_error(Y.numpy(), y_pred)
print('rmse:',mse**0.5)
# rmse: 0.015116264382228517

# 2) 회귀선    
import matplotlib.pyplot as plt 

plt.plot(X.numpy(), Y.numpy(), 'bo') # 실제값
plt.plot(X.numpy(), y_pred, 'r-') 
plt.plot()

    
  
    