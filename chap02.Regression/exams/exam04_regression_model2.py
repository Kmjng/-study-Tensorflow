'''
문4) california housing데이터셋을 이용하여 다음과 같이 선형회귀모델 생성하시오.
     <조건1> x변수 : housing.data,  y변수 : housing.target
     <조건2> w변수, b변수 정의 : tf.random.normal() 이용 
     <조건3> learning_rate=0.5
     <조건4> 최적화함수 : Adam
     <조건5> 학습 횟수 100회
     <조건6> 학습과정과 MSE 출력 : <출력결과> 참고 
     
 <출력결과>
step = 10, loss = 0.2613501572917889
step = 20, loss = 0.11251915198288974
step = 30, loss = 0.08120963255748127
step = 40, loss = 0.06318893386605076
step = 50, loss = 0.06312333055042012
step = 60, loss = 0.059066445807798906
step = 70, loss = 0.05835104585754438
step = 80, loss = 0.057889051866334564
step = 90, loss = 0.05770074365070097
step = 100, loss = 0.05747814056010608
========================================
MSE = 0.05434408070431542
'''

import tensorflow as tf # ver2.x
from sklearn.model_selection import train_test_split # datast splits
from sklearn.metrics import mean_squared_error # model 평가 
from sklearn.datasets import fetch_california_housing # dataset 
from sklearn.preprocessing import minmax_scale # 정규화(0~1) 

# 1. data loading
housing = fetch_california_housing()


# 변수 선택 
X = housing.data # x 
y = housing.target # y : 숫자 class(0~2)
X.shape # (20640, 8)

# X, y변수 정규화(스케일링) 
X = minmax_scale(X)
y = minmax_scale(y)

X = X.astype('float32')
# 2. train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

# 3. w, b변수 정의 : tf.random.normal() 함수 이용 
w = tf.Variable(tf.random.normal([8,1]))
b = tf.Variable(tf.random.normal([1]))
# 4. 회귀모델 : 행렬곱 
def linear_model(X): 
    y_pred = tf.linalg.matmul(X, w) + b
    return y_pred


# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 
def loss_fn(): 
    y_pred = linear_model(x_train)  # 함수 안에서 함수 실행
    err = tf.math.subtract(y_train, y_pred)
    loss = tf.reduce_mean(tf.square(err)) # mse
    return loss 


# 6. 최적화 객체 
opt = tf.optimizers.Adam(learning_rate = 0.5)
print(f'초기값: w ={w.numpy()}, b ={b.numpy()}')
# 7. 반복학습 
loss_value = []
for step in range(100):
    opt.minimize(loss = loss_fn, var_list = [w,b])
    loss_value.append(loss_fn().numpy())
    if (step+1)% 10 == 0 : 
        print(f'step = {step+1}, loss value = {loss_fn().numpy()}')
# 8. 최적화된 model 평가
from sklearn.metrics import mean_sqaured_error

# test set에 대한 평가 
y_pred = linear_model(x_test) 
 
print('rmse:',mean_sqaured_error(y_test, y_pred)**0.5)

'''
초기값: w =[[-0.25055045]
 [-0.08064709]
 [ 0.69627273]
 [ 1.601835  ]
 [ 1.6900092 ]
 [ 0.43828106]
 [-0.02178756]
 [-0.80817044]], b =[-1.1120325]
step = 10, loss value = 0.7651374936103821
'''

import matplotlib.pyplot as plt 
plt.plot(loss_value)
plt.xlabel('epochs')
plt.ylabel('Loss value')
plt.show()