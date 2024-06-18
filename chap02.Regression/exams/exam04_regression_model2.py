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


# X, y변수 정규화(스케일링) 
X = minmax_scale(X)
y = minmax_scale(y)


# 2. train/test split(70 vs 30)
x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

# 3. w, b변수 정의 : tf.random.normal() 함수 이용 

# 4. 회귀모델 : 행렬곱 

# 5. 비용 함수 정의 : 예측치 > 오차 > 손실함수 

# 6. 최적화 객체 

# 7. 반복학습 

# 8. 최적화된 model 평가





