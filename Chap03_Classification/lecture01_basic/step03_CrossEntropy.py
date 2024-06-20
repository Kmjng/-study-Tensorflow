"""
엔트로피(Entropy) 
 - 확률변수 p에 대한 불확실성의 측정 지수 
 - 값이 클 수록 일정한 방향성과 규칙성이 없는 무질서(chaos) 의미
 - Entropy = -sum(p * log(p))
"""

import numpy as np

# 1. 불확실성이 큰 경우(p1: 앞면, p2: 뒷면)
p1 = 0.5; p2 = 0.5

entropy = -sum([p1 * np.log2(p1), p2 * np.log2(p2)])  
print('entropy =', entropy) # entropy = 1.0


# 2. 불확실성이 작은 경우(x1: 앞면, x2: 뒷면) 
p1 = 0.9; p2 = 0.1

entropy2 = -sum([p1 * np.log2(p1), p2 * np.log2(p2)])
print('entropy2 =', entropy2) # entropy = 0.468995


'''
Cross Entropy    
  - 두 확률변수 x와 y가 있을 때 x를 관찰한 후 y에 대한 불확실성 측정
  - Cross 의미 :  y=1, y=0 일때 서로 교차하여 손실 계산 
  - 식 = -( y * log(y_pred) + (1-y) * log(1-y_pred))
          => (y가 1일 때) + (y가 0일 때)
          =>   ~ (y정답*y예측) ~

  왼쪽 식 : y * log(y_pred) -> y=1 일 때 손실값 계산  
  오른쪽 식 : (1-y) * log(1-y_pred) -> y=0 일 때 손실값 계산 
'''

import tensorflow as tf 

y_preds = [0.02, 0.98] # model 예측값

y = 1 # 정답
for y_pred in y_preds :
    loss_val = -(y * tf.math.log(y_pred)) # y=1 일때 손실값 
    print(loss_val.numpy())
'''
3.912023
0.020202687

y정답: 1, model예측값: 0.02 => 손실 3.912023
'''


y = 0 # 정답
for y_pred in y_preds :
    loss_val = -((1-y) * tf.math.log(1-y_pred)) # y=0 일때 손실값 
    print(loss_val.numpy())



