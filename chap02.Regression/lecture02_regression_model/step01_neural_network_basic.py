# -*- coding: utf-8 -*-
"""
퍼셉트론 구현 : ppt.3 참고  

- 망의 총합(sum of X*w + bias) 에 **활성함수**를 거쳐서 출력
    - 가중치 (w): 기울기
    - 편향 (b) : 절편 ; 모델이 자료에 더 잘 맞추기(fitting) 위해서 기울기를 조절해주는 역할
- 활성함수는 망의 총합을 확률로 예측 :: 활성함수
- 값 그대로 예측 :: 항등함수 (기존의 선형회귀모델)


"""

import numpy as np 


# 1. 신경망 조절변수(W, b)  
def init_variable() :
    variable = {} # dict
    variable['W'] = np.array([[0.1], [0.3]]) # (2, 1)
    variable['b'] = 0.1       
    return variable
    


# 2. 활성함수 : 항등함수
def activation(model) :
    return model 
 

    
# 3. 순방향(forward) 실행 
def forward(variable, X) : # (조절변수, X변수)
    W = variable['W']   
    b = variable['b']    
    model = np.dot(X, W) + b # 망의총합
    y = activation(model) # 활성함수 
    return y



# 프로그램 시작점 
variable = init_variable() # 조절변수(W, b) 생성  
X = np.array([[1.0, 0.5]]) # X변수 (입력변수)
X.shape # (1, 2)
y = forward(variable, X) # 순방향 연산 

print('y_pred =', y) # y_pred = [[0.35]]

'''
★★★

입력변수 차원이 (1,3)이면 가중치 행렬 w은 (3,1)

입력변수가 (1,2,3)형태의 3차원 텐서라면, 가중치 행렬w은 (3,1)

'''

