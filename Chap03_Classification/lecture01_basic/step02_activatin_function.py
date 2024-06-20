'''
활성함수(activation function)
 - model의 결과를 출력 y로 활성화 시키는 비선형 함수 
 - 유형 : sigmoid, softmax 
'''

import tensorflow as tf
import numpy as np

### 1. sigmoid function : 이항분류
def sigmoid_fn(x) : # x : 입력변수 
    ex = tf.math.exp(-x)   
    y = 1 / (1 + ex)
    return y # y : 출력변수(예측치)    


for x in np.arange(-5.0, 6.0) : # -5 ~ 5 사이
    y = sigmoid_fn(x)  
    print(f"x : {x} -> y : {y.numpy()}")
        
    
### 2. softmax function : 다항분류
def softmax_fn(x) :    
    ex = tf.math.exp(x - x.max())
    #print(ex.numpy())
    y = ex / ex.numpy().sum()
    return y


x_data = np.arange(1.0, 6.0) # 1~5
y = softmax_fn(x_data)  
print(y.numpy())
 
    
 
    
 
    
 
    
 