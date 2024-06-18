'''
step04_math_module.py

 <수학 관련 주요 함수> 
version 1.x   -> version 2.x
tf.add() -> tf.math.add() 변경 
tf.subtract() -> tf.math.subtract() 변경 
tf.multiply() -> tf.math.multiply() 변경 
tf.div() -> tf.math.divide() 변경 
tf.mod() : 나머지 -> tf.math.mod() 변경 
tf.abs() : 절대값 -> tf.math.abs() 변경 
tf.square() : 제곱  -> tf.math.square() 변경
tf.sqrt() : 제곱근  -> tf.math.sqrt() 변경
tf.round() : 반올림  -> tf.math.round() 변경
tf.pow() : 거듭제곱 -> tf.math.pow() 변경
tf.exp() : 지수값 -> tf.math.exp() 변경
tf.log() : 로그값 -> tf.math.log() 변경
'''

import tensorflow as tf

x = tf.constant([1,2,-3,4])
y = tf.constant([5,6,7,8])


# 덧셈/뺄샘/나눗셈/곱셈
print(tf.math.add(x, y, name='adder'))
print(tf.math.subtract(x, y, name='adder'))
print(tf.math.multiply(x, y, name='adder'))
print(tf.math.divide(x, y, name='divide'))
print(tf.math.mod(x, y, name='mod')) # [1 2 4 4]

# 음수, 부호 반환 
print('tf.neg=', tf.math.negative(x)) # [-1 -2  3 -4]
print('tf.sign=', tf.math.sign(x)) # [ 1  1 -1  1]

# 제곱/제곱근/거듭제곱 
print(tf.math.abs(x)) # [1 2 3 4]
print(tf.math.square(x)) # 제곱 - [ 1  4  9 16]
print(tf.math.sqrt([4.0, 9.0, 6.0])) # 제곱근
print(tf.math.pow(x, 3)) # 거듭제곱-[  1   8 -27  64]

# 지수와 로그 
print('e=', tf.math.exp(1.0).numpy()) # e= 2.7182817
print(tf.math.exp(2.0)) 
print(tf.math.log(8.0)) # 밑수e 자연로그



########################################
## 지수 함수를 이용한 sigmoid 활성함수 
########################################
 
import numpy as np 

def sigmoid(x) : # x : 입력변수 
    ex = tf.math.exp(-x)
    y = 1 / (1 + ex)
    return y # y : 출력변수(예측치)


x_data = np.array([1.0, 2.0, 5.0]) # 입력자료


# 함수 호출 
y1 = sigmoid(x_data)
print(y1.numpy())# [0.73105858 0.88079708 0.99330715] : 확률 


########################################
## 지수 함수를 이용한 softmax 활성함수 
########################################
    
# 함수 정의  
def softmax(x) :    
    ex = tf.math.exp(x - x.max())
    y = ex / sum(ex.numpy())
    return y


# 함수 호출 
y2 = softmax(x_data)
print(y2.numpy())# [0.01714783 0.04661262 0.93623955] : 확률 

