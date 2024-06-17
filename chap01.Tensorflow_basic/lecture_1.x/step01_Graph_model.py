# -*- coding: utf-8 -*-
"""
step01_tensorflow_basic.py

- tensorflow ver1.x 작업 환경
TensorFlow 1.x: 
    기본적으로 그래프 모드를 사용하여 연산을 정의하고 실행
    Session을 생성하고 그래프를 실행
TensorFlow 2.x: 
    즉시 실행(eager execution) 모드가 기본으로 활성화되어 있어 
    Python과 유사하게 동작함. 
    (즉, 연산이 발생하는 즉시 결과를 반환)



- Graph 모델 정의 & 실행  
<Graph 구성요소>
- node : 그래프를 구성하는 연산자(Operation) ; 타원 표기 
- edge : 그래프를 구성하는 다차원배열(Tensor) ; 화살표 표기 

<매커니즘>
- 프로그램(Graph)을 세션(Session)을 통해 Device(CPU1,..GPU1)에 배정
- 세션이라는 객체로 이원화되어 있다. 

"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver2.x 환경에서 ver1.x 사용
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' Graph 모델 정의 ''' 
# 1. 프로그램 정의 영역 : 모델 구성 

# (1) 상수 정의 
x = tf.constant(10)  
y = tf.constant(20)  
x # <tf.Tensor 'Const:0' shape=() dtype=int32>
y # <tf.Tensor 'Const_1:0' shape=() dtype=int32>

# (2) 식 정의 
z = tf.add(x, y) 
z # <tf.Tensor 'Add:0' shape=() dtype=int32>
'''
<Tensorflow 사칙연산 함수>  
adder = tf.Variable(a + b)
subtract = tf.Variable(a - b)
multiply = tf.Variable(a * b)
divide = tf.Variable(a / b)
'''


''' Graph 모델 실행 ''' 
# 2. 프로그램 실행 영역: 모델 실행 
with tf.Session() as sess : # 세션 생성 
    print('x=', sess.run(x)) # 세션 할당 
    print('y=', sess.run(y))
    print('z=', sess.run(z)) 
'''
sess.run(z))
x= 10
y= 20
z= 30
'''





