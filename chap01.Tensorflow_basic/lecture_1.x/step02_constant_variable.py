# -*- coding: utf-8 -*-
"""
step02_variable.py

Tensorflow 상수와 변수 정의 
"""

# Tensorflow code 
import tensorflow.compat.v1 as tf # ver1.x -> ver2.x 마이그레이션 
tf.disable_v2_behavior() # ver2.x 사용 안함 

''' Graph 모델 정의 '''
# 상수 정의  
x = tf.constant([1.5, 2.5, 3.5]) # 상수는 수정 불가능 

# 변수 정의  
y = tf.Variable([1.0, 2.0, 3.0])  # 변수는 수정 가능 
y # ...shape=(3,)...

''' Graph 모델 실행 '''
with tf.Session() as sess : # 세션 객체 생성     
    print('x =', sess.run(x)) # 상수 실행 
    
    # 변수를 사용하기 전에 초기화 과정을 거쳐야 함 ★★★
    sess.run(tf.global_variables_initializer()) #변수 초기화 
    # 단 ver1에서만 가능
    print('y=', sess.run(y)) # 변수 실행  
    
'''
x = [1.5 2.5 3.5]
y= [1. 2. 3.]
'''



