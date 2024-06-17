# -*- coding: utf-8 -*-
"""
step03_tensorboard.py

Tensorboard 
 - [Graph 모델 시각화 도구]가 필요함 
 - Tensor의 연산 흐름 확인  

"""

import tensorflow.compat.v1 as tf # ver1.x 사용  
tf.disable_v2_behavior() # ver2.x 사용 안함 

# tensorboard 초기화 
tf.reset_default_graph()


''' Graph 모델 정의 '''
 
# 상수 정의 
x = tf.constant(1)
y = tf.constant(2)

# 사칙연산식 정의 
a = tf.add(x, y, name='a') # a = x + y
b = tf.multiply(a, 6, name='b') # b = a * 6
c = tf.subtract(20, 10, name='c') # c = 20 - 10
d = tf.div(c, 2, name = 'd') # d = c / 2

g = tf.add(b, d, name='g') # g = b + d
h = tf.multiply(g, d, name='h') # h = g * d

''' Graph 모델 실행 '''

with tf.Session() as sess :
    h_calc = sess.run(h) # device 할당 : 연산 
    print('h = ', h_calc) # h =  115
    
    # tensorboard graph 생성
    tf.summary.merge_all() # Graph 모으기  
    writer = tf.summary.FileWriter("C:/ITWILL/7_Tensorflow/graph", sess.graph)
    writer.close()



