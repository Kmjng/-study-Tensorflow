# -*- coding: utf-8 -*-
"""
step02_@tf.function.py

@tf.function 함수 장식자
 - ver2.x에서 그래프 모드 지원 
 - 함수내에서 python code 작성 지원 
"""

import tensorflow as tf

def add_eager_mode(a, b): # 즉시 실행 모드 
    return a + b


@tf.function # 그래프 모드 지원 
def add_graph_mode(a, b):
    return a + b


# 실인수 
a = tf.constant(2)
b = tf.constant(3)


# 즉시 실행 모드
result_eager = add_eager_mode(a, b)
print("Eager mode result:", result_eager)  


# 그래프 모드
result_graph = add_graph_mode(a, b)
print("Graph mode result:", result_graph)  

















