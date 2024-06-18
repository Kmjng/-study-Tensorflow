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
# 내부적으로 그래프가 생성됨 
result_graph = add_graph_mode(a, b)
print("Graph mode result:", result_graph)  


# 수업 外

# 로그 디렉토리 설정
import datetime
log_dir = "C:/ITWILL/7_Tensorflow/graph" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# 그래프 모드 실행 및 기록
with writer.as_default():
    tf.summary.trace_on(graph=True, profiler=True)
    result_graph = add_graph_mode(a, b)
    tf.summary.trace_export(name="add_graph_mode_trace", step=0, profiler_outdir=log_dir)

# "C:\ITWILL\7_Tensorflow\graph20240617-153309" 에 로그파일 생성됨

