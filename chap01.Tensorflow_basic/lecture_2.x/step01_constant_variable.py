# -*- coding: utf-8 -*-
"""
Tensorflow ver2. 특징 
✓ 단순성과 편의성에 초점을 두고 업그레이드
✓ 즉시 실행(eager execution) : session 없이 코드 실행
✓ @tf.function : AutoGraph 지원
✓ Python 코드 지원
✓ Keras 모듈 : tensorflow 머신러닝(딥러닝) 모델 제공
✓ 중복된 API 정리 및 단순화
✓ Tensorflow 1.x 기능 제공
✓ 플랫폼에 독립적인 탄탄한(robust) 모델 배포

step01_constant_variable.py

1. Tensorflow 상수와 변수 
2. 즉시 실행(eager execution) 모드
 - session 사용 없이 자동으로 컴파일 
 - python 처럼 즉시 실행하는 모드 제공(python 코드 사용 권장)
 - API 정리 : tf.global_variables_initializer() 삭제됨 
"""


import tensorflow as tf # ver 2.x
print(tf.__version__) 


# 즉시 실행 모드 
# 프로그램 정의+실행 함께 진행 
tf.executing_eagerly() # 기본(default)으로 활성화 됨  


# 상수 정의  
x = tf.constant(value = [1.5, 2.5, 3.5]) # 1차원   
print('x =', x) 
# x = tf.Tensor([1.5 2.5 3.5], shape=(3,), dtype=float32)


# 변수 정의  
y = tf.Variable([1.0, 2.0, 3.0]) # 1차원  
print('y =', y)
# y = <tf.Variable 'Variable:0' shape=(3,) dtype=float32, numpy=array([1., 2., 3.], dtype=float32)>

# 식 정의 : 상수 or 변수 참조 
mul = tf.math.multiply(x, y) # x * y 
print('mul =', mul) 
# mul = tf.Tensor([ 1.5  5.  10.5], shape=(3,), dtype=float32)

# 연산 결과만 확인하려면? numpy()
val = mul.numpy() # 연산 결과만 볼 수 있음
# [ 1.5  5.  10.5]

