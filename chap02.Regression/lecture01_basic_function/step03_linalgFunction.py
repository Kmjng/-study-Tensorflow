'''
선형대수 연산 함수  
  단위행렬 -> tf.linalg.eye(dim) 
  정방행렬의 대각행렬 -> tf.linalg.diag(x)  
  정방행렬의 행렬식 -> tf.linalg.det(x)
  정방행렬의 역행렬 -> tf.linalg.inv(x)
  두 텐서의 행렬곱 -> tf.linalg.matmul(x, y) # numpy의 dot()와 동일함
'''

import tensorflow as tf
import numpy as np

# 정방행렬 데이터 생성 
x = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 
y = np.random.rand(2, 2) # 지정한 shape에 따라서  0~1 난수 


eye = tf.linalg.eye(2) # 2*2 단위행렬
print(eye.numpy()) 
 

dia = tf.linalg.diag(x) # 대각행렬 
mat_deter = tf.linalg.det(x) # 정방행렬의 행렬식  
mat_inver = tf.linalg.inv(x) # 정방행렬의 역행렬
mat = tf.linalg.matmul(x, y) # 행렬곱 반환 

print(x)
print(dia.numpy()) 
print(mat_deter.numpy())
print(mat_inver.numpy())
print(mat.numpy())


## 행렬곱 
A = tf.constant([[1,2,3], [3,4,2], [3,2,5]]) # A행렬 
B = tf.constant([[15,3, 5], [3, 4, 2]]) # B행렬  

A.get_shape() # [3, 3]
B.get_shape() # [2, 3]

# 행렬곱 연산 
mat_mul = tf.linalg.matmul(a=A, b=B)
print(mat_mul.numpy())


'''
첫번째 입력에서 연산되어 값이 저장된 노드(뉴런)를 Hidden Layer라 한다. 
입력이 2개, H1노드가 3개이면 
가중치(W)가 총 6개임 
'''


