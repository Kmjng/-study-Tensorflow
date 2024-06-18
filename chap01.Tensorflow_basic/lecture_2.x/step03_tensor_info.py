'''
step03_tensor_info.py

Tensor 정보 제공 함수 
 1. tensor shape
 2. tensor rank
 3. tensor size
 4. tensor reshape 
'''

import tensorflow as tf
print(tf.__version__) # 2.3.0

scalar = tf.constant(1234) # 상수 
vector = tf.constant([1,2,3,4,5]) # 1차원 
matrix = tf.constant([ [1,2,3], [4,5,6] ]) # 2차원
cube = tf.constant([[ [1,2,3], [4,5,6], [7,8,9] ]]) # 3차원 

print(scalar)
print(vector)
print(matrix)
print(cube)

# 1. tensor shape 

print('\ntensor shape')
print(scalar.get_shape()) # () scalar.shape과 같은 의미 
print(vector.get_shape()) # (5,)
print(matrix.get_shape()) # (2, 3)
print(cube.get_shape()) # (1, 3, 3)

  
# 2. tensor rank 
print('\ntensor rank') # 차원
print(tf.rank(scalar)) # tf.Tensor(0, shape=(), dtype=int32)
print(tf.rank(vector)) # tf.Tensor(1, shape=(), dtype=int32)
print(tf.rank(matrix)) # tf.Tensor(2, shape=(), dtype=int32)
print(tf.rank(cube))   # tf.Tensor(3, shape=(), dtype=int32)

# 3. tensor size
print('\ntensor size') # 원소 수 
print(tf.size(scalar)) # tf.Tensor(1, shape=(), dtype=int32)
print(tf.size(vector)) # tf.Tensor(5, shape=(), dtype=int32)
print(tf.size(matrix)) # tf.Tensor(6, shape=(), dtype=int32)
print(tf.size(cube))   # tf.Tensor(9, shape=(), dtype=int32)


dir(tf)

# 4. tensor reshape 
cube
cube.shape # TensorShape([1, 3, 3])
cube_2D = tf.reshape(cube, (3,3))
cube_2D









