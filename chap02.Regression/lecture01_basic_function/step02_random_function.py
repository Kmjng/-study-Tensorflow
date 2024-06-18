#######################
## 3. 난수 생성 함수 
#######################
''' 
tf.random.normal(shape, mean, stddev)  : 평균,표준편차 정규분포
tf.truncated.normal(shape, mean, stddev) : 표준편차의 2배 수보다 큰 값은 제거하여 정규분포 생성 
tf.random.uniform(shape, minval, maxval) : 균등분포 난수 생성
tf.random.shuffle(value) : 첫 번째 차원 기준으로 텐서의 원소 섞기
tf.random.set_seed(seed)  : 난수 seed값 설정 
'''

import tensorflow as tf # ver2.x


# 표준정규분포를 따르는 난수 생성(2행3열의 6개 난수)  
norm = tf.random.normal([2,3], mean=0, stddev=1) 
print(norm) # 객체 보기 

uniform = tf.random.uniform([2,3], minval=0, maxval=1) 
print(uniform) # 객체 보기 

matrix = [[1,2], [3,4], [5,6]] # 중첩list : (3, 2)
shuff = tf.random.shuffle(matrix) 
print(shuff) 

# seed값 지정 : 동일한 난수 생성   
tf.random.set_seed(1234)
a = tf.random.uniform([1]) 
b = tf.random.normal([1])  

print('a=',a.numpy())  
print('b=',b.numpy())  

####################################
# 정규분포, 균등분포 차트 시각화
####################################
import matplotlib.pyplot as plt # 시각화 도구 

# 시각화 오류 해결방법 : Fatal Python error: Aborted  
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 정규분포(평균:0, 표준편차:2) 
norm = tf.random.normal([1000], mean=175, stddev=5.5) 
data = norm.numpy()
plt.hist(data) 
plt.show()
 
# 균등분포(0~1) 
uniform = tf.random.uniform([1000], minval=2.5, maxval=5.5) 
data2 = uniform.numpy()
plt.hist(data2) 
plt.show() 
