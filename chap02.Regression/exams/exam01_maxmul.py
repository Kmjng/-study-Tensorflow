'''
문1) 다음과 같이 X, a 행렬을 상수로 정의하고 행렬곱으로 연산하시오.
    단계1 : X, a 행렬의 상수 정의  
        X 행렬 : iris 2번~4번 칼럼으로 상수 정의(자료형 : dtype = 'float32') 
        a 행렬 : [[0.2],[0.1],[0.3]] 값으로 상수 정의(자료형 : dtype = 'float32')  
    단계2 : 행렬곱 : y 계산하기  
        y = X @ a
    단계3 : y 결과 출력
'''

import tensorflow as tf
import pandas as pd 


iris = pd.read_csv(r"C:\ITWILL\7_Tensorflow\Tensorflow\data\iris.csv")
iris.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64  
 1   Sepal.Width   150 non-null    float64
 2   Petal.Length  150 non-null    float64
 3   Petal.Width   150 non-null    float64
 4   Species       150 non-null    object 
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
'''
#  단계1 : X, a 상수 정의 
X = tf.constant(value=iris.iloc[:, 1:-1], dtype="float32") # 2번~4번 칼럼
X.shape # TensorShape([150, 3])
a = tf.constant(value=[[0.2],[0.1],[0.3]], dtype='float32') # 가중치 (기울기) 초기화 


# 단계2 : 행렬곱 식 정의 
y = tf.linalg.matmul(X,a) # y = X @ a

# 단계3 : 행렬곱 결과 출력 
y
