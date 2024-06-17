# -*- coding: utf-8 -*-
"""
tensorflow import test
"""

import tensorflow as tf  
print(tf.__version__) # 2.10.0 : tensorflow버전 

# keras에서 제공하는 MNIST 데이터셋
mnist = tf.keras.datasets.mnist # dataset 로드 
# 다운로드 진행됨 [==============================] - 0s 0us/step

train, test = mnist.load_data()

# X와 y 나누기 
X_train, y_train = train
X_train.shape # (60000, 28, 28) # 60000장의 이미지 
y_train.shape # (60000,)

X_test, y_test = test
X_test.shape # (10000, 28, 28)
y_test.shape # (10000,)
