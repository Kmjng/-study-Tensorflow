# -*- coding: utf-8 -*-
"""
문4) fashion_mnist 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
    
  조건1> keras layer
       L1 =  (28, 28) x 128
       L2 =  128 x 64
       L3 =  64 x 32
       L4 =  32 x 16
       L5 =  16 x 10
  조건2> output layer 활성함수 : softmax     
  조건3> optimizer = 'Adam',
  조건4> loss = 'categorical_crossentropy'
  조건5> metrics = 'accuracy'
  조건6> epochs = 15, batch_size = 32   
  조건7> model evaluation : validation dataset
"""
from tensorflow.keras.utils import to_categorical # one hot
from tensorflow.keras.datasets import fashion_mnist # fashion
from tensorflow.keras import Sequential # keras model 
from tensorflow.keras.layers import Dense, Flatten # model layer
import matplotlib.pyplot as plt

# 1. MNIST dataset loading
(train_img, train_lab),(val_img, val_lab)=fashion_mnist.load_data() # (images, labels)
train_img.shape # (60000, 28, 28) 
train_lab.shape # (60000,) 
 


# 2. x, y변수 전처리 
# x변수 : 정규화(0~1)
train_img = train_img / 255.
val_img = val_img / 255.
train_img[0] # first image(0~1)
val_img[0] # first image(0~1)


# y변수 : one hot encoding 
train_lab = to_categorical(train_lab)
val_lab = to_categorical(val_lab)
val_lab.shape # (10000, 10)


from tensorflow.keras.layers import Input # input layer
from tensorflow.keras.models import Model # DNN Model 생성

input_dim = (28,28) # 2차원 이미지 자료 
output_dim = 10

# 3. keras model & layer 구축(Functional API 방식) 

# 1) input layer
inputs = Input(shape=(input_dim)) # 입력층 

# flatten layer : 2d -> 1d  
flatten = Flatten(input_shape = input_dim)(inputs)  

# 2) hidden layer1
hidden1 = Dense(units=128, activation='relu')(flatten) # 1층

# 3) hidden layer2
hidden2 = Dense(units=64, activation='relu')(hidden1) # 2층

# 3) hidden layer3
hidden3 = Dense(units=32, activation='relu')(hidden2) # 3층

# 3) hidden layer4
hidden4 = Dense(units=16, activation='relu')(hidden3) # 4층

# 4) output layer
outputs = Dense(units=output_dim, activation ='softmax')(hidden4)# 5층 

# model 생성 
model = Model(inputs, outputs) # Model 클래스 이용

model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_5 (InputLayer)        [(None, 28, 28)]          0      -> 입력층    
                                                                 
 flatten_3 (Flatten)         (None, 784)               0      -> 2d -> 1d    
                                                                 
 dense_41 (Dense)            (None, 128)               100480 -> 은닉층1   
                                                                 
 dense_42 (Dense)            (None, 64)                8256   -> 은닉층2    
                                                                 
 dense_43 (Dense)            (None, 32)                2080   -> 은닉층3    
                                                                 
 dense_44 (Dense)            (None, 16)                528    -> 은닉층4    
                                                                 
 dense_45 (Dense)            (None, 10)                170    -> 출력층    
                                                                 
=================================================================
Total params: 111,514
'''



# 4. model compile : 학습환경  
model.compile(optimizer='adam', 
              loss = 'categorical_crossentropy', 
              metrics=['accuracy'])


# 5. model training  : 학습 
model.fit(x=train_img, y=train_lab, # 훈련셋 
          epochs=15, # 반복학습 15*60000=900,000장 이미지 
          verbose=1, # 출력여부
          batch_size = 32, # 32*1875 = 60,000(1epoch)
          validation_data=(val_img, val_lab)) # 검증셋

'''
Epoch 15/15
1875/1875 : 반복횟수  
- loss: 0.2179 - accuracy: 0.9184 - val_loss: 0.3541 - val_accuracy: 0.8854
'''

# 6. model evaluation : validation dataset
print('='*30)
print('model evaluation')
model.evaluate(val_img, val_lab)











