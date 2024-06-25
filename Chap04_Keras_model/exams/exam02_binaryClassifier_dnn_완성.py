'''
문2) breast_cancer 데이터셋을 이용하여 다음과 같이 keras 모델을 생성하시오.
  조건1> keras layer
       L1 =  30 x 64
       L2 =  64 x 32
       L3 =  32 x 2
  조건2> optimizer = 'adam',
  조건3> loss = 'binary_crossentropy'
  조건4> metrics = 'accuracy'
  조건5> epochs = 30 
'''

from sklearn.datasets import load_breast_cancer # data set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale # 정규화 
from tensorflow.keras.utils import to_categorical # one hot encoding

# 1. breast_cancer data load
cancer = load_breast_cancer()

x_data = cancer.data
y_data = cancer.target
print(x_data.shape) # (569, 30) : matrix
print(y_data.shape) # (569,) : vector

# x_data : 정규화 
x_data = minmax_scale(x_data) # 0~1

# y변수 one-hot-encoding 
y_one_hot = to_categorical(y_data)
y_one_hot.shape # (569, 2)


# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


from tensorflow.keras import Sequential # model 생성
from tensorflow.keras.layers import Input, Dense # layer 추가 
from tensorflow.keras.models import Model # DNN Model 생성 

# 3. keras model & layer 구축(Functional API 방식) 
#################################################
## 1. Sequential API 방식 : 초보자용
#################################################
#model = Sequential()

# 1층 : hidden1 : w[30,64], b=30
#model.add(Dense(units=64, input_shape=(30,), activation='relu'))

# 2층 : hidden2 : w[64,32], b=32
#model.add(Dense(units=32, activation='relu'))

# 3층 : output  : w[32,2], b=2
#model.add(Dense(units=2, activation='sigmoid'))

#model.summary()
'''
=================================================================
 dense_31 (Dense)            (None, 64)                1984      
                                                                 
 dense_32 (Dense)            (None, 32)                2080      
                                                                 
 dense_33 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 4,130
'''


#################################################
## 2. Functional API 방식 : 개발자용(엔지니어)
#################################################
input_dim = 30
output_dim = 2

# 1) input layer
inputs = Input(shape=(input_dim,)) # Input 클래스 이용

# 2) hidden layer1
hidden1 = Dense(units=64, activation='relu')(inputs) # 1층

# 3) hidden layer2
hidden2 = Dense(units=32, activation='relu')(hidden1) # 2층

# 4) output layer
outputs = Dense(units=output_dim, activation ='sigmoid')(hidden2)

# model 생성 
model = Model(inputs, outputs)
model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 30)]              0         
                                                                 
 dense_43 (Dense)            (None, 64)                1984      
                                                                 
 dense_44 (Dense)            (None, 32)                2080      
                                                                 
 dense_45 (Dense)            (None, 2)                 66        
                                                                 
=================================================================
Total params: 4,130
'''


# 4. model compile 
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# 5. model training : training dataset
model.fit(x_train, y_train,
          epochs = 30,
          validation_data=(x_val, y_val))

'''
Epoch 30/30
13/13 [==============================] - 0s 5ms/step -
 loss: 0.0812 - accuracy: 0.9724 - val_loss: 0.0858 - val_accuracy: 0.9649
'''
# 6. model evaluation : validation dataset
model.evaluate(x_val, y_val)
'''
6/6 [==============================] - 0s 1ms/step - 
loss: 0.0858 - accuracy: 0.9649
'''






