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
y_one_hot
'''
array([[1., 0.],
       [1., 0.],
       [1., 0.],
       ...,
       [1., 0.],
       [1., 0.],
       [0., 1.]], dtype=float32)
'''
# 2. 공급 data 생성 : 훈련용, 검증용 
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_one_hot, test_size = 0.3)


from tensorflow.keras.layers import Input, Dense # layer 추가 
from tensorflow.keras.models import Model # DNN Model 생성 
'''
<keras.models.>
Sequential API는 단순히 레이어를 쌓아 올리는 방식
Model (Functional API) 은 비순차적인 구조에서 사용 
'''
# 3. keras model & layer 구축(Functional API 방식) 
model = Model()

model.add(Dense(units=64, input_shape =(30, ), activation = 'relu')) # 1층 
model.add(Dense(units=32, activation = 'relu')) # 2층 
model.add(Dense(units=2, activation = 'sigmoid')) # 3층 


# 4. model compile 
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy',  
              metrics=['accuracy'])

# 5. model training : training dataset
model.fit(x=x_train, y=y_train, 
          epochs=30,  
          verbose=1,  
          validation_data=(x_val, y_val)) 


# 6. model evaluation : validation dataset
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)

