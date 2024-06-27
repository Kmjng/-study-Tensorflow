# -*- coding: utf-8 -*-
"""

스팸 메시지 분류기 : ppt.33 참고 
"""

# texts 처리 
import pandas as pd # csv file
import numpy as np # list -> numpy 
import string # texts 전처리  
from sklearn.model_selection import train_test_split # split
from tensorflow.keras.preprocessing.text import Tokenizer # 토큰 생성기 
from tensorflow.keras.preprocessing.sequence import pad_sequences # 패딩 

# DNN model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM # 순환신경망 

# 1. csv file laod 
path = r'C:\ITWILL\7_Tensorflow\data'
spam_data = pd.read_csv(path + '/spam_data.csv', header = None)


label = spam_data[0] 
texts = spam_data[1]


# 2. texts와 label 전처리

# 1) label 전처리 
label = [1 if lab=='spam' else 0  for lab in label]

# list -> numpy 형변환 
label = np.array(label)

# 2) texts 전처리 
def text_prepro(texts): # [text_sample.txt] 참고 
    # Lower case
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return texts


# 함수 호출 
texts = text_prepro(texts)
print(texts)


# 3. 단어 생성기 
tokenizer = Tokenizer()  

tokenizer.fit_on_texts(texts = texts) # 텍스트 반영 -> token 생성  

words = tokenizer.index_word # 단어 반환 
print(words)

words_size = len(words) + 1 # 전체 단어수+1 



# 4. 정수색인 : 단어 고유숫자 변환 
seq_result = tokenizer.texts_to_sequences(texts)
print(seq_result)

# 최대 단어길이 
lens = [len(sent) for sent in seq_result]
print(lens)

maxlen = max(lens)


# 5. padding : maxlen 기준으로 모든 문장의 단어 길이 맞춤 
x_data = pad_sequences(seq_result, maxlen = maxlen)
x_data.shape # (5574, 171) 


# 6. train/test split : 80% vs 20%
x_train, x_val, y_train, y_val = train_test_split(
    x_data, label, test_size=20)


# 임베딩 차원 : 16, 32, 64, 128,…1024차원(단어가 많은 경우)
embedding_dim = 32 


# 7. DNN model & layer
model = Sequential()  

# Embedding layer : 1층 
model.add(Embedding(input_dim=words_size, 
                    output_dim=embedding_dim, 
                    input_length=maxlen))


# 순환신경망(RNN layer) 
model.add(LSTM(units= 64, activation='tanh')) # 2층 


# hidden layer1 : w[64, 32] 
model.add(Dense(units=32,  activation='relu')) # 3층 

# output layer : [32, 1]
model.add(Dense(units = 1, activation='sigmoid')) # 4층 
          

# 8. model compile : 학습과정 설정(이항분류기)
model.compile(optimizer='adam', 
              loss = 'binary_crossentropy', 
              metrics=['accuracy'])

# 9. model training : train(80) vs val(20) 
model.fit(x=x_train, y=y_train, # 훈련셋 
          epochs=5, # 반복학습 
          batch_size = 512,
          verbose=1, # 출력여부 
          validation_data=(x_val, y_val)) # 검증셋 


# 10. model evaluation : val dataset 
print('='*30)
print('model evaluation')
model.evaluate(x=x_val, y=y_val)
