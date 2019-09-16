#학습 모델 생성 소스

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\after_day.xlsx",header=0)

datasets = df.values
X = datasets[:, 0:4]
Y = datasets[:, 4:]

minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X)
scaled_X = minmax_scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=0.2, random_state=seed)
print(X_test)
model = Sequential()
model.add(Dense(124, input_dim=4, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(124, activation='relu'))
model.add(Dense(1))

model.compile(loss = 'mean_absolute_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=200, batch_size=10)


Y_prediction = model.predict(X_test) #.flatten()
sum = 0

for i in range(len(Y_test)):
    label = Y_test[i]
    prediction = Y_prediction[i]

    print(label, prediction)

model.save('model_after')

'''
Y_prediction = model.predict(X_test).flatten()
sum = 0
for i in range(len(Y_test)):
    label = Y_test[i]
    prediction = Y_prediction[i]

    if label > prediction:
        ratio = prediction / label
    else:
        ratio = label / prediction

    print("실제 관객수: {:.0f}, 예상관객수:{:.0f}, 정확도:{:.4f}".format(label, prediction, ratio))
    sum += ratio
print("평균 정확도: {:.4f}%".format((sum/len(Y_test))*100))
'''

