import numpy as np  ## 백터, 행렬 데이터 전문 모듈 numpy
import pandas as pd  ## 고수준 데이터 모델 (DataFrame) 모듈 pandas
import scipy.stats as stats      ## 통계 등 과학용 모듈 scipy
import tensorflow as tf
from scipy.stats import pearsonr

data = pd.read_csv('C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day66.csv', sep=',')
data = data.dropna()
movie_y = data["최종관객수"]
movie_x = data.drop(['최종관객수'], axis=1)

x_input = movie_x
a = np.array([movie_y])
y_output = a.T

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, test_size=0.2, random_state=42)

def normalize_features(dataset):
    data_min = np.min(dataset,axis=0)
    data_max = np.max(dataset,axis=0)
    return (dataset - data_min) / (data_max - data_min)

normed_x_train = normalize_features(x_input)
normed_x_test = normalize_features(x_test)
normed_y_train = normalize_features(y_output)
normed_y_test = normalize_features(y_test)

print(normed_x_train.shape)

X = tf.placeholder(dtype = tf.float32, shape=[None, 4])
Y = tf.placeholder(dtype = tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_uniform([4, 1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#가설 설정 - 우리가 모은 데이터에 Weight값 곱하고 b값 더해서 학습
hypothesis = tf.matmul(X, W) + b

#설정한 가설에서 실제값을 뺀 것을 최소화 하는 것이 목표
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#정확도
acc = tf.equal(tf.round(hypothesis), Y)
acc = tf.reduce_mean(tf.cast(acc, tf.float32))

#학습률은 0.01로 했으나 큰 의미는 없음 (데이터가 별로 없어서?)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.3)

train = optimizer.minimize(cost)

#텐서 열고 변수 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#run!
for step in range(50000):
    result = sess.run([cost, hypothesis, train], feed_dict={X:normed_x_train, Y:normed_y_train})
    if step % 1000 == 0:
        print(step, sess.run(cost, feed_dict={X:normed_x_train, Y:normed_y_train}))


test_accuracy = sess.run([cost, acc], feed_dict={X: normed_x_test, Y: normed_y_test})
print("test accuracy :", test_accuracy)

model_test = tf.cast(normed_x_test, tf.float32)

a = sess.run(tf.matmul(model_test[8:9], W) + b)
print(a)

def de_normaliztion_of_output(a):
    data_max = np.max(y_output, axis=0)
    data_min = np.min(y_output, axis=0)
    return a*(data_max - data_min) + data_min

print(de_normaliztion_of_output(a))