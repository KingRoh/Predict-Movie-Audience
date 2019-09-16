import tensorflow as tf
import numpy as np
import pandas as pd

def normalize_features(dataset) :
    data_min = np.min(dataset, axis=0)
    data_max = np.max(dataset, axis=0)
    return (dataset - data_min) / (data_max - data_min)

data = pd.read_excel('C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\2_day.xlsx')
data = data.dropna()
NoneScaled_x_data = data.drop("최종관객수", axis=1)
NoneScaled_x_data = NoneScaled_x_data.drop("2주차관객수", axis=1)
NoneScaled_x_data = NoneScaled_x_data.drop("1주차관객수", axis=1)


df = normalize_features(data)

xy = np.array(df, dtype=np.float32)

x_data = xy[:, 0:4]

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([3]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b


# 저장된 모델을 불러오는 객체를 선언합니다.
saver = tf.train.Saver()
model = tf.global_variables_initializer()
print(x_data)

data_min = np.min(NoneScaled_x_data, axis=0)
data_max = np.max(NoneScaled_x_data, axis=0)
print(data_min)
print(data_max)
dataset = ((17241600, 60178500, 2072, 7503))
inputdata = (dataset - data_min) / (data_max - data_min)

print(inputdata)
# 4가지 변수를 입력 받습니다.
'''sales = float(input('매출액: '))
salesAc = float(input('누적매출액: '))
people = float(input('관객수: '))
peopleAc = float(input('누적관객수: '))'''

with tf.Session() as sess:
    sess.run(model)
    save_path = "./saved_02.cpkt"
    saver.restore(sess, save_path)

    inputdata = (inputdata, (0, 0, 0, 0))
    arr = np.array(inputdata, dtype=np.float32)

    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data})
    print(dict)