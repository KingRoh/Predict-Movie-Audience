import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def normalize_features(dataset) :
    data_min = np.min(dataset, axis=0)
    data_max = np.max(dataset, axis=0)
    return (dataset - data_min) / (data_max - data_min)

model = tf.global_variables_initializer()

data = pd.read_excel('C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\2_day.xlsx')
data = data.dropna()

df = normalize_features(data)
xy = np.array(df, dtype=np.float32)
xy2 = np.array(data, dtype=np.float32)

# 4개의 변인을 입력을 받습니다.
x_data = xy[:, 0:4]

# 가격 값을 입력 받습니다.
y_data = xy2[:, 4:]

X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([3]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 비용 함수를 설정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 함수를 설정합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(cost)

# 세션을 생성합니다.
sess = tf.Session()

# 글로벌 변수를 초기화합니다.
sess.run(tf.global_variables_initializer())

# 학습을 수행합니다.
for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("#", step, " 손실 비용: ", cost_)
        print("- 최종관객수: ", hypo_)

# 학습된 모델을 저장합니다.
saver = tf.train.Saver()
save_path = saver.save(sess, "./saved_02.cpkt")
print('학습된 모델을 저장했습니다.')