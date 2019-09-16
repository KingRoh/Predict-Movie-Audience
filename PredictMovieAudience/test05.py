# 서버 python 만들기 전 테스트 소스

from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from numpy import argmax
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import math

# 1. 실무에 사용할 데이터 준비하기
df00 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\0_day.xlsx",header=0)
df01 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\1_day.xlsx",header=0)
df02 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\2_day.xlsx",header=0)
df03 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\3_day.xlsx",header=0)
df04 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\4_day.xlsx",header=0)
df05 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\5_day.xlsx",header=0)
df06 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\6_day.xlsx",header=0)
df07 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\7_day.xlsx",header=0)
df08 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\8_day.xlsx",header=0)
df09 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\9_day.xlsx",header=0)
df10 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\10_day.xlsx",header=0)
df11 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\11_day.xlsx",header=0)
df12 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\12_day.xlsx",header=0)
dfAfter = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\after_day.xlsx",header=0)

dfToday = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\190518movie.xlsx",header=0)

datasets = [df00.values, df01.values, df02.values, df03.values, df04.values, df05.values, df06.values, df07.values, df08.values, df09.values, df10.values, df11.values, df12.values, dfAfter.values]

X = [datasets[0][:, 0:4], datasets[1][:, 0:4], datasets[2][:, 0:4], datasets[3][:, 0:4], datasets[4][:, 0:4], datasets[5][:, 0:4], datasets[6][:, 0:4],
     datasets[7][:, 0:4], datasets[8][:, 0:4], datasets[9][:, 0:4], datasets[10][:, 0:4], datasets[11][:, 0:4], datasets[12][:, 0:4], datasets[13][:, 0:4]]

Y = [datasets[0][:, 4:], datasets[1][:, 4:], datasets[2][:, 4:], datasets[3][:, 4:], datasets[4][:, 4:], datasets[5][:, 4:], datasets[6][:, 4:],
     datasets[7][:, 4:], datasets[8][:, 4:], datasets[9][:, 4:], datasets[10][:, 4:], datasets[11][:, 4:], datasets[12][:, 4:], datasets[13][:, 4:]]

minmax_scaler = MinMaxScaler()
scaled_X = [minmax_scaler.fit(X[0]), minmax_scaler.fit(X[1]), minmax_scaler.fit(X[2]), minmax_scaler.fit(X[3]), minmax_scaler.fit(X[4]), minmax_scaler.fit(X[5]), minmax_scaler.fit(X[6]), minmax_scaler.fit(X[7])
            , minmax_scaler.fit(X[8]), minmax_scaler.fit(X[9]), minmax_scaler.fit(X[10]), minmax_scaler.fit(X[11]), minmax_scaler.fit(X[12]), minmax_scaler.fit(X[13])]

data_min = [np.min(X[0], axis=0), np.min(X[1], axis=0), np.min(X[2], axis=0), np.min(X[3], axis=0), np.min(X[4], axis=0), np.min(X[5], axis=0), np.min(X[6], axis=0), np.min(X[7], axis=0)
            , np.min(X[8], axis=0), np.min(X[9], axis=0), np.min(X[10], axis=0), np.min(X[11], axis=0), np.min(X[12], axis=0), np.min(X[13], axis=0)]

data_max = [np.max(X[0], axis=0), np.max(X[1], axis=0), np.max(X[2], axis=0), np.max(X[3], axis=0), np.max(X[4], axis=0), np.max(X[5], axis=0), np.max(X[6], axis=0), np.max(X[7], axis=0)
            , np.max(X[8], axis=0), np.max(X[9], axis=0), np.max(X[10], axis=0), np.max(X[11], axis=0), np.max(X[12], axis=0), np.max(X[13], axis=0)]

# 2. 모델 불러오기
model = [load_model("model_00"), load_model("model_01"), load_model("model_02"), load_model("model_03"), load_model("model_04"), load_model("model_05"), load_model("model_06"), load_model("model_07")
         , load_model("model_08"), load_model("model_09"), load_model("model_10"), load_model("model_11"), load_model("model_12"), load_model("model_after")]

# 3. 검색
movieName = input("영화제목 : ")
nowDate = datetime.now()
sales = 0; accumulatedSales = 0; audience = 0; accumulatedAudience = 0
for i in range(0, len(dfToday.index)):
    if (dfToday["영화명"][i] == movieName):
        days = nowDate - (dfToday["개봉일"][i])
        sales = dfToday["매출액"][i]
        accumulatedSales = dfToday["누적매출액"][i]
        audience = dfToday["관객수"][i]
        accumulatedAudience = dfToday["누적관객수"][i]
        break

if (audience == 0):
    print("해당 영화는 현재 상영중인 영화가 아닙니다.")
else:
    day = days.days - 1
    print(day)
    if (math.isnan(day) or day > 50):
        print("해당 영화는 재개봉한 영화입니다.")
    else:
        movieData = ((sales, accumulatedSales, audience, accumulatedAudience))
        print(movieData)
        if (day < 13):
            inputdata = (movieData - data_min[day]) / (data_max[day] - data_min[day])
        else:
            inputdata = (movieData - data_min[13]) / (data_max[13] - data_min[13])

        inputdata = (inputdata, (0, 0, 0, 0))
        arr = np.array(inputdata, dtype=np.float32)
        print(arr)

        # 3. 모델 사용하기
        if (day < 13):
            outputdata = model[day].predict(arr)
        else:
            outputdata = model[13].predict(arr)

        if (day >= 13):
            a = outputdata[0][0]
            b = movieData[3]
            print(a)
            print(b)
            if (a <= b) | (b < a / 1.5):
                result = b + (b / 10)
                print(result)
                outputdata[0][len(outputdata[0].index)] = result

        print(outputdata[0]) # 개수별로 server에 전달하는 모양이 다 다르다.