# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import math
from keras.backend import clear_session
import urllib.request
import json
import re

app = Flask(__name__)
## 실무에 사용할 데이터 준비하기
df00 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\0_day.xlsx", header=0)
df01 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\1_day.xlsx", header=0)
df02 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\2_day.xlsx", header=0)
df03 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\3_day.xlsx", header=0)
df04 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\4_day.xlsx", header=0)
df05 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\5_day.xlsx", header=0)
df06 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\6_day.xlsx", header=0)
df07 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\7_day.xlsx", header=0)
df08 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\8_day.xlsx", header=0)
df09 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\9_day.xlsx", header=0)
df10 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\10_day.xlsx", header=0)
df11 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\11_day.xlsx", header=0)
df12 = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\12_day.xlsx", header=0)
dfAfter = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\day0-14\\after_day.xlsx", header=0)

## 당일의 일일 박스오피스 데이터
dfToday = pd.read_excel("C:/Users\\RohGun\\Desktop\\2019project\\ExcelFiles\\190527movie.xlsx", header=0)

## 개봉일차별 데이터
datasets = [df00.values, df01.values, df02.values, df03.values, df04.values, df05.values, df06.values,
            df07.values, df08.values, df09.values, df10.values, df11.values, df12.values, dfAfter.values]

## X, Y데이터
X = [datasets[0][:, 0:4], datasets[1][:, 0:4], datasets[2][:, 0:4], datasets[3][:, 0:4], datasets[4][:, 0:4],
     datasets[5][:, 0:4], datasets[6][:, 0:4], datasets[7][:, 0:4], datasets[8][:, 0:4], datasets[9][:, 0:4],
     datasets[10][:, 0:4], datasets[11][:, 0:4], datasets[12][:, 0:4], datasets[13][:, 0:4]]

Y = [datasets[0][:, 4:], datasets[1][:, 4:], datasets[2][:, 4:], datasets[3][:, 4:], datasets[4][:, 4:],
     datasets[5][:, 4:], datasets[6][:, 4:], datasets[7][:, 4:], datasets[8][:, 4:], datasets[9][:, 4:],
     datasets[10][:, 4:], datasets[11][:, 4:], datasets[12][:, 4:], datasets[13][:, 4:]]

## 스케일링
minmax_scaler = MinMaxScaler()
scaled_X = [minmax_scaler.fit(X[0]), minmax_scaler.fit(X[1]), minmax_scaler.fit(X[2]), minmax_scaler.fit(X[3]),
            minmax_scaler.fit(X[4]), minmax_scaler.fit(X[5]), minmax_scaler.fit(X[6]), minmax_scaler.fit(X[7]),
            minmax_scaler.fit(X[8]), minmax_scaler.fit(X[9]), minmax_scaler.fit(X[10]), minmax_scaler.fit(X[11]),
            minmax_scaler.fit(X[12]), minmax_scaler.fit(X[13])]

## 최대값 최소값
data_min = [np.min(X[0], axis=0), np.min(X[1], axis=0), np.min(X[2], axis=0), np.min(X[3], axis=0),
            np.min(X[4], axis=0), np.min(X[5], axis=0), np.min(X[6], axis=0), np.min(X[7], axis=0),
            np.min(X[8], axis=0), np.min(X[9], axis=0), np.min(X[10], axis=0), np.min(X[11], axis=0),
            np.min(X[12], axis=0), np.min(X[13], axis=0)]

data_max = [np.max(X[0], axis=0), np.max(X[1], axis=0), np.max(X[2], axis=0), np.max(X[3], axis=0),
            np.max(X[4], axis=0), np.max(X[5], axis=0), np.max(X[6], axis=0), np.max(X[7], axis=0),
            np.max(X[8], axis=0), np.max(X[9], axis=0), np.max(X[10], axis=0), np.max(X[11], axis=0),
            np.max(X[12], axis=0), np.max(X[13], axis=0)]

## 모델 불러오기
clear_session()

## 모델 로딩
model = [load_model("model/model_00"), load_model("model/model_01"), load_model("model/model_02"),
         load_model("model/model_03"), load_model("model/model_04"), load_model("model/model_05"),
         load_model("model/model_06"), load_model("model/model_07"), load_model("model/model_08"),
         load_model("model/model_09"), load_model("model/model_10"), load_model("model/model_11"),
         load_model("model/model_12"), load_model("model/model_after")]

## 플라스크 서버 내에서 구동을 위한 그래프 초기화
graph = tf.get_default_graph()


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        ## 네이버 영화 검색
        movieTitle = request.form['movieTitle']
        client_id = "2F20Z6YIPdWkrFVixfxB"
        client_secret = "RbA6BI0lmt"

        url = "https://openapi.naver.com/v1/search/movie.json"
        option = "&display=20&sort=count"
        query = "?query=" + urllib.parse.quote(movieTitle)
        url_query = url + query + option

        req = urllib.request.Request(url_query)
        req.add_header("X-Naver-Client-Id", client_id)
        req.add_header("X-Naver-Client-Secret", client_secret)

        response = urllib.request.urlopen(req)
        rescode = response.getcode()
        if (rescode == 200):
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))
            items = result.get('items')

        sales = 0; accumulatedSales = 0; audience = 0; accumulatedAudience = 0
        nowDate = datetime.now()
        ## 일일 영화데이터 내에서 검색 후 변수 받아오기
        for i in range(0, len(dfToday.index)):
            if (dfToday["영화명"][i] == movieTitle):
                days = nowDate - (dfToday["개봉일"][i])
                sales = dfToday["매출액"][i]
                accumulatedSales = dfToday["누적매출액"][i]
                audience = dfToday["관객수"][i]
                accumulatedAudience = dfToday["누적관객수"][i]
                break

        if (audience == 0):
            return render_template('index.html', len=len(items), items=items,
                                   message1="해당 영화가 현재 상영중이지 않거나,", message2="없는 영화입니다.")    # 이 함수로 html에 변수를 보냄
        else:
            day = days.days - 1   # 개봉일차
            if (math.isnan(day) or day > 50):
                return render_template('index.html', len=len(items), items=items, message1="해당 영화는 재개봉중인 영화입니다.")
            else:
                global graph
                with graph.as_default():
                    movieData = (sales, accumulatedSales, audience, accumulatedAudience)
                    if (day < 13):
                        inputdata = (movieData - data_min[day]) / (data_max[day] - data_min[day])
                    else:
                        inputdata = (movieData - data_min[13]) / (data_max[13] - data_min[13])

                    inputdata = (inputdata, (0, 0, 0, 0))
                    arr = np.array(inputdata, dtype=np.float32)

                    ## 모델 사용하기
                    if (day < 13):
                        outputdata = model[day].predict(arr)
                    else:
                        outputdata = model[13].predict(arr)

                    UBD = 0
                    if (day < 6 and day >= 0):
                        UBD = round(outputdata[0][2] / 172212, 1)
                        return render_template('index.html', len=len(items), items=items, outputdata1=int(outputdata[0][0]),
                                               outputdata2=int(outputdata[0][1]),
                                               outputdata3=int(outputdata[0][2]),
                                               UBD=UBD)

                    elif (day >= 6 and day < 13):
                        UBD = round(outputdata[0][1] / 172212, 1)
                        return render_template('index.html', len=len(items), items=items, outputdata2=int(outputdata[0][0]),
                                               outputdata3=int(outputdata[0][1]),
                                               UBD=UBD)
                    else:
                        a = outputdata[0][0]
                        b = movieData[3]
                        if (a <= b) | (b < a / 1.5):
                            result = b + (b / 20)
                            outputdata[0][0] = result
                        UBD = round(outputdata[0][0] / 172212, 1)
                        return render_template('index.html', len=len(items), items=items, outputdata3=int(outputdata[0][0]),
                                               UBD=UBD)

if __name__ == '__main__':
    app.run(debug=True)