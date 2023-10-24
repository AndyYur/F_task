import flask
from flask import render_template
import pickle
import sklearn
from sklearn.multioutput import MultiOutputRegressor

##-----------загружаем из файла обученную модель -----------
model_load = pickle.load(open('model.pkl', 'rb'))
##-----------в этом блоке вычисляем min и max значения необходимые для последующей нормировки 
##-----------с использованием датасета на котором проходило обучение модели  ------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("ebw_data.csv") # загрузка датасета на котором обучалась модель
X = df.drop(["Width", "Depth"], axis=1) # убираем из датасета целевые параметры
X_train, X_test = train_test_split(X,train_size=0.7,test_size=0.3,random_state=0,shuffle=True)
scaler = MinMaxScaler()
X_train = scaler.fit(X_train)
##-------------------------------------------------------------


app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST','GET'])

@app.route('/index', methods = ['POST','GET'])
def main():
    message = ""
    if flask.request.method == "GET":
        return render_template('main.html')

    if flask.request.method == "POST":
        
        IW = flask.request.form.get("IW")
        IF = flask.request.form.get("IF")
        VW = flask.request.form.get("VW")
        FP =flask. request.form.get("FP")

        in_parametrs = [[float(IW), float(IF), float(VW), float(FP)]]
        
        #----------- нормируем параметры перед подачей на вход модели --------
        in_parametrs = scaler.transform(in_parametrs)
               
        out_parametrs = model_load.predict(in_parametrs)

        message = f"глубина шва {out_parametrs[0][0]} \n ширина шва {out_parametrs[0][1]}"
    
        return render_template('main.html', message=message)

if __name__ == '__main__':
    app.run()
