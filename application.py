from flask import Flask, render_template, request, jsonify 
import  pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

lasso_model = pickle.load(open('models/lasso.pkl' , 'rb'))
stdndard_scaler = pickle.load(open('models/scaler.pkl' , 'rb'))


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST' :
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Region = int(request.form.get('Region'))
        Classes = int(request.form.get('Classes'))

        output_scaled = stdndard_scaler.transform([[Temperature , RH , Ws , Rain ,FFMC,DMC,ISI , Classes , Region]])
        result = lasso_model.predict(output_scaled)

        return render_template("home.html" , results = result[0])

    else:
        return render_template("home.html" )


if __name__ == '__main__':
    app.run(host = "0.0.0.0")