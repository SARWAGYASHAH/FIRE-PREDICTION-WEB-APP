from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

appliaction=Flask(__name__) ## web app created 
app=appliaction

# 1st step---> importing pickle
ridge_model=pickle.load(open('ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict_data",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        # Collect form values from the post part
            temp = float(request.form.get('Temperature'))
            rh = float(request.form.get('RH'))
            ws = float(request.form.get('Ws'))
            rain = float(request.form.get('Rain'))
            ffmc = float(request.form.get('FFMC'))
            dmc = float(request.form.get('DMC'))
            isi = float(request.form.get('ISI'))
            classes = int(request.form.get('Classes'))
            region = int(request.form.get('Region'))

            new_scaled_data=Standard_Scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
            result=ridge_model.predict(new_scaled_data)
            return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
# this is getting mapped to your local IP address of the machine
