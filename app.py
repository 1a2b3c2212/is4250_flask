import pickle
import os
from flask import Flask
from flask import request
import numpy as np
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
nb = pickle.load(open('diabetes_model.p', 'rb'))

@app.after_request # blueprint can also be app~~
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/')
@cross_origin()
def welcome():
    return "Welcomeee"

@app.route('/predict')
@cross_origin()
def hello():
    def getAgeCat(age):
        age = float(age)
        if age < 25:
            return 1.0
        elif age < 35:
            return 2.0
        elif age < 45:
            return 3.0
        elif age < 55:
            return 4.0
        elif age < 65:
            return 5.0
        else:
            return 6.0
    def getHeight(height):
        height = float(height)
        return 100*height
    def getWeight(weight):
        weight = float(weight)
        return 100*weight
        
    def getBMICat(height,weight):
        bmi = 100*weight/(height*height)
        if bmi < 18.50:
            return 1.0
        elif bmi < 25.00:
            return 2.0
        elif bmi < 30.00:
            return 3.0
        else:
            return 4.0

    a = request.args.get('age')
    h = request.args.get('height')
    w = request.args.get('weight')

    age_cat = getAgeCat(a)
    ht = getHeight(h)
    wt = getWeight(w)
    bmi_cat = getBMICat(ht,wt)
    # print(age_cat,ht,wt,bmi_cat)
    result = nb.predict_proba([[age_cat,ht,wt,bmi_cat]])
    # print(result)
    return str(result[0][1])

port = int(os.environ.get('PORT', 33507))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
# app.run()