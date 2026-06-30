from flask import Flask, request, jsonify
from datetime import datetime
import joblib
import numpy as np
from flask_cors import CORS

app = Flask (__name__)
CORS(app)

@app.route("/")
def home():
    return"Hello"

@app.route("/housepriceprediction", methods=["POST"])
def housepriceprediction():
    model = joblib.load("models/saved_house_price (2).pkl")
    #request: bedrooms, yearBuilt, yearRemodAdd, fullBath, garageArea, grLivArea, lotArea
    #response; House Price
    #postAPI
    data = request.get_json()
    print(data)

    bedrooms = data.get("bedrooms")
    yearBuilt = data.get("yearBuilt")
    yearRemodAdd = data.get("yearRemodAdd")
    fullBath = data.get("fullBath")
    garageArea = data.get("garageArea")
    grLivArea = data.get("grLivArea")
    
    

    new_house = [[bedrooms, yearBuilt, yearRemodAdd, fullBath, garageArea, grLivArea]]
    print(new_house)
    house_price = model.predict(new_house)[0]
    print(house_price)

    return jsonify({"house_price": house_price})



@app.route("/loanApprovalPrediction", methods=["POST"])
def loanApprovalPrediction():
    model_loanprediction = joblib.load("model_loanprediction/Loan_Approval.joblib")
    
    #request : income, creditScore, loanAmount, employmentStatus
    #response : loanApprovalStatus
    #postAPI
    data = request.get_json()
    print(data)
    income = data.get("income")
    creditScore = data.get("creditScore")
    loanAmount = data.get("loanAmount")
    employmentStatus = data.get("employmentStatus")

    loan_Approval = [[income, creditScore, loanAmount, employmentStatus]]
    approval_Status = model_loanprediction.predict(loan_Approval)[0]

    if approval_Status == 1:
        approval_Status = "Approved"
    else:
        approval_Status = "Not Approved"

    return jsonify({"loan_approval_status": approval_Status})    

@app.route("/irisflowerprediction", methods=["POST"])
def irisflowerprediction():
    model_iris = joblib.load("model_irisprediction/Flower_prediction_Knn (1).joblib")
    #request : sepalLength, sepalWidth, petalLength, petalWidth
    #response : irisSpecies
    #postAPI
    data = request.get_json()

    sepalLength = data.get("sepalLength")
    sepalWidth = data.get("sepalWidth")
    petalLength = data.get("petalLength")
    petalWidth = data.get("petalWidth")

    iris_features = [[sepalLength, sepalWidth, petalLength, petalWidth]]
    species = model_iris.predict(iris_features)[0]

    return jsonify({"iris_species": species})

@app.route("/mobilepriceprediction", methods=["POST"])
def mobilepriceprediction():
    model_mobilepriceprediction = joblib.load("model_mobilepriceprediction/Mobile_price_prediction (2).pkl")
    #request : battery_power, dual_sim, int_memory, px_height, px_width, touch_screen, wifi
    #response : mobilePriceCategory
    #postAPI
    data = request.get_json()

    battery_power = data.get("battery_power")
    dual_sim = data.get("dual_sim")
    int_memory = data.get("int_memory")
    px_height = data.get("px_height")
    px_width = data.get("px_width")
    touch_screen = data.get("touch_screen")
    wifi = data.get("wifi")

    mobile_features = [[battery_power, dual_sim, int_memory, px_height, px_width, touch_screen, wifi]]
    price_category = model_mobilepriceprediction.predict(mobile_features)[0]

    if price_category == 1:
        price_category = "Below 10000"
    else:
        price_category = "Above 10000"

    return jsonify({"mobile_price_category": price_category})

if __name__ == "__main__":
    app.run(debug="True")


