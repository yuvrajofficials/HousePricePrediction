import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, render_template, url_for
import pickle

app = Flask(__name__)

# Correctly load the model and scaler
with open("regmodel.pkl", "rb") as model_file:
    regmodel = pickle.load(model_file)

with open("scaling.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data)
    newdata = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(newdata)
    print(output)
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
