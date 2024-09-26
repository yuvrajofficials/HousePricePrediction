import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
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


@app.route('/predict', methods=['POST'])
def final_price():
    # Collecting the data from the form
    data = [float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveBedrms']),
            float(request.form['Population']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])]

    # Transforming and predicting
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    
    # Display the result
    return render_template("home.html", prediction_text=f"The House price is ${output:.2f}")


if __name__ == "__main__":
    app.run(debug=True)
