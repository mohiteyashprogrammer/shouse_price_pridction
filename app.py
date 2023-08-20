from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData
import os
import sys
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        csv_data = pd.read_csv(os.path.join("artifcats","raw.csv"))
        return render_template("form.html",Brand = sorted(csv_data["Brand"].unique()),
            Model = sorted(csv_data["Model"].unique()),
            Color = sorted(csv_data["Color"].unique()),
            Material = sorted(csv_data["Material"].unique()))

    else:
        data = CustomData(
            Brand = request.form.get("Brand"), 
            Model = request.form.get("Model"),
            Type = int(request.form.get("Type")),
            Gender = int(request.form.get("Gender")),
            Size = float(request.form.get("Size")),
            Color = request.form.get("Color"),
            Material = request.form.get("Material")
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)

        result = round(pred[0])

        return render_template("form.html",final_result = "Your Shoes Price IS : {}".format(result))

if __name__=='__main__':
    app.run(host="0.0.0.0",debug = True)