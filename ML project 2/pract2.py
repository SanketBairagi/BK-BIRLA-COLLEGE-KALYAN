import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("pract2_model.sav",'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]
    if output == 1 :
        return render_template("index.html",prediction_text = "Passenger Survived")
    else: 
        return render_template("index.html",prediction_text = "Passenger Not Survived")

if __name__=="__main__":
    app.run(debug=True)