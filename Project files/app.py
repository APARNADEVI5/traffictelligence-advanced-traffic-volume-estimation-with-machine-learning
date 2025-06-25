import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scale = pickle.load(open('scale.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template("index.html")  # Show the input form

# Predict route
@app.route('/predict', methods=["POST"])
def predict():
    # 1. Get input from form in same order as during training
    input_feature = [float(x) for x in request.form.values()]
    
    # 2. Define feature names in training order
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 
             'year', 'month', 'day', 'hours', 'minutes', 'seconds']
    
    # 3. Create DataFrame from input
    data = pd.DataFrame([input_feature], columns=names)
    
    # 4. Apply scaling
    scaled_data = scale.transform(data)
    
    # 5. Predict using model
    prediction = model.predict(scaled_data)
    
    # 6. Render result page
    return render_template("result.html", prediction=round(prediction[0], 2))

# Run the Flask app
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True, use_reloader=False)
