import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from flask import Flask, render_template, request

# Load dataset
df = pd.read_csv('mosquito_data.csv')

# Define features and target
X = df[[ 'temperature', 'vegetation', 'rainfall']]
y = df['breeding_sites']

# Impute missing values
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Create Flask application
app=Flask(__name__,template_folder='template')

# Define homepage route
@app.route('/')
def home():
    return render_template('home.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    lat = float(request.form['latitude'])
    long = float(request.form['longitude'])
    temp = float(request.form['temperature'])
    veg = float(request.form['vegetation'])
    rain = float(request.form['rainfall'])

    input_data = [[temp, veg, rain]]
    predicted_breeding_sites = rf.predict(input_data)

    return render_template('prediction.html', breeding_sites=round(predicted_breeding_sites[0], 2))

# Run Flask application
if __name__ == '__main__':
    app.run(debug=True)
