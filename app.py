from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    form_data = request.form
    try:
        no_of_dependents = int(form_data['no_of_dependents'])
        education = 1 if form_data['education'] == 'Graduate' else 0
        self_employed = 1 if form_data['self_employed'] == 'Yes' else 0
        income_annum = float(form_data['income_annum'])
        loan_amount = float(form_data['loan_amount'])
        loan_term = int(form_data['loan_term'])
        cibil_score = int(form_data['cibil_score'])
        residential_assets_value = float(form_data['residential_assets_value'])
        commercial_assets_value = float(form_data['commercial_assets_value'])
        luxury_assets_value = float(form_data['luxury_assets_value'])
        bank_asset_value = float(form_data['bank_asset_value'])
    except ValueError:
        return render_template('result.html', result="Invalid input. Please enter correct values.")

    # Prepare data for prediction
    features = np.array([[no_of_dependents, education, self_employed, income_annum,
                          loan_amount, loan_term, cibil_score, residential_assets_value,
                          commercial_assets_value, luxury_assets_value, bank_asset_value]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    result = "Rejected" if prediction[0] == 1 else "Approved"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
