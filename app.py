from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('pickle_models/model.pkl', 'rb'))
model2 = pickle.load(open('pickle_models/model2.pkl', 'rb'))

anemia_model = pickle.load(open('pickle_models/anemia_model.pkl', 'rb'))
anemia_model2 = pickle.load(open('pickle_models/anemia_model2.pkl', 'rb'))

chd_model = pickle.load(open('pickle_models/chd_model.pkl', 'rb'))
chd_model2 = pickle.load(open('pickle_models/chd_model2.pkl', 'rb'))


def compute_risk_score(estimator, processed_input, prediction):
    """Return risk score (0-100) and level label for UI."""
    try:
        decision = float(estimator.decision_function(processed_input)[0])
        score = int(round(100 / (1 + np.exp(-decision))))
    except Exception:
        score = 75 if int(prediction[0]) == 1 else 28

    score = max(1, min(99, score))

    if score < 35:
        level = 'Low Risk'
    elif score < 65:
        level = 'Moderate Risk'
    else:
        level = 'High Risk'

    return score, level


@app.route('/')
def hello_world():
    return render_template("home.html")


@app.route('/diabetes')
def diabetes():
    return render_template("Diabetes/diabetes.html")


@app.route('/diabetesParameters')
def diabetesParameters():
    return render_template("Diabetes/diabetesParameters.html")


@app.route('/diabetesHelp')
def diabetesHelp():
    return render_template("Diabetes/diabetesHelp.html")


diabetes_param_ranges = {
    'Pregnancies': [0, 15],
    'Glucose': [60, 200],
    'Blood Pressure': [0, 150],
    'Skin Thickness': [0, 100],
    'Insulin': [0, 900],
    'BMI': [0, 70],
    'Diabetes Pedigree Function': [0, 3],
    'Age': [18, 90]
}


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    # Check the validity of each input value
    for param, range_ in diabetes_param_ranges.items():
        value = request.form.get(param)
        try:
            value = float(value)
            if value < range_[0] or value > range_[1]:
                return render_template('Diabetes/diabetes.html', pred=f'{param.title()} value should be between {range_[0]} and {range_[1]}')
        except ValueError:
            return render_template('Diabetes/diabetes.html', pred=f'Invalid {param.title()} value')
    int_features = [float(request.form.get(param))
                    for param in diabetes_param_ranges.keys()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = model2.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    score, risk_level = compute_risk_score(model, std_data, prediction)

    if prediction[0] == 1:
        return render_template('Diabetes/diabetes.html', pred='Person is Diabetic.', score=score, risk_level=risk_level)
    else:
        return render_template('Diabetes/diabetes.html', pred='Person is Non-Diabetic.', score=score, risk_level=risk_level)


@app.route('/anemia')
def anemia():
    return render_template("Anemia/anemia.html")


@app.route('/anemiaParameters')
def anemiaParameters():
    return render_template("Anemia/anemiaParameters.html")


@app.route('/anemiaHelp')
def anemiaHelp():
    return render_template("Anemia/anemiaHelp.html")


anemia_param_ranges = {
    'Gender': [0, 1],
    'Hemoglobin': [5, 20],
    'MCH': [10, 40],
    'MCHC': [15, 50],
    'MCV': [60, 110],
}


@app.route('/predictAnemia', methods=['POST', 'GET'])
def predictAnemia():
    # Check the validity of each input value
    for param, range_ in anemia_param_ranges.items():
        value = request.form.get(param)
        try:
            value = float(value)
            if value < range_[0] or value > range_[1]:
                return render_template('Anemia/anemia.html', pred=f'{param.title()} value should be between {range_[0]} and {range_[1]}')
        except ValueError:
            return render_template('Anemia/anemia.html', pred=f'Invalid {param.title()} value')
    int_features = [float(request.form.get(param))
                    for param in anemia_param_ranges.keys()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = anemia_model2.transform(input_data_reshaped)
    prediction = anemia_model.predict(std_data)
    score, risk_level = compute_risk_score(anemia_model, std_data, prediction)

    if prediction[0] == 1:
        return render_template('Anemia/anemia.html', pred='Person is Anemic.', score=score, risk_level=risk_level)
    else:
        return render_template('Anemia/anemia.html', pred='Person is Not Anemic.', score=score, risk_level=risk_level)


@app.route('/chd')
def chd():
    return render_template("CHD/chd.html")


@app.route('/chdParameters')
def chdParameters():
    return render_template("CHD/chdParameters.html")


@app.route('/chdHelp')
def chdHelp():
    return render_template("CHD/chdHelp.html")


chd_param_ranges = {
    'Male': [0, 1],
    'Age': [20, 90],
    'Education': [1, 4],
    'Current Smoker': [0, 1],
    'Cigarettes Per Day': [0, 50],
    'BP Medicines': [0, 1],
    'Prevalent Stroke': [0, 1],
    'Prevalent Hypertensive': [0, 1],
    'Diabetes': [0, 1],
    'Total Cholesterol': [0, 500],
    'Systolic Blood Pressure': [80, 300],
    'Diastolic Blood Pressure': [45, 200],
    'BMI': [0, 70],
    'Heart Rate': [0, 130],
    'Glucose': [0, 300],
}


@app.route('/predictCHD', methods=['POST', 'GET'])
def predictCHD():
    # Check the validity of each input value
    for param, range_ in chd_param_ranges.items():
        value = request.form.get(param)
        try:
            value = float(value)
            if value < range_[0] or value > range_[1]:
                return render_template('CHD/chd.html', pred=f'{param.title()} value should be between {range_[0]} and {range_[1]}')
        except ValueError:
            return render_template('CHD/chd.html', pred=f'Invalid {param.title()} value')
    int_features = [float(request.form.get(param))
                    for param in chd_param_ranges.keys()]
    final = np.asarray(int_features)
    input_data_reshaped = final.reshape(1, -1)
    std_data = chd_model2.transform(input_data_reshaped)
    prediction = chd_model.predict(std_data)
    score, risk_level = compute_risk_score(chd_model, std_data, prediction)

    if prediction[0] == 1:
        return render_template('CHD/chd.html', pred='Person is having Ten Year CHD Risk.', score=score, risk_level=risk_level)
    else:
        return render_template('CHD/chd.html', pred='Person is not having Ten Year CHD Risk.', score=score, risk_level=risk_level)


if __name__ == '__main__':
    app.run(debug=True)
