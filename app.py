from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models and encoders
math_class_model = joblib.load('trained_data/model_math_class.pkl')
reading_class_model = joblib.load('trained_data/model_reading_class.pkl')
writing_class_model = joblib.load('trained_data/model_writing_class.pkl')

math_reg_model = joblib.load('trained_data/model_math_reg.pkl')
reading_reg_model = joblib.load('trained_data/model_reading_reg.pkl')
writing_reg_model = joblib.load('trained_data/model_writing_reg.pkl')

result_encoder = joblib.load('trained_data/result_encoder.pkl')
model_features = joblib.load('trained_data/model_features.pkl')

@app.route('/')
def home():
    return "Welcome to the Gradeometer API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data])

    # Ensure the input data has the same features as the trained model
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # Make predictions for pass/fail
    math_class_prediction = math_class_model.predict(input_df)[0]
    reading_class_prediction = reading_class_model.predict(input_df)[0]
    writing_class_prediction = writing_class_model.predict(input_df)[0]

    # Decode pass/fail predictions
    math_result = result_encoder.inverse_transform([math_class_prediction])[0]
    reading_result = result_encoder.inverse_transform([reading_class_prediction])[0]
    writing_result = result_encoder.inverse_transform([writing_class_prediction])[0]

    # Make predictions for grades
    math_grade_prediction = math_reg_model.predict(input_df)[0]
    reading_grade_prediction = reading_reg_model.predict(input_df)[0]
    writing_grade_prediction = writing_reg_model.predict(input_df)[0]

    # Return predictions as JSON
    return jsonify({
        'Math_Result': math_result,
        'Math_Grade': round(math_grade_prediction, 2),
        'Reading_Result': reading_result,
        'Reading_Grade': round(reading_grade_prediction, 2),
        'Writing_Result': writing_result,
        'Writing_Grade': round(writing_grade_prediction, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)