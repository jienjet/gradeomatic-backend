from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the new trained models and metadata
try:
    gpa_model = joblib.load('trained_data/gpa_model.pkl')
    grade_class_model = joblib.load('trained_data/grade_class_model.pkl')
    feature_columns = joblib.load('trained_data/feature_columns.pkl')
    model_info = joblib.load('trained_data/model_info.pkl')
    print("✅ Models loaded successfully!")
    print(f"📊 Features: {len(feature_columns)}")
    print(f"🎯 Features used: {feature_columns}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# Grade class to letter grade mapping
GRADE_CLASS_MAPPING = {
    0: "A",   # Excellent (3.5-4.0 GPA)
    1: "B",   # Good (2.5-3.49 GPA)
    2: "C",   # Average (1.5-2.49 GPA)
    3: "D",   # Below Average (0.5-1.49 GPA)
    4: "F"    # Failing (0.0-0.49 GPA)
}

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Gradeometer API!",
        "version": "2.0 - Student Performance Prediction",
        "endpoints": {
            "/predict": "POST - Predict student GPA and Grade Class",
            "/model-info": "GET - Get information about the trained models"
        }
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Return information about the loaded models"""
    return jsonify({
        "model_type": "Student Performance Prediction",
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "outputs": {
            "predicted_gpa": "Continuous value (0.0 - 4.0)",
            "predicted_grade_class": "Classification (0-4)",
            "predicted_letter_grade": "Letter grade (A, B, C, D, F)"
        },
        "training_info": {
            "training_samples": model_info.get('training_samples', 'Unknown'),
            "test_samples": model_info.get('test_samples', 'Unknown'),
            "gpa_model_r2": model_info.get('gpa_test_r2', 'Unknown'),
            "grade_class_accuracy": model_info.get('grade_class_test_accuracy', 'Unknown')
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        print(f"📥 Received data: {data}")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure we have all required features
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {list(missing_features)}",
                "required_features": feature_columns
            }), 400
        
        # Select only the features used in training (in correct order)
        input_features = input_df[feature_columns]
        
        print(f"🔧 Input features shape: {input_features.shape}")
        print(f"🔧 Input features: {input_features.iloc[0].to_dict()}")
        
        # Make predictions
        predicted_gpa = gpa_model.predict(input_features)[0]
        predicted_grade_class = grade_class_model.predict(input_features)[0]
        
        # Convert grade class to letter grade
        predicted_letter_grade = GRADE_CLASS_MAPPING.get(int(predicted_grade_class), "Unknown")
        
        # Get prediction probabilities for grade class (if available)
        try:
            grade_class_proba = grade_class_model.predict_proba(input_features)[0]
            class_probabilities = {
                GRADE_CLASS_MAPPING[i]: float(prob) 
                for i, prob in enumerate(grade_class_proba)
            }
        except:
            class_probabilities = None
        
        # Prepare response
        response = {
            "success": True,
            "predictions": {
                "predicted_gpa": round(float(predicted_gpa), 3),
                "predicted_grade_class": int(predicted_grade_class),
                "predicted_letter_grade": predicted_letter_grade
            },
            "input_summary": {
                "age": data.get('Age', 'N/A'),
                "gender": data.get('Gender', 'N/A'),
                "study_time_weekly": data.get('StudyTimeWeekly', 'N/A'),
                "absences": data.get('Absences', 'N/A'),
                "parental_support": data.get('ParentalSupport', 'N/A')
            }
        }
        
        # Add class probabilities if available
        if class_probabilities:
            response["class_probabilities"] = class_probabilities
        
        print(f"📤 Response: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "An error occurred while making predictions"
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Handle batch predictions for multiple students"""
    try:
        data = request.get_json()
        
        if not data or 'students' not in data:
            return jsonify({"error": "No student data provided. Expected format: {'students': [...]}"}, 400)
        
        students = data['students']
        predictions = []
        
        for i, student_data in enumerate(students):
            try:
                # Convert to DataFrame
                input_df = pd.DataFrame([student_data])
                
                # Check for missing features
                missing_features = set(feature_columns) - set(input_df.columns)
                if missing_features:
                    predictions.append({
                        "student_index": i,
                        "error": f"Missing features: {list(missing_features)}"
                    })
                    continue
                
                # Select features and predict
                input_features = input_df[feature_columns]
                predicted_gpa = gpa_model.predict(input_features)[0]
                predicted_grade_class = grade_class_model.predict(input_features)[0]
                predicted_letter_grade = GRADE_CLASS_MAPPING.get(int(predicted_grade_class), "Unknown")
                
                predictions.append({
                    "student_index": i,
                    "predicted_gpa": round(float(predicted_gpa), 3),
                    "predicted_grade_class": int(predicted_grade_class),
                    "predicted_letter_grade": predicted_letter_grade,
                    "student_id": student_data.get('StudentID', f'Student_{i}')
                })
                
            except Exception as e:
                predictions.append({
                    "student_index": i,
                    "error": str(e)
                })
        
        return jsonify({
            "success": True,
            "batch_size": len(students),
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Gradeometer API Server...")
    print(f"📋 Loaded features: {feature_columns}")
    print(f"🎯 Ready to predict GPA and Grade Class!")
    app.run(debug=True, host='0.0.0.0', port=5000)