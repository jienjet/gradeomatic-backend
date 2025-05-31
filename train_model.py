# train_model.py

# 📦 Import libraries needed for data handling, machine learning, and saving models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
import joblib
import os

# 📁 Make sure the folder for saving trained models exists (create if not)
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the dataset containing student performance data
df = pd.read_csv('csv/StudentsPerformance.csv')

# ✅ Define a "pass/fail" criterion for each subject
# Assuming a passing grade is 75 or higher
def determine_pass_fail(score):
    return 'Pass' if score >= 75 else 'Fail'

# Apply the pass/fail criterion to each subject
df['Math_Result'] = df['math score'].apply(determine_pass_fail)
df['Reading_Result'] = df['reading score'].apply(determine_pass_fail)
df['Writing_Result'] = df['writing score'].apply(determine_pass_fail)

# 🔤 Encode the pass/fail results as binary values (Pass = 1, Fail = 0)
result_encoder = LabelEncoder()
df['Math_Result'] = result_encoder.fit_transform(df['Math_Result'])
df['Reading_Result'] = result_encoder.fit_transform(df['Reading_Result'])
df['Writing_Result'] = result_encoder.fit_transform(df['Writing_Result'])

# Save the encoder for future use
joblib.dump(result_encoder, 'trained_data/result_encoder.pkl')

# 🔢 Prepare input features by selecting relevant columns
# Drop the original scores and results to avoid data leakage for classification
X_raw = df.drop(columns=[
    'math score', 'reading score', 'writing score',
    'Math_Result', 'Reading_Result', 'Writing_Result'
])

# 🧠 Convert all categorical variables into numerical format using one-hot encoding
X = pd.get_dummies(X_raw)

# Save the list of features for future use
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

# 🎯 Define the targets for classification and regression
y_math_class = df['Math_Result']
y_reading_class = df['Reading_Result']
y_writing_class = df['Writing_Result']

y_math_reg = df['math score']
y_reading_reg = df['reading score']
y_writing_reg = df['writing score']

# 📊 Split the data into training and test sets for each target
X_train, X_test, y_train_math_class, y_test_math_class = train_test_split(X, y_math_class, test_size=0.2, random_state=42)
_, _, y_train_reading_class, y_test_reading_class = train_test_split(X, y_reading_class, test_size=0.2, random_state=42)
_, _, y_train_writing_class, y_test_writing_class = train_test_split(X, y_writing_class, test_size=0.2, random_state=42)

X_train_reg, X_test_reg, y_train_math_reg, y_test_math_reg = train_test_split(X, y_math_reg, test_size=0.2, random_state=42)
_, _, y_train_reading_reg, y_test_reading_reg = train_test_split(X, y_reading_reg, test_size=0.2, random_state=42)
_, _, y_train_writing_reg, y_test_writing_reg = train_test_split(X, y_writing_reg, test_size=0.2, random_state=42)

# 🛠 Create pipelines for classification and regression
def create_classification_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),  # Normalize features for better performance
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64,),   # One hidden layer with 64 neurons
            activation='relu',          # Activation function for hidden layer
            max_iter=2000,              # Maximum training cycles
            early_stopping=True,        # Stop early if model is not improving
            random_state=42             # Fix seed for reproducibility
        ))
    ])

def create_regression_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),  # Normalize features for better performance
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(64,),   # One hidden layer with 64 neurons
            activation='relu',          # Activation function for hidden layer
            max_iter=2000,              # Maximum training cycles
            early_stopping=True,        # Stop early if model is not improving
            random_state=42             # Fix seed for reproducibility
        ))
    ])

# 🧠 Train classification models
math_class_pipeline = create_classification_pipeline()
math_class_pipeline.fit(X_train, y_train_math_class)
joblib.dump(math_class_pipeline, 'trained_data/model_math_class.pkl')

reading_class_pipeline = create_classification_pipeline()
reading_class_pipeline.fit(X_train, y_train_reading_class)
joblib.dump(reading_class_pipeline, 'trained_data/model_reading_class.pkl')

writing_class_pipeline = create_classification_pipeline()
writing_class_pipeline.fit(X_train, y_train_writing_class)
joblib.dump(writing_class_pipeline, 'trained_data/model_writing_class.pkl')

# 🧠 Train regression models
math_reg_pipeline = create_regression_pipeline()
math_reg_pipeline.fit(X_train_reg, y_train_math_reg)
joblib.dump(math_reg_pipeline, 'trained_data/model_math_reg.pkl')

reading_reg_pipeline = create_regression_pipeline()
reading_reg_pipeline.fit(X_train_reg, y_train_reading_reg)
joblib.dump(reading_reg_pipeline, 'trained_data/model_reading_reg.pkl')

writing_reg_pipeline = create_regression_pipeline()
writing_reg_pipeline.fit(X_train_reg, y_train_writing_reg)
joblib.dump(writing_reg_pipeline, 'trained_data/model_writing_reg.pkl')

# ✅ Print final message to indicate all models are ready
print("✅ Training complete. All models saved to 'trained_data/'")
