# train_model.py - Updated for Student Performance Data with GPA and Grade Class prediction

# 📦 Import libraries needed for data handling, machine learning, and saving models
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

# 📁 Make sure the folder for saving trained models exists (create if not)
os.makedirs("trained_data", exist_ok=True)

# 📥 Load the new dataset containing student performance data
df = pd.read_csv('csv/Student_performance_data _.csv')

# 🔍 Display basic info about the dataset
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# 🧹 Data preprocessing - prepare features (X) and targets (y)
# Select features - all columns except StudentID, GPA, and GradeClass
feature_columns = [col for col in df.columns if col not in ['StudentID', 'GPA', 'GradeClass']]
X = df[feature_columns].copy()

# 🎯 Define targets
y_gpa = df['GPA']  # Regression target
y_grade_class = df['GradeClass']  # Classification target

print(f"\nFeatures used: {feature_columns}")
print(f"GPA range: {y_gpa.min():.2f} to {y_gpa.max():.2f}")
print(f"Grade Classes: {sorted(y_grade_class.unique())}")

# 📊 Split data into training and test sets
X_train, X_test, y_train_gpa, y_test_gpa, y_train_grade, y_test_grade = train_test_split(
    X, y_gpa, y_grade_class, test_size=0.2, random_state=42, stratify=y_grade_class
)

# 💾 Save feature names for future use in predictions
joblib.dump(feature_columns, 'trained_data/feature_columns.pkl')

# 🔧 Create pipeline for GPA regression
def create_gpa_regression_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),  # Normalize features for better performance
        ('regressor', MLPRegressor(
            hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
            activation='relu',              # Activation function for hidden layers
            max_iter=3000,                  # Maximum training cycles
            early_stopping=True,            # Stop early if model is not improving
            validation_fraction=0.1,        # Use 10% of training data for validation
            n_iter_no_change=20,            # Stop if no improvement for 20 iterations
            random_state=42                 # Fix seed for reproducibility
        ))
    ])

# 🔧 Create pipeline for Grade Class classification
def create_grade_class_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),  # Normalize features for better performance
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
            activation='relu',              # Activation function for hidden layers
            max_iter=3000,                  # Maximum training cycles
            early_stopping=True,            # Stop early if model is not improving
            validation_fraction=0.1,        # Use 10% of training data for validation
            n_iter_no_change=20,            # Stop if no improvement for 20 iterations
            random_state=42                 # Fix seed for reproducibility
        ))
    ])

# 🧠 Train GPA regression model
print("\n🧠 Training GPA regression model...")
gpa_pipeline = create_gpa_regression_pipeline()
gpa_pipeline.fit(X_train, y_train_gpa)

# Evaluate GPA model
gpa_train_score = gpa_pipeline.score(X_train, y_train_gpa)
gpa_test_score = gpa_pipeline.score(X_test, y_test_gpa)
print(f"GPA Model - Training R²: {gpa_train_score:.4f}")
print(f"GPA Model - Test R²: {gpa_test_score:.4f}")

# Save GPA model
joblib.dump(gpa_pipeline, 'trained_data/gpa_model.pkl')
print("✅ GPA regression model saved!")

# 🧠 Train Grade Class classification model
print("\n🧠 Training Grade Class classification model...")
grade_class_pipeline = create_grade_class_pipeline()
grade_class_pipeline.fit(X_train, y_train_grade)

# Evaluate Grade Class model
grade_train_score = grade_class_pipeline.score(X_train, y_train_grade)
grade_test_score = grade_class_pipeline.score(X_test, y_test_grade)
print(f"Grade Class Model - Training Accuracy: {grade_train_score:.4f}")
print(f"Grade Class Model - Test Accuracy: {grade_test_score:.4f}")

# Save Grade Class model
joblib.dump(grade_class_pipeline, 'trained_data/grade_class_model.pkl')
print("✅ Grade Class classification model saved!")

# 🔍 Display class distribution
print(f"\nGrade Class distribution in training set:")
print(y_train_grade.value_counts().sort_index())

# 📝 Create a simple test prediction to verify models work
print(f"\n🧪 Testing models with first sample...")
sample_features = X_test.iloc[0:1]  # Take first test sample
predicted_gpa = gpa_pipeline.predict(sample_features)[0]
predicted_class = grade_class_pipeline.predict(sample_features)[0]
actual_gpa = y_test_gpa.iloc[0]
actual_class = y_test_grade.iloc[0]

print(f"Sample prediction:")
print(f"  Predicted GPA: {predicted_gpa:.3f} (Actual: {actual_gpa:.3f})")
print(f"  Predicted Grade Class: {predicted_class} (Actual: {actual_class})")

# 💾 Save model metadata
model_info = {
    'features': feature_columns,
    'gpa_model_path': 'trained_data/gpa_model.pkl',
    'grade_class_model_path': 'trained_data/grade_class_model.pkl',
    'feature_columns_path': 'trained_data/feature_columns.pkl',
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'gpa_test_r2': gpa_test_score,
    'grade_class_test_accuracy': grade_test_score
}

joblib.dump(model_info, 'trained_data/model_info.pkl')

print(f"\n✅ Training complete!")
print(f"📁 All models and metadata saved to 'trained_data/' directory")
print(f"📊 Dataset: {len(df)} samples, {len(feature_columns)} features")
print(f"🎯 Models trained for: GPA prediction (regression) and Grade Class prediction (classification)")
