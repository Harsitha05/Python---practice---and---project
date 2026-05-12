import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load the dataset
# Adjust this path if your titanic.csv is right inside your project folder to just "titanic.csv"
csv_path = r"C:\Users\ACER\Downloads\titanic.csv"
df = pd.read_csv(csv_path)

print("--- Initial Data Snapshot ---")
print(df.head())

# 2. Basic Data Cleaning & Preprocessing
# Drop duplicates
df = df.drop_duplicates()

# Handle missing values for key features
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Map categorical 'Sex' column to numbers (0 for female, 1 for male)
# Lowercase just in case your CSV data varies
df['Sex'] = df['Sex'].str.lower().map({'female': 0, 'male': 1})

# Define the features (X) and the target variable (y)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

# 3. Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
print("\nTraining the Random Forest Model...")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate the Model Performance
y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Save the trained model to a file
model_filename = "titanic_model.pkl"
joblib.dump(model, model_filename)
print(f"\nSuccess! Model saved safely as '{model_filename}'")