#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#create dataset
np.random.seed(42)
rows = 200
employee_id = np.arange(1, rows + 1)
age = np.random.randint(21, 60, rows)
gender = np.random.choice(['Male', 'Female'], rows)
education = np.random.choice(['UG', 'PG', 'PhD'], rows)
department = np.random.choice(['HR', 'IT', 'Sales', 'Finance', 'Marketing'], rows)
experience = np.random.randint(1, 20, rows)
working_hours = np.random.randint(35, 60, rows)
performance_score = np.random.randint(1, 10, rows)
projects_completed = np.random.randint(1, 15, rows)
salary = (
    experience * 4000 +
    performance_score * 3000 +
    projects_completed * 1500 +
    working_hours * 500 +
    np.random.randint(5000, 15000, rows)
)

df = pd.DataFrame({
    'Employee_ID': employee_id,
    'Age': age,
    'Gender': gender,
    'Education': education,
    'Department': department,
    'Experience': experience,
    'Working_Hours': working_hours,
    'Performance_Score': performance_score,
    'Projects_Completed': projects_completed,
    'Salary': salary
})

print(df.head())

df.to_csv("employee_salary_dataset.csv", index=False)

print("Dataset saved successfully")

#load dataset and preprocessing
df = pd.read_csv("employee_salary_dataset.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

print("Duplicates Removed")

#encode categorical variable
label_encoder = LabelEncoder()

df['Gender'] = label_encoder.fit_transform(df['Gender'])

df['Education'] = label_encoder.fit_transform(df['Education'])

df['Department'] = label_encoder.fit_transform(df['Department'])

print(df.head())


def salary_category(salary):

    if salary < 70000:
        return "Low"

    elif salary < 120000:
        return "Medium"

    else:
        return "High"

df['Salary_Category'] = df['Salary'].apply(salary_category)

print(df[['Salary', 'Salary_Category']].head())

df['Salary_Category'] = label_encoder.fit_transform(df['Salary_Category'])

print(df.head())
#spitting dataset for regression
X = df.drop(['Salary', 'Salary_Category'], axis=1)

y = df['Salary']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

#linear regression
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

print(linear_predictions[:5])

mae = mean_absolute_error(y_test, linear_predictions)

mse = mean_squared_error(y_test, linear_predictions)

r2 = r2_score(y_test, linear_predictions)

print("Linear Regression Results")

print("MAE :", mae)

print("MSE :", mse)

print("R2 Score :", r2)
#random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

rf_regressor.fit(X_train, y_train)

rf_predictions = rf_regressor.predict(X_test)

print(rf_predictions[:5])

rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_mse = mean_squared_error(y_test, rf_predictions)

rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest Results")

print("MAE :", rf_mae)

print("MSE :", rf_mse)

print("R2 Score :", rf_r2)

#splitting dataset for classification
X_class = df.drop(['Salary', 'Salary_Category'], axis=1)
y_class = df['Salary_Category']

X_class_scaled = scaler.fit_transform(X_class)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class_scaled,
    y_class,
    test_size=0.2,
    random_state=42
)
#logistic regression
logistic_model = LogisticRegression()

logistic_model.fit(Xc_train, yc_train)

logistic_predictions = logistic_model.predict(Xc_test)

print(logistic_predictions[:5])

logistic_accuracy = accuracy_score(yc_test, logistic_predictions)

print("Logistic Regression Accuracy :", logistic_accuracy)

cm = confusion_matrix(yc_test, logistic_predictions)

display = ConfusionMatrixDisplay(confusion_matrix=cm)

display.plot()

plt.show()

#Random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(Xc_train, yc_train)

rf_class_predictions = rf_classifier.predict(Xc_test)

rf_accuracy = accuracy_score(yc_test, rf_class_predictions)

print("Random Forest Classification Accuracy :", rf_accuracy)

plt.figure(figsize=(8,5))

sns.histplot(df['Salary'], bins=20)

plt.title("Salary Distribution")

plt.show()

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.title("Correlation Heatmap")

plt.show()
plt.figure(figsize=(8,5))

plt.scatter(df['Experience'], df['Salary'])

plt.xlabel("Experience")

plt.ylabel("Salary")

plt.title("Experience vs Salary")

plt.show()
plt.figure(figsize=(8,5))

sns.barplot(x='Department', y='Salary', data=df)

plt.title("Department Wise Salary")

plt.show()
importance = rf_regressor.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Feature Importance")

plt.show()
#compare models
print("Model Comparison")

print()

print("Linear Regression R2 Score :", r2)

print("Random Forest R2 Score :", rf_r2)

print("Logistic Regression Accuracy :", logistic_accuracy)

print("Random Forest Classification Accuracy :", rf_accuracy)