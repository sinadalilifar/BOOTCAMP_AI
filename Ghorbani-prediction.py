# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:00:40 2024

@author: Ghorbani Hamidreza
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
file_path = 'C:/Users/NP/Desktop/AI Boot camp/example1(1).xlsx'
data = pd.read_excel(file_path)

# Convert 'Station' to numerical
data = pd.get_dummies(data, columns=['Station'], drop_first=True)

# Drop the 'date' column as it won't be used for prediction
data = data.drop(columns=['date'])

# Handle missing values (if any)
data = data.dropna()

# Split the data into features and target
X = data.drop(columns=['PM 10 ug/m3'])
y = data['PM 10 ug/m3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train and evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, rmse, mae, r2

# Initialize models
rf_model = RandomForestRegressor(random_state=42)
svm_model = SVR()
lr_model = LinearRegression()

# Evaluate models
y_pred_rf, rmse_rf, mae_rf, r2_rf = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
y_pred_svm, rmse_svm, mae_svm, r2_svm = evaluate_model(svm_model, X_train, X_test, y_train, y_test)
y_pred_lr, rmse_lr, mae_lr, r2_lr = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

# Print evaluation metrics
print("Random Forest:")
print(f"RMSE: {rmse_rf}")
print(f"MAE: {mae_rf}")
print(f"R^2: {r2_rf}\n")

print("Support Vector Machine:")
print(f"RMSE: {rmse_svm}")
print(f"MAE: {mae_svm}")
print(f"R^2: {r2_svm}\n")

print("Linear Regression:")
print(f"RMSE: {rmse_lr}")
print(f"MAE: {mae_lr}")
print(f"R^2: {r2_lr}\n")

# Plot predictions
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest')

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_svm, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('SVM')

plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')

plt.tight_layout()
plt.show()
