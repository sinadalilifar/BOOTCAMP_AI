# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:15:44 2024

@author: Ghorbani Hamidreza
"""

import pandas as pd
import numpy as np
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

# Preprocessing
data = pd.get_dummies(data, columns=['Station'], drop_first=True)
data = data.drop(columns=['date'])
data = data.dropna()

# Feature and target split
X = data.drop(columns=['NOx ppb'])
y = data['NOx ppb']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model definitions
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVM': SVR(),
    'Linear Regression': LinearRegression()
}

# Evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_pred, rmse, mae, r2

# Evaluate all models
results = {}
for name, model in models.items():
    y_pred, rmse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {
        'y_pred': y_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Print evaluation metrics
for name, metrics in results.items():
    print(f"{name}:")
    print(f"RMSE: {metrics['rmse']}")
    print(f"MAE: {metrics['mae']}")
    print(f"R^2: {metrics['r2']}\n")

# Plot predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, metrics) in zip(axes, results.items()):
    ax.scatter(y_test, metrics['y_pred'], alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title(name)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

plt.tight_layout()
plt.show()
