# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:52:40 2024

@author: All
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_excel(r'D:/1.Bootcamp AI/جلسه 6/BootCamp-AI-Day-6-Codes/example1.xlsx')

date = df["date"]


new_date = np.zeros_like(date, dtype=np.uint8)
for i in range(date.shape[0]):
    item = date[i]
    new_date[i] = int(str(item).split(" ")[0].split("-")[-1])
    
df = df.drop(["date",], axis=1)
df["date"] = new_date
stations = df["Station"]
stations = stations.to_numpy()

for i, item in enumerate(np.unique(stations)):
    stations = np.where(stations == item, i+1, stations)
    
df["Station"] = stations
df["Station"] = df["Station"].astype(np.uint8)

df1=df[df['Station']==1]

targets = df1["CO ppm"]
inputs = df1.drop(["CO ppm"], axis=1)

inputs = inputs.to_numpy()
targets = targets.to_numpy()
inputs.shape, targets.shape


X_train, X_test, y_train, y_test = train_test_split(inputs, targets, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, y_pred)}")

# plot --------------------

array_ytrain = list(range(len(y_train)))

y_p = np.concatenate((y_train, y_pred))
array_ypred = list(range(len(y_train)+len(y_pred)))
    
    
plt.figure(1)

plt.plot(y_p, array_ypred, color='orange' , label='CO ppm Prediction')
plt.plot(y_train , array_ytrain, label='CO ppm')
plt.legend()

plt.show()



