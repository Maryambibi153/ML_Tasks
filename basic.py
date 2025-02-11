import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = load_breast_cancer()
X = data.data  
y = data.target  


X = X[:, 0].reshape(-1, 1)  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


plt.figure(figsize=(15, 8))
plt.scatter(X_test, y_test, color='green', label='Actual Data')  
plt.plot(X_test, y_pred, color='yellow', label='Regression Line') 
plt.xlabel('Mean Radius')
plt.ylabel('Tumor Classification (0: Malignant, 1: Benign)')
plt.title('Linear Regression on Breast Cancer Dataset')
plt.legend()
plt.show()


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}')