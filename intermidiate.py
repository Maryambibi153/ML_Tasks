import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable


df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y


print("Missing values in each column:\n", df.isnull().sum())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro') 
print(f'Cross-Validation F1 Scores: {cv_scores}')
print(f'Mean F1 Score: {np.mean(cv_scores):.2f}')

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')