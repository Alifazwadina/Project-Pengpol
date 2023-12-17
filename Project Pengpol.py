import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv('polusi.csv')

conditions = [
    ((dataset['pm25'] >= 0) & (dataset['pm25'] <= 30)) & ((dataset['pm10'] >= 0) & (dataset['pm10'] < 50)) & ((dataset['co'] >= 0) & (dataset['co'] <= 50)), 
    ((dataset['pm25'] > 30) & (dataset['pm25'] <= 60)) | ((dataset['pm10'] >= 50) & (dataset['pm10'] < 80)) | ((dataset['co'] > 50) & (dataset['co'] <= 100)),
    ((dataset['pm25'] > 60) & (dataset['pm25'] <= 150)) | ((dataset['pm10'] >= 80) & (dataset['pm10'] < 150)) | ((dataset['co'] > 100) & (dataset['co'] <= 199))
]
choices = ['bersih', 'sedang', 'buruk']
dataset['air_pollution'] = np.select(conditions, choices, default='unknown')

dataset = dataset.dropna()

X = dataset[['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']]  
y = dataset['air_pollution']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_train_scaled, y_train)

y_pred = decision_tree_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

samples = X_test.sample(10)

samples['air_pollution'] = decision_tree_model.predict(scaler.transform(samples))

print("\nSample data and predictions:")
print(samples)