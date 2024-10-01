import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data = pd.read_csv('train.csv')
print(train_data.head())
print(train_data.columns)

features = [
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'OverallQual',
    'YearBuilt',
    'TotalBsmtSF',
    'GarageCars',
    'GarageArea',
    'LotArea'
]
X = train_data[features]
y = train_data['SalePrice']

X = X.dropna()
y = y[X.index]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_data)
plt.title('Square Footage vs Price')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=train_data)
plt.title('Bedrooms vs Price')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

train_predictions = model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, train_predictions)
train_r2 = r2_score(y_train, train_predictions)
print(f"Training MSE: {train_mse}, Training R^2: {train_r2}")

test_predictions = model.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)
print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")

test_data = pd.read_csv('test.csv')
print(test_data.isnull().sum())

test_data = test_data[features]

for feature in features:
    if test_data[feature].dtype in ['int64', 'float64']:
        test_data[feature] = test_data[feature].fillna(test_data[feature].mean())

test_data_scaled = scaler.transform(test_data)
test_predictions = model.predict(test_data_scaled)

submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = test_predictions
submission.to_csv('submission.csv', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, model.predict(X_test_scaled))
plt.xlabel('Actual Prices (y_test from train.csv)')
plt.ylabel('Predicted Prices (from train.csv)')
plt.title('Actual vs Predicted House Prices (from train.csv)')
plt.show()
