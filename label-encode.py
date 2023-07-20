
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#reading the dataset:
data = pd.read_csv('bank.csv', sep =";")

#converting the categorical columns into numeric labels by using label encoding:
label_encoder = LabelEncoder()
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])


#features (X) and target (y):
X = data.drop('y', axis=1)
y = data['y']


#linear regression:
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X, y)
linear_reg_pred = linear_reg.predict(X)
linear_reg_rmse = np.sqrt(mean_squared_error(y, linear_reg_pred))
print("RMSE of Linear Regression:", linear_reg_rmse)


#logistic regression:
logistic_reg = linear_model.LogisticRegression()
logistic_reg.fit(X, y)
logistic_reg_pred = logistic_reg.predict(X)
logistic_reg_accuracy = accuracy_score(y, logistic_reg_pred)
print("Accuracy of Logistic Regression:", logistic_reg_accuracy)


