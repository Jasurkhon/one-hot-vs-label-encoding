import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, accuracy_score

#loading the dataset:

df = pd.read_csv('bank.csv', sep =";")

#converting the target variable to 1s and 0s
df['y'] = df['y'].map({'no': 0, 'yes': 1})

#separate the target variable and the feature variables
X = df.drop('y', axis=1)  
y = df['y']  

#one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

#splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions on the testing set
y_pred = model.predict(X_test)

#RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")
