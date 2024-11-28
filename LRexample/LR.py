from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

cars = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
pd.set_option('display.max_columns', None)
le = LabelEncoder()
for col in cars.columns:
    cars[col] = le.fit_transform((cars[col]))

print(cars.head())

X = cars.drop(columns=['selling_price'])
Y = cars['selling_price']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
pred = linreg.predict(X_test)

mse = mean_squared_error(Y_test,pred) ** .5
print("Mean squared error: ", mse)