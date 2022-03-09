import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv('coin_Bitcoin.csv')
df.drop(columns=['SNo', 'Name', 'Symbol', 'High', 'Low', 'Open', 'Date'], inplace=True)

rawData = df[df['Volume'] != 0.0]

X = rawData[['Close', 'Volume']].to_numpy()

y = np.asarray(rawData['Marketcap']).reshape(-1, 1)


cv = KFold(n_splits=10, random_state=42, shuffle=True)

validation_errors = []

for train_index, val_index in cv.split(y):

  X_train, X_val = X[train_index], X[val_index]
  y_train, y_val = y[train_index], y[val_index]

  print(len(X_val))

  lin_regr = LinearRegression()

  lin_regr.fit(X_train, y_train)

  y_pred_val = lin_regr.predict(X_val)
  val_error = mean_squared_error(y_pred_val, y_val)
  
  validation_errors.append(val_error)


print(validation_errors)