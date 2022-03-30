import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv('coin_Bitcoin.csv')
df.drop(columns=['SNo', 'Name', 'Symbol', 'High', 'Low', 'Open', 'Date'], inplace=True)

rawData = df[df['Volume'] != 0.0]

X = rawData[['Close', 'Volume']].to_numpy()

y = rawData['Marketcap'].to_numpy()

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

lin_regr = LinearRegression()

lin_regr.fit(X_train, y_train)

y_pred_train = lin_regr.predict(X_train)
y_pred_test = lin_regr.predict(X_test)
y_pred_val = lin_regr.predict(X_val)

test_error = mean_squared_error(y_pred_test, y_test)
val_error = mean_squared_error(y_pred_val, y_val)
train_error = mean_squared_error(y_pred_train, y_train)

val_mean = mean_absolute_error(y_pred_val, y_val)

print(val_mean)

print(train_error)
print(val_error)
print(test_error)

plt.figure(figsize=(5,2))

plt.bar(1, train_error, label = 'Train')
plt.bar(2, val_error, label = 'Val')
plt.bar(3, test_error, label = 'Test')

plt.legend(loc = 'upper left')

plt.xlabel('sets')
plt.ylabel('Loss')
plt.title('Train, validation and test loss')
plt.show()
