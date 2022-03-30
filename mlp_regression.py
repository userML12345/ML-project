import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

df = pd.read_csv('coin_Bitcoin.csv')
df.drop(columns=['SNo', 'Name', 'Symbol', 'High', 'Low', 'Open', 'Date'], inplace=True)

rawData = df[df['Volume'] != 0.0]

X = rawData[['Volume']].to_numpy()

y = rawData['Marketcap'].to_numpy()

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

layers = range(21)

layers = layers[2: -1]

errors = []

for l in layers:
  mlp = MLPRegressor(hidden_layer_sizes=l, activation="identity", solver="lbfgs", batch_size="auto")

  mlp.fit(X_train, y_train)

  y_pred_train = mlp.predict(X_train)
  y_pred_test = mlp.predict(X_test)
  y_pred_val = mlp.predict(X_val)

  test_error = mean_squared_error(y_pred_test, y_test)
  val_error = mean_squared_error(y_pred_val, y_val)
  train_error = mean_squared_error(y_pred_train, y_train)

  val_mean = mean_absolute_error(y_pred_val, y_val)

  print(val_mean)

  _errors = [train_error, test_error, val_error, l]

  errors.append(_errors)

min_val = min(errors)

print(min)
plt.figure(figsize=(5,2))

plt.bar(1, min_val[0], label = 'Train')
plt.bar(2, min_val[2], label = 'Val')
plt.bar(3, min_val[1], label = 'Test')

plt.legend(loc = 'upper left')

plt.xlabel('sets')
plt.ylabel('Loss')
plt.title('Train, validation and test loss')
plt.show()