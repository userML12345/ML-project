import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('coin_Bitcoin.csv')
df.drop(columns=['SNo', 'Name', 'Symbol', 'High', 'Low', 'Open', 'Date'], inplace=True)

rawData = df[df['Volume'] != 0.0]

X = rawData[['Close', 'Volume']].to_numpy()

y = np.asarray(rawData['Marketcap']).reshape(-1, 1)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.4, random_state=42)

print(len(X_test))
print(len(X_val))
print(len(X))



lin_regr = LinearRegression()

poly = PolynomialFeatures(degree=1)

X_train_poly = poly.fit_transform(X_train)
lin_regr.fit(X_train_poly, y_train)

y_pred_train = lin_regr.predict(X_train_poly)
tr_error = mean_squared_error(y_train, y_pred_train)

X_val_poly = poly.fit_transform(X_val)
y_pred_val = lin_regr.predict(X_val_poly)
val_error = mean_squared_error(y_val, y_pred_val)

X_test_poly = poly.fit_transform(X_test)
y_pred_test = lin_regr.predict(X_test_poly)
test_error = mean_squared_error(y_test, y_pred_test)


print(tr_error)
print(val_error)
print(test_error)

print(r2_score(y_test, y_pred_test))
