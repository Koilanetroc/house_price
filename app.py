import numpy as np
import pandas as pd
import sklearn as sc
import math
from sklearn import preprocessing,linear_model,ensemble
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier

train_data = pd.read_csv('encoded_train.csv')
Y = train_data['SalePrice']
X = train_data.drop(columns=['SalePrice', 'Unnamed: 0'])
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X))

# model = sc.neural_network.MLPRegressor(activation="identity",solver='lbfgs',learning_rate='adaptive')
# model = linear_model.Ridge(alpha=.1)
model = ensemble.RandomForestClassifier()
# model = linear_model.LinearRegression()
model.fit(X,Y)
train_predictions = model.predict(X)
errors = sc.metrics.mean_squared_error(Y,train_predictions)
print(f'Cost function = {errors}')

test_data = pd.read_csv('encoded_test.csv')
test_X = test_data.drop(columns=['Unnamed: 0'])
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(test_X)
test_X = pd.DataFrame(imp.transform(test_X))
print(test_X)
predictions = model.predict(test_X)

print('Start writing...')
f = open('random_forest_submission.csv', 'w+')
f.write('Id,SalePrice\n')
counter = 1461
for y in predictions:
    f.write(f'{counter},{int(y)}\n')
    counter += 1
print('Finished writing')
