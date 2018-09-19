import numpy as np
import pandas as pd
import sklearn as sc
from catboost import CatBoostRegressor, Pool
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

train_df = pd.read_csv('encoded_train.csv')
Y = train_df.SalePrice
X = train_df.drop(['SalePrice', 'Unnamed: 0'], axis=1)
# find categorial features
categorial_features = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

# ignored_features = [0,4,5,8,9,13,15,20,21,22,23,24,27,34,41,44,70,76]
encoded_categorial_features = []
for i in range(len(X.columns)):
    if X.columns[i] in categorial_features:
        # print(f'{i} = {X.columns[i]}')
        encoded_categorial_features.append(i)

X_train, X_validation, y_train, y_validation = train_test_split(X, Y, train_size=0.8, random_state=1234, shuffle=True)

model = CatBoostRegressor(
    iterations=2500,
    learning_rate=0.01,
    random_seed=63,
    # rsm=0.5,
    early_stopping_rounds=40,
    one_hot_max_size=5,
    l2_leaf_reg=5,
    random_strength=3,
    leaf_estimation_method = 'Newton'
)
model.fit(
    X_train, y_train,
    cat_features = encoded_categorial_features,
    eval_set=(X_validation,y_validation),
    metric_period=10,
)
print(f'Model params: {model.get_params()}')

# print('Start training best model...')
# best_model = CatBoostRegressor(
#     random_seed=63,
#     iterations=1200,
#     learning_rate=0.01,
#     # l2_leaf_reg=3,
#     # bagging_temperature=1,
#     # random_strength=1,
#     early_stopping_rounds=40,
# )
#
# best_model.fit(
#     X,Y,
#     cat_features=encoded_categorial_features,
#     metric_period=10,
# )
# print(f'BEST Model params: {best_model.get_params()}')
# print(f'Best model score = {best_model.score(X,Y)}')
#
# predict test data
test_df = pd.read_csv('encoded_test.csv')
test_x = test_df.drop(['Unnamed: 0'], axis=1)
predictions = model.predict(data=test_x)

print('Start writing...')
f = open('submissions/catboost_last_try_for_today.csv', 'w+')
f.write('Id,SalePrice\n')
counter = 1461
for y in predictions:
    f.write(f'{counter},{int(y)}\n')
    counter += 1
print('Finished writing')
