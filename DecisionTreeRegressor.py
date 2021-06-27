import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('sales_train.csv')
df_shops = pd.read_csv('shops.csv')
df_items = pd.read_csv('items.csv')
df_item_categories = pd.read_csv('item_categories.csv')
df_test = pd.read_csv('test.csv')
df_train.isnull().sum()
df_test.isna().sum()
df_train.drop(['date_block_num','item_price'], axis=1, inplace=True)
df_train['date'] = pd.to_datetime(df_train['date'], dayfirst=True)
df_train['date'] = df_train['date'].apply(lambda x: x.strftime('%Y-%m'))
df_train.head()
df = df_train.groupby(['date','shop_id','item_id']).sum()
df = df.pivot_table(index=['shop_id','item_id'], columns='date', values='item_cnt_day', fill_value=0)
df.reset_index(inplace=True)
df_test = pd.merge(df_test, df, on=['shop_id','item_id'], how='left')
df_test.drop(['ID', '2013-01'], axis=1, inplace=True)
print(df)
df_test = df_test.fillna(0)
Y_train = df['2015-10'].values
X_train = df.drop(['2015-10'], axis = 1)
X_test = df_test

print(X_train.shape, Y_train.shape)
print(X_test.shape)
x_train, x_test, y_train, y_test = train_test_split( X_train, Y_train, test_size=0.20, random_state=1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
DTreeRegressor = DecisionTreeRegressor()
DTreeRegressor.fit(x_train, y_train)
print('___________Decision Tree Regression_____________')
DTreePrediction = DTreeRegressor.predict(X_test)
# approximate the result and put it in a list
DTreePrediction = list(map(round, DTreePrediction))
df_submission = pd.read_csv('sample_submission.csv')
df_submission['item_cnt_month'] = DTreePrediction
df_submission.to_csv('DTreePrediction.csv', index=False)

#make the no of rows less to get the accuracy between the y test and the y predict
DTreePrediction = DTreePrediction[0:84825]
print ('Accuracy : ', accuracy_score(y_test,DTreePrediction))



