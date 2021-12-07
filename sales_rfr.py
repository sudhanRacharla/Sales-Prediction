import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import warnings

dataset = pd.read_csv("advertising.csv")

x = dataset.drop(['Sales'], axis=1)
y = dataset['Sales'].values.reshape(-1,1)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print("Enter the amount you will invest on:")
print("on TV:")
TV = float(input(""))
print("on SocialMedia:")
SocialMedia = float(input(""))
print("on Newspaper:")
Newspaper = float(input(""))

output  = regressor.predict([[TV, SocialMedia, Newspaper]])

print(output," Cr")

pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))