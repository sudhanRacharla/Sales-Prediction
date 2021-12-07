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

reg = LinearRegression()
reg.fit(x_train,y_train)

pred = reg.predict(x_test)

reg.coef_
reg.intercept_

z = r2_score(y_test, pred)
print(z)
print("Enter the amount you will invest on:")
print("on TV:")
TV = float(input(""))
print("on SocialMedia:")
SocialMedia = float(input(""))
print("on Newspaper:")
Newspaper = float(input(""))

output  = reg.predict([[TV, SocialMedia, Newspaper]])

print("{:.2f} Cr".format(output[0][0] if output else "not predictable"))

#pickle.dump(reg, open('model.pkl','wb'))
#model = pickle.load(open('model.pkl','rb'))