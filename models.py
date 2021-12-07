import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv("advertising.csv")
df = pd.DataFrame(dataset)
x = dataset.drop(['Sales'], axis=1)
y = dataset['Sales'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

reg = LinearRegression()
reg.fit(x_train,y_train)
pred = reg.predict(x_test)

z = r2_score(y_test, pred)
print("r2 score using linear regression: ",z)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

print("r2 score using random forest regressor: ",r2_score(y_test, y_pred))

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)

print("r2 score using decision tree regression: ",r2_score(y_test, y_pred))
