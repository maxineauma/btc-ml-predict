import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# models:
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = []
models.append(LinearRegression())
models.append(DecisionTreeRegressor())
models.append(RandomForestRegressor())

# Defining the dataset, preprocessing 
btc_data = pd.read_csv("data/BTC-USD.csv")
X = btc_data[["Open", "High", "Low", "Adj Close", "Volume"]] 
y = btc_data["Close"]

# Splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#Running each algorithm model
for m in models:
    clf = m 
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    res = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

    print("-"*80)
    print("Model: " + str(m))
    print(y_pred)
    print("r2: " + str(r2))
    print("MER: " + str(mean_absolute_percentage_error(y_test, y_pred)))
    print("-"*80)
