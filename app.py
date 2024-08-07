import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def get_model_name():
    from_date = request.args.get('from')
    to_date = request.args.get('to')
    # from_date = "2002-01-01"
    # to_date = "2003-01-01"


    df = pd.read_csv('./housing_in_london_monthly_variables.csv')
    df_1 = pd.read_csv('./housing_in_london_yearly_variables.csv')

    pd.options.display.float_format = '{:,.2f}'.format
    pd.options.display.max_rows = 999

    df['date'] = pd.to_datetime(df['date'])
    df_1['date'] = pd.to_datetime(df_1['date'])

    df.set_index('date', inplace=True)
    df_1.set_index('date', inplace=True)

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day


    features = ['year', 'month', 'day', ]  # Example features, you can add more relevant features
    target = 'average_price'

    X = df[features]
    y = df[target]



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dates = y_train.index.strftime('%Y-%d-%m').tolist()
    train_data = [{"date": date, "predictedPrice": price} for date, price in zip(train_dates, y_train.tolist())]

    unique_dates = {}

    for date, price in zip(train_dates, y_train.tolist()):
        unique_dates[date] = price

    train_data = [{"date": date, "predictedPrice": price} for date, price in unique_dates.items()]
    train_data = sorted(train_data, key=lambda x: pd.to_datetime(x['date']))

    if from_date and to_date:
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        train_data = [item for item in train_data if from_date <= pd.to_datetime(item['date']) <= to_date]

    print("train_data ", train_data)

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    y_pred_linear = linear_model.predict(X_test)
    mae_linear = mean_absolute_error(y_test, y_pred_linear)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    rmse_linear = np.sqrt(mse_linear)

    print(f'Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, RMSE: {rmse_linear}')


    max_depth_str = request.args.get('max_depth')
    
    try:
        max_depth = int(max_depth_str) if max_depth_str is not None else 42
    except (TypeError, ValueError):
        max_depth = 42
    
    tree_model = DecisionTreeRegressor(random_state=max_depth)    
    tree_model.fit(X_train, y_train)

    y_pred_tree = tree_model.predict(X_test)
    mae_tree = mean_absolute_error(y_test, y_pred_tree)
    mse_tree = mean_squared_error(y_test, y_pred_tree)
    rmse_tree = np.sqrt(mse_tree)

    print(f'Decision Tree - MAE: {mae_tree}, MSE: {mse_tree}, RMSE: {rmse_tree}')



    # Random Forest
    n_estimators_str = request.args.get('n_estimators')
    
    try:
        n_estimators = int(n_estimators_str) if n_estimators_str is not None else 100
    except (TypeError, ValueError):
        n_estimators = 100
    
    max_depth_str = request.args.get('max_depth')
    
    try:
        max_depth = int(max_depth_str) if max_depth_str is not None else 42
    except (TypeError, ValueError):
        max_depth = 42
    
    forest_model = RandomForestRegressor(n_estimators=n_estimators, random_state=max_depth)
    forest_model.fit(X_train, y_train)

    y_pred_forest = forest_model.predict(X_test)
    mae_forest = mean_absolute_error(y_test, y_pred_forest)
    mse_forest = mean_squared_error(y_test, y_pred_forest)
    rmse_forest = np.sqrt(mse_forest)

    print(f'Random Forest - MAE: {mae_forest}, MSE: {mse_forest}, RMSE: {rmse_forest}')

    

    data = {
        "prices":train_data,
        "linearRegressionModel": {
            "mae_linear": mae_linear,
            "mse_linear": mse_linear,
            "rmse_linear": rmse_linear
        },
        "decisiontree": {
            "mae_tree": mae_tree,
            "mse_tree": mse_tree,
            "rmse_tree": rmse_tree,
        },
        "randomForest": {
            "mae_forest": mae_forest,
            "mse_forest": mse_forest,
            "rmse_forest": rmse_forest   
        }
    }

    return jsonify(data)
    

if __name__ == '__main__':
    app.run()



# switch to current folder where this app.py file is present
# install python and pip
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt (for verfiication , type pip list)


# how to start
# python3 app.py