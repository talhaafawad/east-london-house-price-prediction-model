# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Plotly for interactive visualization
import plotly.graph_objects as go

# Data Collection
df = pd.read_csv('/content/housing_in_london_monthly_variables.csv')
df_1 = pd.read_csv('/content/housing_in_london_yearly_variables.csv')

display(df.head())
display(df_1.head())

# Data Preprocessing

# Set float data type format for better readability of numerical data
pd.options.display.float_format = '{:,.2f}'.format

# Set the maximum number of rows to be displayed in outputs
pd.options.display.max_rows = 999

# Convert 'date' column to datetime format and set it as the index for both datasets
df = df.set_index(pd.to_datetime(df['date']))
df_1 = df_1.set_index(pd.to_datetime(df_1['date']))

# Drop unnecessary columns
df = df.drop(['date'], axis=1)
df_1 = df_1.drop(['date'], axis=1)

print("Data Preprocessing completed.")

# Data Exploration
# Display the first few rows and the summary statistics
display(df.head())
display(df_1.head())
display(df.describe())
display(df_1.describe())

# Feature Engineering
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day

# Select features and target
features = ['year', 'month', 'day']  # Example features, you can add more relevant features
target = 'average_price'

# Train-Test Split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Prediction
y_pred_linear = linear_model.predict(X_test)

# Evaluation
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)

print(f'Linear Regression - MAE: {mae_linear}, MSE: {mse_linear}, RMSE: {rmse_linear}')

# Decision Tree
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Prediction
y_pred_tree = tree_model.predict(X_test)

# Evaluation
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)

print(f'Decision Tree - MAE: {mae_tree}, MSE: {mse_tree}, RMSE: {rmse_tree}')

# Random Forest
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

# Prediction
y_pred_forest = forest_model.predict(X_test)

# Evaluation
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)
rmse_forest = np.sqrt(mse_forest)

print(f'Random Forest - MAE: {mae_forest}, MSE: {mse_forest}, RMSE: {rmse_forest}')

# Visualization of Predictions
fig = go.Figure()

# Add traces for each model
fig.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual Prices'))
fig.add_trace(go.Scatter(x=X_test.index, y=y_pred_linear, mode='lines', name='Linear Regression'))
fig.add_trace(go.Scatter(x=X_test.index, y=y_pred_tree, mode='lines', name='Decision Tree'))
fig.add_trace(go.Scatter(x=X_test.index, y=y_pred_forest, mode='lines', name='Random Forest'))

fig.update_layout(
    title='House Price Predictions',
    xaxis_title='Date',
    yaxis_title='Price (£)',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    legend=dict(y=-.2, orientation='h')
)

fig.show()

# Main Variable Analysis
# Define East London boroughs
east_london_boroughs = [
    'Barking and Dagenham', 'Hackney', 'Havering',
    'Newham', 'Redbridge', 'Tower Hamlets', 'Waltham Forest'
]

# Filter the dataset for East London boroughs
east_london_prices = df[df['area'].isin([borough.lower() for borough in east_london_boroughs])]

# Calculate the mean prices for East London
east_london_mean_price = east_london_prices.groupby('date')['average_price'].mean()

# Plot the data
fig = go.Figure()

fig.add_trace(go.Scatter(x=east_london_mean_price.index,
                         y=east_london_mean_price.values,
                         mode='lines',
                         name='East London Mean House Price',
                        ))

fig.update_layout(
    template='gridon',
    title='Average Monthly House Price in East London',
    xaxis_title='Year',
    yaxis_title='Price (£)',
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    legend=dict(y=-.2, orientation='h'),
    shapes=[
        dict(
            type="line",
            x0='2016-06-01',
            x1='2016-06-01',
            y0=0,
            y1=east_london_mean_price.values.max()*1.2,
            line=dict(
            color="LightSalmon",
            dash="dashdot"
            )
        ),
        dict(
            type="rect",
            x0="2007-12-01",
            y0=0,
            x1="2009-06-01",
            y1=east_london_mean_price.values.max()*1.2,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        ),
        dict(
            type="rect",
            x0="2001-03-01",
            y0=0,
            x1="2001-11-01",
            y1=east_london_mean_price.values.max()*1.2,
            fillcolor="LightSalmon",
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    ],
    annotations=[
            dict(text="The Great Recession", x='2007-12-01', y=east_london_mean_price.values.max()*1.2),
            dict(text="Brexit Vote", x='2016-06-01', y=east_london_mean_price.values.max()*1.2),
            dict(text="Dot-Com Bubble Recession", x='2001-03-01', y=east_london_mean_price.values.max()*1.2)
    ]
)

fig.show()