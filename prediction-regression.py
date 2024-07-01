#%%
## Regression Model to predict
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

#%%
# Load data
df = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/FinalSalesData.csv')

# Convert Purchase Date to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

# Create a target variable (Example)
df['Purchase Likelihood'] = (df['Unit Volume'] / df['Total Money Spent']) * 100

# Split data
test_start = datetime(2024, 3, 1)
test_end = datetime(2024, 6, 5)
train = df[(df['Purchase Date'] < test_start)]
test = df[(df['Purchase Date'] >= test_start) & (df['Purchase Date'] <= test_end)]

# Define features and target
X_train = train.drop(['Names', 'Email', 'Purchase Date', 'Purchase Likelihood'], axis=1)
y_train = train['Purchase Likelihood']
X_test = test.drop(['Names', 'Email', 'Purchase Date', 'Purchase Likelihood'], axis=1)
y_test = test['Purchase Likelihood']

# Identify categorical features
categorical_features = ['Product', 'Company Name']

# Create Pool objects for CatBoost
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
test_pool = Pool(X_test, y_test, cat_features=categorical_features)

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=100)

# Train the model
model.fit(train_pool)

# Make predictions
predictions = model.predict(test_pool)

# Evaluate
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)

print(f'RMSE: {rmse}, MAE: {mae}')
