#%%
## Regression Model to predict
import pandas as pd
import numpy as np
from  sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%%
# Load the dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
model1_data = pd.read_csv(file_path)

