#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm
#%%
# Combine datasets, features manipulation, and one hot encode date
customers_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv'
sales_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SALESDATA.csv'
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_logistic.csv'


customers = pd.read_csv(customers_path, dtype={'ID': str})
sales = pd.read_csv(sales_path, dtype={'ID': str})


merged_data = pd.merge(sales, customers, left_on='ID', right_on='ID', how='left')
merged_data.columns = merged_data.columns.str.replace('_x', '')
merged_data.columns = merged_data.columns.str.replace('_y', '')
merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data['Amount'] = merged_data['Amount'].replace('[\$,]', '', regex=True)

# Convert 'Qty. Sold' and 'Amount' to numeric, forcing errors to NaN
merged_data['Qty. Sold'] = pd.to_numeric(merged_data['Qty. Sold'], errors='coerce')
merged_data['Amount'] = pd.to_numeric(merged_data['Amount'], errors='coerce')

merged_data['Email'].fillna('No Email', inplace=True)

filtered_data = merged_data[(merged_data['Qty. Sold'] > 0) & (merged_data['Amount'] > 0)]
filtered_data = filtered_data[['ID', 'Email', 'Top Level Parent', 'Item', 'Date', 'Qty. Sold', 'Amount']]
#%%
# Rename columns
filtered_data.rename(columns={
    'ID': 'Customer ID',
    'Top Level Parent': 'Organization',
    'Item': 'Product'
}, inplace=True)

filtered_data['YearMonth'] = filtered_data['Date'].dt.strftime('%m_%Y')

# Encode YearMonth as binary (0 and 1)
year_months = pd.date_range(start='2021-01', end='2024-07', freq='M').strftime('%m_%Y')
for ym in year_months:
    filtered_data[f'YM_{ym}'] = (filtered_data['YearMonth'] == ym).astype(int)

filtered_data.drop(columns=['YearMonth'], inplace=True)
filtered_data.to_csv(output_path, index=False)
print(filtered_data.head())
# %%
# Create target varaible
# Load the data
data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_logistic.csv'
data = pd.read_csv(data_path)

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Customer ID', 'Date'])

# Generate target variable for next three months
data['Target'] = 0

# Use a rolling window to check for purchases within the next 3 months
data['Future Purchase'] = data.groupby('Customer ID')['Date'].transform(lambda x: x.rolling('90D').max())
data['Target'] = (data['Future Purchase'] > data['Date']).astype(int)
data.drop(columns=['Future Purchase'], inplace=True)

print("Target variable generation complete.")
#%%
# Prepare features and target variable
X = data.drop(columns=['Customer ID', 'Email', 'Organization', 'Product', 'Date', 'Target'])
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Logistic Regression Accuracy: {accuracy}')
print(f'Logistic Regression Precision: {precision}')
print(f'Logistic Regression Recall: {recall}')
print(f'Classification Report:\n{class_report}')

# Save the results to a CSV file
results_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/logistic_regression_results.csv'
results = pd.DataFrame({'Customer ID': data['Customer ID'], 'Email': data['Email'], 'Organization': data['Organization'],
                        'Product': data['Product'], 'Date': data['Date'], 'Target': data['Target'], 'Prediction': log_reg.predict(X)})
results.to_csv(results_path, index=False)
print(f'Results saved to {results_path}')
