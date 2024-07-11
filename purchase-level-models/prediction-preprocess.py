# %%
# Preprocessing
import pandas as pd
import numpy as np

# Paths to the datasets
customers_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv'
sales_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales21-24.csv'

# Load the datasets
customers = pd.read_csv(customers_path, dtype={'ID': str})
sales = pd.read_csv(sales_path, dtype={'Customer ID': str})

# Merge the datasets
merged_data = pd.merge(sales, customers, left_on='Customer ID', right_on='ID', how='left')

# Convert 'Date' to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.sort_values(by=['Customer ID', 'Date'], inplace=True)

# Fill empty emails with 'No Email'
merged_data['Email'].fillna('No Email', inplace=True)

# Drop rows where Qty. Sold or Amount is zero or negative
merged_data = merged_data[(merged_data['Qty. Sold'] > 0) & (merged_data['Amount'] > 0)]

# Create a target variable for purchase next month
merged_data['Next Purchase Date'] = merged_data.groupby('Customer ID')['Date'].shift(-1)
merged_data['Purchase Next Month'] = ((merged_data['Next Purchase Date'] - merged_data['Date']).dt.days <= 30).astype(int)

# Select relevant columns
final_data = merged_data[['Customer ID', 'Email', 'Company Name', 'Item', 'Date', 'Qty. Sold', 'Amount', 'Purchase Next Month']]

# Save to CSV
final_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv', index=False)

# Print the final dataframe to verify the changes
print(final_data.head())


# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load the prepared dataset
data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv')

# Ensure date is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Define features and target variable
features = ['Qty. Sold', 'Amount']
target = 'Purchase Next Month'

# Use data from January 2021 to May 2024 for training
train_data = data[(data['Date'] >= '2021-01-01') & (data['Date'] < '2024-06-01')]

X_train = train_data[features]
y_train = train_data[target]

# Use data from June 2024 for testing
test_data = data[(data['Date'] >= '2024-06-01') & (data['Date'] < '2024-07-01')]
X_test = test_data[features]
y_test = test_data[target]

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # 'liblinear' supports l1 and l2 penalties
}

# Initialize the logistic regression model
logreg = LogisticRegression()

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='f1')

# Fit the model to the resampled training data
grid_search.fit(X_train_resampled_scaled, y_train_resampled)

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best f1 score: ", grid_search.best_score_)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_

# Evaluate the best model on the training set
y_train_pred = best_model.predict(X_train_resampled_scaled)
print("Training Classification Report:")
print(classification_report(y_train_resampled, y_train_pred))

# Evaluate the best model on the test set
y_test_pred = best_model.predict(X_test_scaled)
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))


# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_resampled_scaled, y_train_resampled)

# Evaluate the model on the training set
y_train_pred = model.predict(X_train_resampled_scaled)
print("Training Classification Report:")
print(classification_report(y_train_resampled, y_train_pred))

# Evaluate the model on the test set
y_test_pred = model.predict(X_test_scaled)
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))

# %%
# Predict purchases for June 2024
test_data['Predicted Purchase Next Month'] = best_model.predict(X_test_scaled)

# Output the necessary information
output_data = test_data[['Customer ID', 'Email', 'Company Name', 'Predicted Purchase Next Month']]

# Drop duplicates based on Customer ID, keeping the first occurrence
output_data = output_data.drop_duplicates(subset=['Customer ID'], keep='first')

# Save the output data to a CSV file
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/output_month_6_2024.csv'
output_data.to_csv(output_path, index=False)

# Print the output data
print(output_data)

# Count the number of predictions where 'Predicted Purchase Next Month' == 1
num_predictions_1 = output_data['Predicted Purchase Next Month'].sum()
print(f'Number of predictions for purchases in June 2024: {num_predictions_1}')

# Check the data for customer ID 100011
customer_data = data[data['Customer ID'] == '100011']
print(customer_data)

# Predict for customer ID 100011 if present in the test data
customer_latest_data = test_data[test_data['Customer ID'] == '100011']
if not customer_latest_data.empty:
    X_customer_latest = customer_latest_data[features]
    X_customer_latest_scaled = scaler.transform(X_customer_latest)
    
    # Predict for this specific customer
    customer_prediction = best_model.predict(X_customer_latest_scaled)
    print(f'Prediction for customer ID 100011: {customer_prediction}')
else:
    print(f'Customer ID 100011 not present in the test data.')


# %%
