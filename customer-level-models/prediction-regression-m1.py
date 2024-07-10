#%%
## currently using this for preprocess
import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
data = pd.read_csv(file_path)

# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Split the dataset into records before 2024 and records in 2024
data_before_2024 = data[data['Purchase Date'] < '2024-01-01']
data_2024 = data[(data['Purchase Date'] >= '2024-01-01') & (data['Purchase Date'] < '2025-01-01')]

# Sort the dataset by 'Names' (Customer ID) and 'Purchase Date'
data_before_2024 = data_before_2024.sort_values(by=['Names', 'Purchase Date'])
data_2024 = data_2024.sort_values(by=['Names', 'Purchase Date'])

# Rename 'Names' to 'ID'
data_before_2024.rename(columns={'Names': 'ID'}, inplace=True)
data_2024.rename(columns={'Names': 'ID'}, inplace=True)

# Create a new DataFrame for aggregated features
customer_features = pd.DataFrame()
customer_features['ID'] = data_before_2024['ID'].unique()

# Calculate purchase frequency
purchase_frequency = data_before_2024.groupby('ID').size().reset_index(name='purchase_frequency')

# Calculate total and mean spent
total_spent = data_before_2024.groupby('ID')['Total Money Spent'].sum().reset_index(name='total_spent')
mean_spent = data_before_2024.groupby('ID')['Total Money Spent'].mean().reset_index(name='mean_spent')

# Calculate mean and max purchase period
data_before_2024['previous_purchase_date'] = data_before_2024.groupby('ID')['Purchase Date'].shift(1)
data_before_2024['purchase_period'] = (data_before_2024['Purchase Date'] - data_before_2024['previous_purchase_date']).dt.days
mean_purchase_period = data_before_2024.groupby('ID')['purchase_period'].mean().reset_index(name='mean_purchase_period')
max_purchase_period = data_before_2024.groupby('ID')['purchase_period'].max().reset_index(name='max_purchase_period')

# Calculate max spent per purchase
max_spent = data_before_2024.groupby('ID')['Total Money Spent'].max().reset_index(name='max_spent')

# Get the last purchase date
last_purchase_date = data_before_2024.groupby('ID')['Purchase Date'].max().reset_index(name='last_purchase_date')

# Merge all features into the customer_features DataFrame
customer_features = customer_features.merge(purchase_frequency, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(total_spent, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(mean_spent, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(mean_purchase_period, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(max_purchase_period, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(max_spent, on='ID', how='left', suffixes=('', '_y'))
customer_features = customer_features.merge(last_purchase_date, on='ID', how='left', suffixes=('', '_y'))

# Drop the redundant 'Names_y' columns from the merges
customer_features.drop(columns=[col for col in customer_features.columns if col.endswith('_y')], inplace=True)

# Retain the email feature
email = data_before_2024[['ID', 'Email']].drop_duplicates()
customer_features = customer_features.merge(email, on='ID', how='left')

# One-hot encode the product feature
product_encoded = pd.get_dummies(data_before_2024[['ID', 'Product']], columns=['Product'])
product_encoded = product_encoded.groupby('ID').sum().reset_index()

# Merge one-hot encoded product features into customer_features
customer_features = customer_features.merge(product_encoded, on='ID', how='left')

# One-hot encode the purchase dates by month
data_before_2024['month_year'] = data_before_2024['Purchase Date'].dt.to_period('M')
purchase_dates_encoded = pd.get_dummies(data_before_2024[['ID', 'month_year']], columns=['month_year'])
purchase_dates_encoded = purchase_dates_encoded.groupby('ID').sum().reset_index()

# Merge one-hot encoded purchase dates into customer_features
customer_features = customer_features.merge(purchase_dates_encoded, on='ID', how='left')

# Add target columns for each month in 2024 (January to July)
for month in range(1, 8):
    target_column = f'purchase_in_month_{month}_2024'
    customer_features[target_column] = 0

# Update target columns for January to June based on actual data
for month in range(1, 7):
    purchase_dates_2024 = data_2024[(data_2024['Purchase Date'].dt.month == month)]
    customer_features.loc[customer_features['ID'].isin(purchase_dates_2024['ID']), f'purchase_in_month_{month}_2024'] = 1

# Display the final customer_features DataFrame
print(customer_features.head())

# Save the final dataset
customer_features.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv', index=False)

print("Customer-level dataset created successfully.")

# %%
# Regression Model 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#%%
# Load the customer-level dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv'
customer_data = pd.read_csv(file_path)

# Ensure 'ID' column exists
if 'ID' not in customer_data.columns:
    raise KeyError("The column 'ID' does not exist in the dataset.")

# Convert 'last_purchase_date' to datetime
customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])

# Create 'days_since_last_purchase' feature (number of days since a reference date)
reference_date = pd.to_datetime('2024-01-01')
customer_data['days_since_last_purchase'] = (reference_date - customer_data['last_purchase_date']).dt.days

# Define the features (X) excluding the target columns for each month in 2024
target_columns = [f'purchase_in_month_{month}_2024' for month in range(1, 7)]
X = customer_data.drop(columns=['ID', 'Email', 'last_purchase_date'] + target_columns)

# One-hot encode categorical features if any
X = pd.get_dummies(X, drop_first=True)

# Handle any remaining NaN values in the feature set
X.fillna(0, inplace=True)

# Initialize a list to store prediction dataframes for each month
predictions_list = []
#%%
# Loop through each month to build and evaluate a model
for month in range(1, 7):
    # Define the target variable for the specific month
    target_column = f'purchase_in_month_{month}_2024'
    if target_column not in customer_data.columns:
        continue
    y = customer_data[target_column]
    
    # Check the distribution of the target variable
    print(f"Distribution for month {month}:")
    print(y.value_counts())
    
    # Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Train the Logistic Regression model with increased iterations
    model = LogisticRegression(max_iter=5000)  # Increase max_iter
    model.fit(X_train, y_train)
    
    # Predict on the test set
    predictions = model.predict(X_test).astype(int)  # Convert predictions to integers
    
    # Prepare the final output dataframe with ID and predicted target
    output_df = X_test.copy()
    output_df[f'predicted_purchase_in_month_{month}_2024'] = predictions
    output_df['ID'] = customer_data.loc[X_test.index, 'ID']
    
    # Select relevant columns to display
    final_output = output_df[['ID', f'predicted_purchase_in_month_{month}_2024']]
    
    # Append the prediction dataframe to the list
    predictions_list.append(final_output)
    
    # Print out the classification report
    report = classification_report(y_test, predictions)
    print(f"Classification Report for month {month}:\n", report)
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix for month {month}:\n", conf_matrix)
#%%
# Merge all prediction dataframes on 'ID'
if predictions_list:
    combined_predictions = predictions_list[0]
    for df in predictions_list[1:]:
        combined_predictions = combined_predictions.merge(df, on='ID', how='outer')

    # Ensure prediction columns are integers
    for col in combined_predictions.columns:
        if col.startswith('predicted_purchase_in_month_'):
            combined_predictions[col] = combined_predictions[col].astype(int)

    # Save the final predictions dataset
    combined_predictions.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Predictions_2024_Jan_to_July.csv', index=False)
    print("Predictions saved successfully.")
else:
    print("No predictions were made due to insufficient data variation.")







# %%
# to predict july 2024
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
import numpy as np

#%%
# Load the Data
data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv'
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/output_month_7_2024.csv'
data = pd.read_csv(data_path)

# Save original ID, Email for the final output
original_data = data[['ID', 'Email']]
product_columns = [col for col in data.columns if col.startswith('Product_')]
month_year_columns = [col for col in data.columns if col.startswith('month_year_')]

# Drop irrelevant columns for model training
data = data.drop(columns=['Email', 'ID'])

# Convert date columns to numeric features
date_columns = ['last_purchase_date']
for col in date_columns:
    data[col] = pd.to_datetime(data[col])
    data[col] = (data[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Define the target variable and feature set
target_cols = ['purchase_in_month_1_2024', 'purchase_in_month_2_2024', 'purchase_in_month_3_2024', 
               'purchase_in_month_4_2024', 'purchase_in_month_5_2024', 'purchase_in_month_6_2024']

# Stack the targets to create a binary target variable for all periods except July 2024
y = data[target_cols].stack().reset_index(level=1, drop=True)
y = y.rename('purchase_event')
X = data.drop(columns=target_cols + ['purchase_in_month_7_2024'])

# Remove specific products
products_to_remove = ['Product_Wire Transfer Fee (Other Charge)', 'Product_Hazardous_Fee']
X = X.drop(columns=[col for col in X.columns if col in products_to_remove])

#%%
# Repeat the features for each target to match the length of the target variable
X_repeated = pd.concat([X] * len(target_cols), ignore_index=True)

# Ensure features and targets have the same length
assert len(X_repeated) == len(y), "Features and targets must have the same length"

# Replace missing values with 0
X_repeated = X_repeated.fillna(0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_repeated, y, test_size=0.3, random_state=42)
#%%
# Feature selection using RandomForestClassifier to determine feature importance
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Select features based on importance
sfm = SelectFromModel(clf, threshold='mean')
sfm.fit(X_train, y_train)
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

# Calculate class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
#%%
# Define the Model with class weights
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights_dict)

# Train the model
model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred = model.predict(X_test_selected)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, zero_division=1)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')
#%%
# Prepare the data for July 2024 prediction (using the existing features)
X_july_2024 = data.drop(columns=target_cols + ['purchase_in_month_7_2024'])

for col in date_columns:
    X_july_2024[col] = pd.to_datetime(X_july_2024[col])
    X_july_2024[col] = (X_july_2024[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

X_july_2024 = X_july_2024.fillna(0)

# Select the same features for July 2024 prediction
X_july_2024_selected = sfm.transform(X_july_2024)
july_2024_predictions = model.predict(X_july_2024_selected)

# Find the product with the highest value for each customer
predicted_products = []
for i in range(len(X_july_2024_selected)):
    max_product_index = np.argmax(X_july_2024.iloc[i][product_columns])
    predicted_product = product_columns[max_product_index]
    predicted_products.append(predicted_product)

# Prepare the final output DataFrame
original_data['predicted_product'] = predicted_products
original_data['predicted_target'] = july_2024_predictions

# Include ID, Email, predicted_product, and predicted_target in the output
output_data = original_data[['ID', 'Email', 'predicted_product', 'predicted_target']]
output_data.to_csv(output_path, index=False)

# Save rows where predicted_target = 1
predicted_purchases = output_data[output_data['predicted_target'] == 1]
predicted_purchases.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions/predicted_purchases.csv', index=False)

# %%
