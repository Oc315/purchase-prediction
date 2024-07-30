#%% 
## Loading packages
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
sales_data_21_24 = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales21-24.csv', dtype={'Customer ID': str})
sales_data_15_21 = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales15-21.csv', dtype={'Customer ID': str})
sales_data = pd.concat([sales_data_15_21, sales_data_21_24], ignore_index=True)
sales_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales15-24.csv', index=False)
#%%
sales_data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales15-24.csv')
customer_data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv')

sales_data['Customer ID'] = sales_data['Customer ID'].astype(str).str.strip()
customer_data['ID'] = customer_data['ID'].astype(str).str.strip()

# Merge the datasets using customer IDs
combined_model1_data = pd.merge(sales_data, customer_data, left_on='Customer ID', right_on='ID', how='left')

combined_model1_data.rename(columns={
    'Date': 'Purchase Date',
    'Item': 'Product',
    'Qty. Sold': 'Unit Volume',
    'Amount': 'Total Money Spent'
}, inplace=True)

# Calculate Unit Price
combined_model1_data['Unit Price'] = combined_model1_data['Total Money Spent'] / combined_model1_data['Unit Volume']

model1_data = combined_model1_data[['Customer ID', 'Email', 'Purchase Date', 'Product', 'Unit Volume', 'Total Money Spent', 'Unit Price']].copy()
model1_data.rename(columns={'Customer ID': 'Names'}, inplace=True)

model1_data['Email'].fillna('No Email Provided', inplace=True)

model1_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv', index=False)

#%%n
print(model1_data)

# %%
# EDA: Data cleaning and feature engineering
# Remove negative values: shipping or refund
model1_data = model1_data[(model1_data['Unit Volume'] > 0) & (model1_data['Total Money Spent'] > 0)]

model1_data = model1_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['Unit Price'])

model1_data['Purchase Date'] = pd.to_datetime(model1_data['Purchase Date'])

# Extract Purchase Year, Month and Day
model1_data['Purchase Year'] = model1_data['Purchase Date'].dt.year
model1_data['Purchase Month'] = model1_data['Purchase Date'].dt.month
model1_data['Purchase Day'] = model1_data['Purchase Date'].dt.day

cleaned_enhanced_file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
model1_data.to_csv(cleaned_enhanced_file_path, index=False)

# %%
# Function to calculate Z-scores
def calculate_z_scores(series):
    return (series - series.mean()) / series.std()

model1_data['Purchase Month Z'] = calculate_z_scores(model1_data['Purchase Month'])
model1_data['Purchase Day Z'] = calculate_z_scores(model1_data['Purchase Day'])
model1_data['Purchase Year Z'] = calculate_z_scores(model1_data['Purchase Year'])

# Plot histogram for Purchase Month Z-scores
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Month Z'], bins=30, edgecolor='black')
plt.title('Histogram of Z-Scores for Purchases by Month')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Plot histogram for Purchase Day Z-scores
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Day Z'], bins=30, edgecolor='black')
plt.title('Histogram of Z-Scores for Purchases by Day')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Plot histogram for Purchase Year Z-scores
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Year Z'], bins=30, edgecolor='black')
plt.title('Histogram of Z-Scores for Purchases by Year')
plt.xlabel('Z-Score')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()
# %%
# EDA: Plot histogram for Purchase Month for Each Year
unique_years = model1_data['Purchase Year'].unique()
for year in unique_years:
    yearly_data = model1_data[model1_data['Purchase Year'] == year]
    plt.figure(figsize=(10, 5))
    plt.hist(yearly_data['Purchase Month'], bins=12, edgecolor='black')
    plt.title(f'Histogram of Purchases by Month in {year}')
    plt.xlabel('Month')
    plt.ylabel('Number of Purchases')
    plt.xticks(range(1, 13))
    plt.grid(axis='y')
    plt.show()

# Plot histogram for Purchase Day for each Year
for year in unique_years:
    yearly_data = model1_data[model1_data['Purchase Year'] == year]
    plt.figure(figsize=(10, 5))
    plt.hist(yearly_data['Purchase Day'], bins=31, edgecolor='black')
    plt.title(f'Histogram of Purchases by Day in {year}')
    plt.xlabel('Day of the Month')
    plt.ylabel('Number of Purchases')
    plt.xticks(range(1, 32))
    plt.grid(axis='y')
    plt.show()
# %%
# EDA: Aggregate Total Money Spent by Year
yearly_spending = model1_data.groupby('Purchase Year')['Total Money Spent'].sum().reset_index()

# Plot Total Money Spent by Year
plt.figure(figsize=(10, 5))
bars = plt.bar(yearly_spending['Purchase Year'], yearly_spending['Total Money Spent'], color='skyblue', edgecolor='black')
plt.title('Total Money Spent by Year')
plt.xlabel('Year')
plt.ylabel('Total Money Spent')
plt.xticks(yearly_spending['Purchase Year'])
plt.grid(axis='y')
plt.show()

# Aggregate Total Money Spent by Month
monthly_spending = model1_data.groupby('Purchase Month')['Total Money Spent'].sum().reset_index()

# Plot Total Money Spent by Month
plt.figure(figsize=(10, 5))
plt.bar(monthly_spending['Purchase Month'], monthly_spending['Total Money Spent'], color='lightgreen', edgecolor='black')
plt.title('Total Money Spent by Month')
plt.xlabel('Month')
plt.ylabel('Total Money Spent')
plt.xticks(range(1, 13))
plt.grid(axis='y')
plt.show()
# %%
# EDA: Total Money spent
# Filter out the year 2024
filtered_data = model1_data[model1_data['Purchase Date'].dt.year < 2024]

# Extract Purchase Year for filtered data
filtered_data['Purchase Year'] = filtered_data['Purchase Date'].dt.year

# Calculate the total money spent for each year (2021 and 2022)
total_money_2021 = filtered_data[filtered_data['Purchase Year'] == 2021]['Total Money Spent'].sum()
total_money_2022 = filtered_data[filtered_data['Purchase Year'] == 2022]['Total Money Spent'].sum()
total_money_2023 = filtered_data[filtered_data['Purchase Year'] == 2023]['Total Money Spent'].sum()

print(f"Total Money Spent in 2021: ${total_money_2021:.2f}")
print(f"Total Money Spent in 2022: ${total_money_2022:.2f}")
print(f"Total Money Spent in 2022: ${total_money_2023:.2f}")

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
data = pd.read_csv(file_path)

# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Sort the dataset by 'Names' (Customer ID) and 'Purchase Date'
data = data.sort_values(by=['Names', 'Purchase Date'])

# Rename 'Names' to 'ID'
data.rename(columns={'Names': 'ID'}, inplace=True)

# Create a new DataFrame for aggregated features
customer_features = pd.DataFrame()
customer_features['ID'] = data['ID'].unique()

# Calculate purchase frequency
purchase_frequency = data.groupby('ID').size().reset_index(name='purchase_frequency')

# Calculate total and mean spent
total_spent = data.groupby('ID')['Total Money Spent'].sum().reset_index(name='total_spent')
mean_spent = data.groupby('ID')['Total Money Spent'].mean().reset_index(name='mean_spent')

# Calculate mean and max purchase period
data['previous_purchase_date'] = data.groupby('ID')['Purchase Date'].shift(1)
data['purchase_period'] = (data['Purchase Date'] - data['previous_purchase_date']).dt.days
mean_purchase_period = data.groupby('ID')['purchase_period'].mean().reset_index(name='mean_purchase_period')
max_purchase_period = data.groupby('ID')['purchase_period'].max().reset_index(name='max_purchase_period')

# Calculate max spent per purchase
max_spent = data.groupby('ID')['Total Money Spent'].max().reset_index(name='max_spent')

# Get the last purchase date
last_purchase_date = data.groupby('ID')['Purchase Date'].max().reset_index(name='last_purchase_date')

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
email = data[['ID', 'Email']].drop_duplicates()
customer_features = customer_features.merge(email, on='ID', how='left')

# One-hot encode the product feature
product_encoded = pd.get_dummies(data[['ID', 'Product']], columns=['Product'])
product_encoded = product_encoded.groupby('ID').sum().reset_index()

# Merge one-hot encoded product features into customer_features
customer_features = customer_features.merge(product_encoded, on='ID', how='left')

# Display the final customer_features DataFrame
print(customer_features.head())

# Save the final dataset
customer_features.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv', index=False)

print("Customer-level dataset created successfully.")

# %%
import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
data = pd.read_csv(file_path)

# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Split the dataset into records before 2024 and records in and after 2024
data_before_2024 = data[data['Purchase Date'] < '2024-01-01']
data_2024_and_after = data[data['Purchase Date'] >= '2024-01-01']

# Sort the dataset by 'Names' (Customer ID) and 'Purchase Date'
data_before_2024 = data_before_2024.sort_values(by=['Names', 'Purchase Date'])
data_2024_and_after = data_2024_and_after.sort_values(by=['Names', 'Purchase Date'])

# Rename 'Names' to 'ID'
data_before_2024.rename(columns={'Names': 'ID'}, inplace=True)
data_2024_and_after.rename(columns={'Names': 'ID'}, inplace=True)

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

# Add the '2024' column to indicate whether the customer made a purchase in 2024
customer_features['2024'] = customer_features['ID'].isin(data_2024_and_after['ID']).astype(int)

# One-hot encode the product feature
product_encoded = pd.get_dummies(data_before_2024[['ID', 'Product']], columns=['Product'])
product_encoded = product_encoded.groupby('ID').sum().reset_index()

# Merge one-hot encoded product features into customer_features
customer_features = customer_features.merge(product_encoded, on='ID', how='left')

# Display the final customer_features DataFrame
print(customer_features.head())

# Save the final dataset
customer_features.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv', index=False)

print("Customer-level dataset created successfully.")


# %%
## currently using this
import pandas as pd
import numpy as np
from datetime import datetime
#%%

file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv'
data = pd.read_csv(file_path)

# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Split the dataset into records before 2024 and records in and after 2024
data_before_2024 = data[data['Purchase Date'] < '2024-01-01']
data_2024_and_after = data[data['Purchase Date'] >= '2024-01-01']

# Sort the dataset by 'Names' (Customer ID) and 'Purchase Date'
data_before_2024 = data_before_2024.sort_values(by=['Names', 'Purchase Date'])
data_2024_and_after = data_2024_and_after.sort_values(by=['Names', 'Purchase Date'])

# Rename 'Names' to 'ID'
data_before_2024.rename(columns={'Names': 'ID'}, inplace=True)
data_2024_and_after.rename(columns={'Names': 'ID'}, inplace=True)

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

# One-hot encode the purchase dates by month for 2021 to 2023
data_before_2024['month_year'] = data_before_2024['Purchase Date'].dt.to_period('M')
purchase_dates_encoded = pd.get_dummies(data_before_2024[['ID', 'month_year']], columns=['month_year'])
purchase_dates_encoded = purchase_dates_encoded.groupby('ID').sum().reset_index()

# Merge one-hot encoded purchase dates into customer_features
customer_features = customer_features.merge(purchase_dates_encoded, on='ID', how='left')

# One-hot encode purchase dates by month for 2024
for month in range(1, 7):
    month_column = f'purchase_in_month_{month}_2024'
    data_2024_and_after[month_column] = data_2024_and_after['Purchase Date'].dt.month == month
    data_2024_and_after[month_column] = data_2024_and_after[month_column].astype(int)

# Aggregate the monthly purchase indicators for 2024
monthly_2024_encoded = data_2024_and_after.groupby('ID')[['purchase_in_month_' + str(month) + '_2024' for month in range(1, 7)]].max().reset_index()

# Merge the 2024 monthly purchase indicators into customer_features
customer_features = customer_features.merge(monthly_2024_encoded, on='ID', how='left')

# Fill NaN values with 0
customer_features.fillna(0, inplace=True)

# Display the final customer_features DataFrame
print(customer_features.head())

# Save the final dataset
customer_features.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv', index=False)

print("Customer-level dataset created successfully.")

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
#%%
# Load the customer-level dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv'
customer_data = pd.read_csv(file_path)

# Convert 'last_purchase_date' to datetime
customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])

# Create 'days_since_last_purchase' feature (number of days since a reference date)
reference_date = pd.to_datetime('2024-01-01')
customer_data['days_since_last_purchase'] = (reference_date - customer_data['last_purchase_date']).dt.days

# Define the features (X) excluding the target columns for each month in 2024
X = customer_data.drop(columns=['ID', 'Email', 'last_purchase_date', 'purchase_in_month_1_2024', 'purchase_in_month_2_2024',
                                'purchase_in_month_3_2024', 'purchase_in_month_4_2024', 'purchase_in_month_5_2024',
                                'purchase_in_month_6_2024'])

# One-hot encode categorical features if any
X = pd.get_dummies(X, drop_first=True)

# Handle any remaining NaN values in the feature set
X.fillna(0, inplace=True)
#%%
# Loop through each month to build and evaluate a model
for month in range(1, 7):
    # Define the target variable for the specific month
    y = customer_data[f'purchase_in_month_{month}_2024']
    
    # Check if there are at least two classes in the target variable
    if len(y.unique()) > 1:
        # Split the data into training (70%) and testing (30%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train the Logistic Regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        # Save the trained model
        model_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/models/trained_logistic_regression_model_month_{month}.pkl'
        joblib.dump(model, model_path)
        
        # Predict on the test set
        predictions = model.predict(X_test)
        
        # Prepare the final output dataframe with ID, email, and predicted target
        output_df = X_test.copy()
        output_df['predicted_target'] = predictions
        output_df['ID'] = customer_data.loc[X_test.index, 'ID']
        output_df['Email'] = customer_data.loc[X_test.index, 'Email'].fillna('No Email')
        
        # Extract the product columns and find the product with the highest probability for each customer
        product_columns = [col for col in customer_data.columns if col.startswith('Product_')]
        output_df['Product'] = customer_data.loc[X_test.index, product_columns].idxmax(axis=1)
        
        # Select relevant columns to display
        final_output = output_df[['ID', 'Email', 'Product', 'predicted_target']]
        print(f"Results for month {month}:\n", final_output.head())
        
        # Save the final output to a CSV file
        final_output.to_csv(f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/output_month_{month}_2024.csv', index=False)
        
        # Print any one result where 'predicted_target' equals 1
        if not final_output[final_output['predicted_target'] == 1].empty:
            print(f"Predicted purchases for month {month}:\n", final_output[final_output['predicted_target'] == 1].sample(n=1))
        else:
            print(f"No predicted purchases in month {month} 2024.")
        
        # Print out the classification report
        report = classification_report(y_test, predictions)
        print(f"Classification Report for month {month}:\n", report)
        
        # Display confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        print(f"Confusion Matrix for month {month}:\n", conf_matrix)
        
        # Interpretation
        print("\nInterpretation:")
        print("Classification Report:")
        print("Precision: The proportion of positive identifications that were actually correct.")
        print("Recall: The proportion of actual positives that were correctly identified.")
        print("F1-Score: The harmonic mean of precision and recall.")
        print("Support: The number of true instances for each label.")
        
        print("\nConfusion Matrix:")
        print("The confusion matrix shows the counts of true positive, true negative, false positive, and false negative predictions.")
        print(f"True Negatives (TN): {conf_matrix[0, 0]}")
        print(f"False Positives (FP): {conf_matrix[0, 1]}")
        print(f"False Negatives (FN): {conf_matrix[1, 0]}")
        print(f"True Positives (TP): {conf_matrix[1, 1]}")
        
        print(f"\nOverall accuracy: {(conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum():.2f}")
        
        # Count the number of predictions made
        num_predictions = final_output['predicted_target'].count()
        print(f"Number of predictions made for month {month}: {num_predictions}\n")
    else:
        print(f"Skipping month {month} due to insufficient class variation.")

# %%
# train a model jan to june 2024

# Load the customer-level dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv'
customer_data = pd.read_csv(file_path)

# Convert 'last_purchase_date' to datetime
customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])

# Create 'days_since_last_purchase' feature (number of days since a reference date)
reference_date = pd.to_datetime('2024-01-01')
customer_data['days_since_last_purchase'] = (reference_date - customer_data['last_purchase_date']).dt.days

# Define the features (X) and target (y) using data from January to June
X = customer_data.drop(columns=['ID', 'Email', 'last_purchase_date', 'purchase_in_month_7_2024'])
target_cols = ['purchase_in_month_1_2024', 'purchase_in_month_2_2024', 'purchase_in_month_3_2024', 
               'purchase_in_month_4_2024', 'purchase_in_month_5_2024', 'purchase_in_month_6_2024']

# Stack the targets to create a binary target variable for all periods from January to June
y = customer_data[target_cols].stack().reset_index(level=1, drop=True)
y = y.rename('purchase_event')

# Repeat the features for each target to match the length of the target variable
X_repeated = pd.concat([X] * len(target_cols), ignore_index=True)

# Ensure features and targets have the same length
assert len(X_repeated) == len(y), "Features and targets must have the same length"

# Handle any remaining NaN values in the feature set
X_repeated.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_repeated, y, test_size=0.3, random_state=42, stratify=y)
#%%
# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/models/trained_logistic_regression_model_jan_to_june.pkl'
joblib.dump(model, model_path)
#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# %%
# Load the trained model to predict 2024
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the trained model
model_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/models/trained_logistic_regression_model_month_{month}.pkl'
joblib.dump(model, model_path)

# Load the customer-level dataset
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/CustomerFeatures_Model1.csv'
customer_data = pd.read_csv(file_path)

# Convert 'last_purchase_date' to datetime
customer_data['last_purchase_date'] = pd.to_datetime(customer_data['last_purchase_date'])

# Create 'days_since_last_purchase' feature (number of days since a reference date)
reference_date = pd.to_datetime('2024-01-01')
customer_data['days_since_last_purchase'] = (reference_date - customer_data['last_purchase_date']).dt.days

# Define the features (X) excluding the target columns for each month in 2024
X = customer_data.drop(columns=['ID', 'Email', 'last_purchase_date', 'purchase_in_month_1_2024', 'purchase_in_month_2_2024',
                                'purchase_in_month_3_2024', 'purchase_in_month_4_2024', 'purchase_in_month_5_2024',
                                'purchase_in_month_6_2024', 'purchase_in_month_7_2024'])

# One-hot encode categorical features if any
X = pd.get_dummies(X, drop_first=True)

# Handle any remaining NaN values in the feature set
X.fillna(0, inplace=True)

# Ensure feature names match between training and prediction datasets
X_july_2024 = X.copy()

# Predict purchase events for July 2024
july_2024_predictions = model.predict(X_july_2024)

# Extract product columns
product_columns = [col for col in customer_data.columns if col.startswith('Product_')]

# Find the product with the highest value for each customer
predicted_products = []
for i in range(len(X_july_2024)):
    max_product_index = np.argmax(customer_data.iloc[i][product_columns])
    predicted_product = product_columns[max_product_index]
    predicted_products.append(predicted_product)

# Prepare the final output DataFrame
original_data = customer_data[['ID', 'Email']]
original_data['predicted_product'] = predicted_products
original_data['predicted_target'] = july_2024_predictions

# Include ID, Email, predicted_product, and predicted_target in the output
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/output_month_7_2024.csv'
output_data = original_data[['ID', 'Email', 'predicted_product', 'predicted_target']]
output_data.to_csv(output_path, index=False)

# Save rows where predicted_target = 1
predicted_purchases = output_data[output_data['predicted_target'] == 1]
predicted_purchases.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions/predicted_purchases.csv', index=False)

# Print evaluation metrics for debugging purposes
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# %%
