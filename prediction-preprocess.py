#%% 
## Loading packages
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
sales_data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales21-24.csv')
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
# EDA: Plot histogram for Purchase Month
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Month'], bins=12, edgecolor='black')
plt.title('Histogram of Purchases by Month')
plt.xlabel('Month')
plt.ylabel('Number of Purchases')
plt.xticks(range(1, 13))
plt.grid(axis='y')
plt.show()

# Plot histogram for Purchase Day
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Day'], bins=31, edgecolor='black')
plt.title('Histogram of Purchases by Day')
plt.xlabel('Day of the Month')
plt.ylabel('Number of Purchases')
plt.xticks(range(1, 32))
plt.grid(axis='y')
plt.show()

# Plot histogram for Purchase Year
plt.figure(figsize=(10, 5))
plt.hist(model1_data['Purchase Year'], bins=len(model1_data['Purchase Year'].unique()), edgecolor='black')
plt.title('Histogram of Purchases by Year')
plt.xlabel('Year')
plt.ylabel('Number of Purchases')
plt.xticks(model1_data['Purchase Year'].unique())
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
