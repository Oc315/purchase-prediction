#%%
import pandas as pd

# Define file paths
customer_product_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-product.csv'
rna_products_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/RNA Products.csv'
output_path_first = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/rna-1st.csv'
output_path_second = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/rna-2nd.csv'
#%%
# Load the datasets
customer_product_data = pd.read_csv(customer_product_path)
rna_products_data = pd.read_csv(rna_products_path, encoding='latin1')

# Rename the incorrect column name to 'RNA Product'
rna_products_data.rename(columns={'ï»¿Product_ID': 'RNA Item'}, inplace=True)

# Merge datasets 
merged_data = pd.merge(customer_product_data, rna_products_data, left_on='Item', right_on='RNA Item', how='inner')

# Sort the data by 'Customer ID' and 'Date' to ensure chronological order
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data.sort_values(by=['Customer ID', 'Date'], inplace=True)

# Group by 'Customer ID' and filter to keep only the first and second purchase records
filtered_data = merged_data.groupby('Customer ID').head(2)

# Extract second purchase records
second_purchases = filtered_data.groupby('Customer ID').nth(1).reset_index()

# Identify and remove second purchase records from the original filtered dataset
first_purchases = filtered_data[~filtered_data.set_index(['Customer ID', 'Date']).index.isin(second_purchases.set_index(['Customer ID', 'Date']).index)]

# Select relevant columns for first and second purchases, with rearranged order
columns_to_keep = ['Customer ID', 'Email', 'Company Name', 'RNA Item', 'Item Description', 'Date', 'Qty. Sold', 'Amount', 'Organization Type']
first_purchases = first_purchases[columns_to_keep]
second_purchases = second_purchases[columns_to_keep]

# Save the first purchase records to the specified path
first_purchases.to_csv(output_path_first, index=False)

# Save the second purchase records to the specified path
second_purchases.to_csv(output_path_second, index=False)

# Print a few rows of the datasets to verify the changes
print("First Purchases:")
print(first_purchases.head())

print("\nSecond Purchases:")
print(second_purchases.head())

# %%

rna_1st_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/rna-1st.csv'
rna_2nd_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/rna-2nd.csv'
output_summary_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/rna-summary.csv'
rna_1st = pd.read_csv(rna_1st_path)
rna_2nd = pd.read_csv(rna_2nd_path)

# Calculate total number of purchases
total_rna_1st = len(rna_1st)
total_rna_2nd = len(rna_2nd)

# Calculate the frequency of each RNA product
freq_rna_1st = rna_1st[['RNA Item', 'Item Description']].value_counts().reset_index(name='Frequency in RNA 1st')
freq_rna_2nd = rna_2nd[['RNA Item', 'Item Description']].value_counts().reset_index(name='Frequency in RNA 2nd')

# Merge the frequency tables on RNA Item and Item Description
summary = pd.merge(freq_rna_1st, freq_rna_2nd, on=['RNA Item', 'Item Description'], how='outer').fillna(0)
summary['Frequency in RNA 1st'] = summary['Frequency in RNA 1st'].astype(int)
summary['Frequency in RNA 2nd'] = summary['Frequency in RNA 2nd'].astype(int)

# Add the total number of purchases row at the top
summary = pd.concat([
    pd.DataFrame([['Total Purchases', 'N/A', total_rna_1st, total_rna_2nd]], columns=['RNA Item', 'Item Description', 'Frequency in RNA 1st', 'Frequency in RNA 2nd']),
    summary
])

# Save the summary table to the specified path
summary.to_csv(output_summary_path, index=False)

# %%
