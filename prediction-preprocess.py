#%% 
## Loading packages
import pandas as pd

#%%
## Data Preprocessing
sales_data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Sales21-24.csv')
customer_data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv')

sales_data['Customer ID'] = sales_data['Customer ID'].astype(str).str.strip()
customer_data['ID'] = customer_data['ID'].astype(str).str.strip()

combined_data = pd.merge(sales_data, customer_data, left_on='Customer ID', right_on='ID', how='left')

combined_data.rename(columns={
    'Date': 'Purchase Date',
    'Item': 'Product',
    'Qty. Sold': 'Unit Volume',
    'Amount': 'Total Money Spent'
}, inplace=True)
combined_data['Unit Price'] = combined_data['Total Money Spent'] / combined_data['Unit Volume']

final_data = combined_data[['Customer ID', 'Email', 'Purchase Date', 'Product', 'Unit Volume', 'Total Money Spent', 'Unit Price']].copy()
final_data.rename(columns={'Customer ID': 'Names'}, inplace=True)

final_data['Email'].fillna('No Email Provided', inplace=True)

final_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/FinalSalesData.csv', index=False)
print("Data preparation is complete. The final merged dataset is saved.")
#%%
print(final_data)
# %%
