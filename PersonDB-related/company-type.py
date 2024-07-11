#%%
import pandas as pd
#%%
customer_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
item_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Item Dataset.csv'
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-product.csv'

customer_data = pd.read_csv(customer_data_path)
item_data = pd.read_csv(item_data_path, encoding='latin1')
customer_data.drop(columns=['Purchase Next Month'], inplace=True)

item_data = item_data[['Item name/SKU#', 'Description']]

# Merge datasets on 'Item' 
merged_data = pd.merge(customer_data, item_data, left_on='Item', right_on='Item name/SKU#', how='left')
merged_data.rename(columns={'Description': 'Item Description'}, inplace=True)
merged_data.rename(columns={'Item name/SKU#': 'Item'}, inplace=True)
#%%
# Define a function to classify the organization type based on email
def classify_organization(email):
    if '.edu' in email or '.ac.' in email or '.org' in email:
        return 'Academia'
    elif '.com' in email or 'zymo' in email:
        return 'Industry'
    elif '.gov' in email:
        return 'Government'
    else:
        return 'Other'

merged_data['Organization Type'] = merged_data['Email'].apply(classify_organization)

merged_data.to_csv(output_path, index=False)

# %%
print(f"Number of 'Other' organization types: {merged_data['Organization Type'].value_counts().get('Other', 0)}")

# %%
merged_data['Organization Type'].value_counts().plot.pie(autopct='%1.1f%%', title='Organization Type Distribution')

# %%
