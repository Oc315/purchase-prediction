#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%

customer_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
item_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/Item Dataset.csv'
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-product.csv'

customer_data = pd.read_csv(customer_data_path)
item_data = pd.read_csv(item_data_path, encoding='latin1')

customer_data.drop(columns=['Purchase Next Month'], inplace=True)
item_data = item_data[['Item name/SKU#', 'Description']]

# Merge on 'Item'
merged_data = pd.merge(customer_data, item_data, left_on='Item', right_on='Item name/SKU#', how='left')
merged_data.drop(columns=['Item name/SKU#'], inplace=True)
merged_data.rename(columns={'Description': 'Item Description', 'Company Name': 'Organization'}, inplace=True)
#%%
# Classification functions
def classify_organization_by_email(email):
    email = email.lower()
    if '.edu' in email or '.ac.' in email or '.org' in email:
        return 'Academia'
    elif '.com' in email or 'zymo' in email or '.co' in email:
        return 'Industry'
    elif '.gov' in email or '.fed' in email or '.us' in email or '.mil' in email:
        return 'Government'
    else:
        return 'Other'

def classify_organization_type(org_name):
    org_name = org_name.lower()
    academia_keywords = ['university', 'college', 'institute', 'school', 'academy', 'foundation', 'hospital', 'universidad']
    industry_keywords = ['inc', 'llc', 'ltd.', 'corporation', 'company', 'enterprise', 'biotech', 'biotechnology', 'protein', 'evolution', 'agriculture', 'pharmaceuticals', 'therapeutics', 'lab', 'health', 'tech', 'technology', '23andme', 'amazon',' aecom', 'scientific', 'industries', 'bio', 'canvio', 'trader', 'zymo', 'specialities', 'consultants', 'unimed healthcare', 'innovacion', 'solutions']
    government_keywords = ['bureau', 'government', 'military', 'federal', 'national', 'us naval research lab', 'centers for disease control', 'us environmental protection agency', 'u.s.']

    if any(keyword in org_name for keyword in academia_keywords):
        return 'Academia'
    elif any(keyword in org_name for keyword in industry_keywords):
        return 'Industry'
    elif any(keyword in org_name for keyword in government_keywords):
        return 'Government'
    else:
        return 'Other'
#%%
# Apply the classification functions
merged_data['Email Classification'] = merged_data['Email'].apply(classify_organization_by_email)
merged_data['Organization Classification'] = merged_data['Organization'].apply(classify_organization_type)

# Combining classifications
merged_data['Organization Type'] = np.where(merged_data['Email Classification'] != 'Other', 
                                            merged_data['Email Classification'], 
                                            merged_data['Organization Classification'])
# Drop the now unnecessary 'Email Classification' and 'Organization Classification' columns
merged_data.drop(['Email Classification', 'Organization Classification'], axis=1, inplace=True)

# Save the dataframe to CSV
merged_data.to_csv(output_path, index=False)

# Print the count of 'Other' organization types
print(f"Number of 'Other' organization types: {merged_data['Organization Type'].value_counts().get('Other', 0)}")
#%%
# Plot distribution of Organization Types
merged_data['Organization Type'].value_counts().plot.pie(autopct='%1.1f%%', title='Organization Type Distribution')
plt.show()
#%%
# Filter data where Organization Type is 'Other'
filtered_data = merged_data.loc[merged_data['Organization Type'] == 'Other']
print(filtered_data)

# Save filtered data to CSV
filtered_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/Other-OrgType.csv', index=False)

# %%
