
# %%
import pandas as pd

# Paths to the data files
purchase_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-product.csv'
rna_products_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/RNA Products.csv'
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'

# Load the datasets
purchase_data = pd.read_csv(purchase_data_path)
rna_products = pd.read_csv(rna_products_path)

# Ensure 'Customer ID' is treated as a string and remove '.0'
purchase_data['Customer ID'] = purchase_data['Customer ID'].apply(lambda x: str(int(x)) if not pd.isnull(x) else '')

# Filter the rows where Customer Status is 'New'
purchase_data = purchase_data[purchase_data['Customer Status'] == 'New']

# Filter out RNA products from purchase data
rna_product_set = set(rna_products['Product_ID'])  # Create a set of RNA product IDs for faster lookup
rna_purchases = purchase_data[purchase_data['Item'].isin(rna_product_set)]

# Save the filtered data
rna_purchases.to_csv(output_path, index=False)

print("Filtered RNA product purchase data saved successfully.")


# %%
# Load the dataset
data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'
purchase_data = pd.read_csv(data_path)

# Ensure the 'Date' column is in datetime format
purchase_data['Date'] = pd.to_datetime(purchase_data['Date'])

# Function to filter earliest and second earliest purchase dates for each customer
def filter_earliest_dates(group):
    # If there is only one unique date, return the group as is
    if group['Date'].nunique() == 1:
        return group
    else:
        # Find the earliest and second earliest dates
        earliest_date = group['Date'].min()
        return group[group['Date'].isin([earliest_date])]

# Apply the function to each group
filtered_data = purchase_data.groupby('Customer ID').apply(filter_earliest_dates).reset_index(drop=True)

# Save the filtered data 
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA-earliest.csv'
filtered_data.to_csv(output_path, index=False)

print("Filtered data saved successfully, containing only the earliest purchase dates for each customer.")
#%%
# Path to the CSV file
data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA-earliest.csv'
purchase_data = pd.read_csv(data_path)
item_frequencies = purchase_data['Item'].value_counts()
# Print the frequencies
print("Frequencies of each item:")
print(item_frequencies)
#%%




# %%
import pandas as pd

# Load the full purchase history
full_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'
data_path_earliest = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA-earliest.csv'

full_purchase_data = pd.read_csv(full_data_path)
purchase_data_earliest = pd.read_csv(data_path_earliest)

# Ensure all date fields are in datetime format and data is sorted
full_purchase_data['Date'] = pd.to_datetime(full_purchase_data['Date'])
full_purchase_data_sorted = full_purchase_data.sort_values(by=['Customer ID', 'Date'])

# Get item frequencies and descriptions from the earliest purchase data
item_frequencies = purchase_data_earliest['Item'].value_counts()

# Create a DataFrame with item frequencies and descriptions
df_frequencies = pd.DataFrame({
    '1st Purchase': [
        f"{item}({purchase_data_earliest.loc[purchase_data_earliest['Item'] == item, 'Item Description'].iloc[0]} - {freq})" 
        for item, freq in item_frequencies.items()
    ]
})
#%%
def analyze_second_purchase(data, item_frequencies):
    results = []
    
    for item, _ in item_frequencies.items():
        # Filter data for customers whose first purchase was this item
        customer_ids = data[(data['Item'] == item) & 
                            (data.groupby('Customer ID')['Date'].transform(min) == data['Date'])]['Customer ID'].unique()
        
        # Filter all subsequent purchases by these customers, excluding the first purchase
        subsequent_purchases = data[data['Customer ID'].isin(customer_ids) & (data['Item'] != item)]
        
        # For each customer, get the second purchase
        second_purchases = subsequent_purchases.groupby('Customer ID').nth(0)  # First item after the first purchase
        
        if not second_purchases.empty:
            # Calculate the frequency of the second purchases
            second_purchase_frequencies = second_purchases['Item'].value_counts()
            most_frequent_second = second_purchase_frequencies.idxmax()
            frequency = second_purchase_frequencies.max()
            most_frequent_second_desc = subsequent_purchases.loc[subsequent_purchases['Item'] == most_frequent_second, 'Item Description'].iloc[0] if 'Item Description' in subsequent_purchases.columns else "No description"
            results.append(f"{most_frequent_second}({most_frequent_second_desc} - {frequency})")
        else:
            results.append("N/A")

    return results
#%%
# Add second purchase frequency and description to the DataFrame
df_frequencies['2nd Purchase'] = analyze_second_purchase(full_purchase_data_sorted, item_frequencies)

# Save the final table
output_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA-subsequent.csv'
df_frequencies.to_csv(output_path, index=False)

print("Detailed purchase analysis with second purchases and descriptions has been saved to", output_path)

# %%
import pandas as pd

# Load the full purchase data
data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'
full_purchase_data = pd.read_csv(data_path)
full_purchase_data['Date'] = pd.to_datetime(full_purchase_data['Date'])
full_purchase_data_sorted = full_purchase_data.sort_values(by=['Customer ID', 'Date'])

# Identify customers whose first purchase was "R1013"
customers_first_purchase_R1013 = full_purchase_data_sorted[full_purchase_data_sorted['Item'] == 'R1013']
first_purchases_R1013 = customers_first_purchase_R1013.groupby('Customer ID').first().reset_index()

# Filter all purchases made by these customers, excluding the first purchase day
all_purchases_by_R1013_buyers = full_purchase_data_sorted[full_purchase_data_sorted['Customer ID'].isin(first_purchases_R1013['Customer ID'])]
second_purchases_by_R1013_buyers = all_purchases_by_R1013_buyers.groupby('Customer ID').apply(lambda x: x[x['Date'] > x.iloc[0]['Date']].iloc[0] if len(x[x['Date'] > x.iloc[0]['Date']]) > 0 else None).dropna()

# Calculate the frequency of the second purchases
second_purchase_frequencies = second_purchases_by_R1013_buyers['Item'].value_counts()

print("Second purchase analysis for customers whose first purchase was R1013:")
print(second_purchase_frequencies)

# %%
## Table for overall RNA top 10 Organization
import pandas as pd

# Load the full purchase history
full_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'
full_purchase_data = pd.read_csv(full_data_path)

# Ensure all date fields are in datetime format and data is sorted
full_purchase_data['Date'] = pd.to_datetime(full_purchase_data['Date'])
full_purchase_data_sorted = full_purchase_data.sort_values(by=['Customer ID', 'Date'])

# Calculate the frequency of organizations
organization_frequencies = full_purchase_data_sorted['Organization'].value_counts().head(10)

# Print the top 10 most frequent organizations
print("Top 10 most frequent organizations:")
print(organization_frequencies)


# %%
## Table for ranked RNA for top 10 organization
import pandas as pd

# Load the full purchase history
full_data_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/PersonDB-related/output/customer-RNA.csv'
full_purchase_data = pd.read_csv(full_data_path)

# Ensure all date fields are in datetime format and data is sorted
full_purchase_data['Date'] = pd.to_datetime(full_purchase_data['Date'])
full_purchase_data_sorted = full_purchase_data.sort_values(by=['Customer ID', 'Date'])

# Identify the first purchase for each customer
first_purchases = full_purchase_data_sorted.groupby('Customer ID').first().reset_index()

# Count the frequency of each first purchase product
first_purchase_counts = first_purchases['Item'].value_counts()

# Create a list to store the results
results = []

# For each product, find the top organizations that made that product their first purchase
for product in first_purchase_counts.index:
    product_purchases = first_purchases[first_purchases['Item'] == product]
    top_organization = product_purchases['Organization'].value_counts().idxmax()
    top_count = product_purchases['Organization'].value_counts().max()
    results.append({'Product': product, 'Top Organization': top_organization, 'Count': top_count})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

#%%
# Display the top 10 results
top_10_results = results_df.head(10)
print("Top organizations for each of the most popular first purchase products:")
print(top_10_results)
# %%
import pandas as pd

# File paths (update these to the correct paths)
purchase_file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
customers_file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv'

# Load the purchase data
purchase_data = pd.read_csv(purchase_file_path, dtype={'Customer ID': str})

# Load the customer data
customer_data = pd.read_csv(customers_file_path, dtype={'ID': str})

# Group by Customer ID, Email, and Company Name, and calculate the total amount spent by each customer
top_customers = purchase_data.groupby(['Customer ID', 'Email', 'Company Name']).agg({'Amount': 'sum'}).reset_index()

# Sort by the total amount spent in descending order and get the top 25 customers
top_100_customers = top_customers.sort_values(by='Amount', ascending=False).head(100)

# Merge with customer data to get the customer names
top_100_customers_with_name = top_100_customers.merge(customer_data[['ID', 'Name']], left_on='Customer ID', right_on='ID', how='left')

# Select the necessary columns
top_100_customers_with_name = top_100_customers_with_name[['Customer ID', 'Email', 'Name', 'Amount']]

# Display the top 100 customers
print("Top 100 Customers:")
print(top_100_customers_with_name)

# Save the result to a CSV file
top_100_customers_with_name.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/top_100_customers.csv', index=False)

# %%
