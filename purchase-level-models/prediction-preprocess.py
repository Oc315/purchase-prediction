# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

#%%
import pandas as pd

# Preprocessing
customers_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/AllCustomers.csv'
sales_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SALESDATA.csv'

customers = pd.read_csv(customers_path, dtype={'ID': str})
sales = pd.read_csv(sales_path, dtype={'ID': str}, low_memory=False)

customers.columns = customers.columns.str.strip()
sales.columns = sales.columns.str.strip()

merged_data = pd.merge(sales, customers, on='ID', how='left')
merged_data['Email'] = merged_data['Email_y']
merged_data['Date'] = pd.to_datetime(merged_data['Date'])

# Sort the merged data by 'ID' and 'Date'
merged_data.sort_values(by=['ID', 'Date'], inplace=True)

# Fill missing 'Email' values with 'No Email'
merged_data['Email'].fillna('No Email', inplace=True)

# Filter out rows with non-positive 'Qty. Sold' and 'Amount'
merged_data = merged_data[(merged_data['Qty. Sold'] > 0) & (merged_data['Amount'] > 0)]

#
merged_data = merged_data.drop(columns=['Email_x', 'Top Level Parent_x', 'Email_y'])

print(merged_data.head())


#%%
# create target variable based on each customer
merged_data['YearMonth'] = merged_data['Date'].dt.to_period('M')
def create_target(df):
    df['Purchase Next Month'] = 0
    customers = df['Customer ID'].unique()
    
    for customer in tqdm(customers, desc="Processing customers"):
        customer_data = df[df['Customer ID'] == customer]
        purchase_dates = customer_data['YearMonth'].unique()
        
        for date in purchase_dates:
            next_month = date + 1
            if next_month in purchase_dates:
                df.loc[(df['Customer ID'] == customer) & (df['YearMonth'] == date), 'Purchase Next Month'] = 1
    
    return df

final_data = create_target(merged_data)
final_data.drop(columns=['YearMonth'], inplace=True)

final_data = final_data[['Customer ID', 'Email', 'Company Name', 'Item', 'Date', 'Qty. Sold', 'Amount', 'Customer Status', 'Purchase Next Month']]
final_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv', index=False)

#%%
# EDA
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
data = pd.read_csv(file_path, dtype={'Customer ID': str})
data['Date'] = pd.to_datetime(data['Date'])
desc_stats = data.describe(include='all')

custom_stats = {
    'Metric': [
        'Total Customers',
        'Total Transactions',
        'Total Revenue',
        'Average Quantity Sold per Transaction',
        'Average Transaction Amount',
        'Unique Items Sold',
        'Most Common Customer Status'
    ],
    'Value': [
        data['Customer ID'].nunique(),
        data.shape[0],
        data['Amount'].sum(),
        data['Qty. Sold'].mean(),
        data['Amount'].mean(),
        data['Item'].nunique(),
        data['Customer Status'].mode()[0]
    ]
}
custom_stats_df = pd.DataFrame(custom_stats)

# display descriptive stats
print("Custom Descriptive Statistics:")
print(tabulate(custom_stats_df, headers='keys', tablefmt='grid'))
print("\nGeneral Descriptive Statistics:")
print(desc_stats)
#%% 
# pick 50-100 customers
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
data = pd.read_csv(file_path, dtype={'Customer ID': str})

data['Date'] = pd.to_datetime(data['Date'])
customer_totals = data.groupby('Customer ID').agg({'Amount': 'sum'}).reset_index()

# Sort customers by total amount spent and select a range of customers
selected_customers = customer_totals.sort_values(by='Amount', ascending=False)
selected_customers = pd.concat([
    selected_customers.head(5),  # Top spenders
    selected_customers.tail(5),  # Bottom spenders
    selected_customers.sample(10, random_state=42)  # Random spenders
]).drop_duplicates().head(20)  # Ensure we have unique customers


selected_customer_ids = selected_customers['Customer ID'].tolist()
selected_histories = data[data['Customer ID'].isin(selected_customer_ids)]


selected_histories['YearMonth'] = selected_histories['Date'].dt.to_period('M').astype(str)
plot_data = selected_histories.groupby(['Customer ID', 'Company Name', 'YearMonth']).agg({'Amount': 'sum'}).reset_index()

# Plot purchases over months for selected customers
plt.figure(figsize=(15, 10))
sns.lineplot(data=plot_data, x='YearMonth', y='Amount', hue='Company Name', palette="tab20", linewidth=2.5)
plt.title('Monthly Purchases for Selected Customers')
plt.xlabel('Month')
plt.ylabel('Total Purchase Amount')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
#%%
# Assuming data is your DataFrame with a 'Date' and 'Amount' columns

# Filter data for the desired date range
filtered_data = data[(data['Date'].dt.year >= 2021) & (data['Date'].dt.year <= 2023)]

# Extract year and month
filtered_data['YearMonth'] = filtered_data['Date'].dt.to_period('M')

monthly_sales = filtered_data.groupby('YearMonth')['Amount'].sum().reset_index()

monthly_sales['YearMonth'] = monthly_sales['YearMonth'].dt.to_timestamp()

monthly_sales['Amount_Z'] = (monthly_sales['Amount'] - monthly_sales['Amount'].mean()) / monthly_sales['Amount'].std()

# Plotting the Z-scores
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['YearMonth'], monthly_sales['Amount_Z'], marker='o')
plt.title('Z-Scores of Monthly Sales Amount from 2021 to 2023')
plt.xlabel('Month')
plt.ylabel('Z-Score of Total Sales Amount')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%

# Convert 'Date' column to datetime if it's not already
data['Date'] = pd.to_datetime(data['Date'])

# Extract Purchase Year, Purchase Month, and Purchase Day from the Date column
data['Purchase Year'] = data['Date'].dt.year
data['Purchase Month'] = data['Date'].dt.month
data['Purchase Day'] = data['Date'].dt.day

# Function to calculate Z-scores
def calculate_z_scores(series):
    return (series - series.mean()) / series.std()

# Calculate the frequency of purchases by year/month
purchase_month_freq = data.groupby(['Purchase Year', 'Purchase Month']).size().unstack(fill_value=0)
purchase_day_freq = data.groupby(['Purchase Year', 'Purchase Day']).size().unstack(fill_value=0)

# Calculate Z-scores for the frequencies
purchase_month_freq_z = purchase_month_freq.apply(calculate_z_scores)
purchase_day_freq_z = purchase_day_freq.apply(calculate_z_scores)

# Plot Z-scores of Purchase Month frequencies
plt.figure(figsize=(12, 6))
for year in purchase_month_freq_z.index:
    plt.plot(purchase_month_freq_z.columns, purchase_month_freq_z.loc[year], marker='o', label=year)

plt.title('Z-Scores of Purchase Frequencies by Month')
plt.xlabel('Month')
plt.ylabel('Z-Score')
plt.xticks(range(1, 13))
plt.legend(title='Year')
plt.grid(axis='y')
plt.show()

# Plot Z-scores of Purchase Day frequencies
plt.figure(figsize=(12, 6))
for year in purchase_day_freq_z.index:
    plt.plot(purchase_day_freq_z.columns, purchase_day_freq_z.loc[year], marker='o', label=year)

plt.title('Z-Scores of Purchase Frequencies by Day')
plt.xlabel('Day of the Month')
plt.ylabel('Z-Score')
plt.xticks(range(1, 32))
plt.legend(title='Year')
plt.grid(axis='y')
plt.show()

# Calculate the frequency of purchases by year
purchase_year_freq = data['Purchase Year'].value_counts().sort_index()
purchase_year_freq_z = calculate_z_scores(purchase_year_freq)

# Plot histogram for Z-scores of Purchase Year frequencies
plt.figure(figsize=(12, 6))
plt.bar(purchase_year_freq_z.index, purchase_year_freq_z, edgecolor='black')
plt.title('Z-Scores of Purchase Frequencies by Year')
plt.xlabel('Year')
plt.ylabel('Z-Score')
plt.xticks(purchase_year_freq_z.index)  # Ensure correct year labels
plt.grid(axis='y')
plt.show()

# For the monthly sales amount Z-scores plot
# Filter data for the desired date range
filtered_data = data[(data['Date'].dt.year >= 2021) & (data['Date'].dt.year <= 2023)]

# Extract year and month
filtered_data['YearMonth'] = filtered_data['Date'].dt.to_period('M')

# Group by YearMonth and sum the Amount
monthly_sales = filtered_data.groupby('YearMonth')['Amount'].sum().reset_index()

# Convert 'YearMonth' back to datetime for plotting
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].dt.to_timestamp()

# Calculate Z-scores for the monthly sales amounts
monthly_sales['Amount_Z'] = (monthly_sales['Amount'] - monthly_sales['Amount'].mean()) / monthly_sales['Amount'].std()

# Plotting the Z-scores
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['YearMonth'], monthly_sales['Amount_Z'], marker='o')
plt.title('Z-Scores of Monthly Sales Amount from 2021 to 2023')
plt.xlabel('Month')
plt.ylabel('Z-Score of Total Sales Amount')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
final_data['Date_Num'] = final_data['Date'].astype(int) // 10**9 

features = ['Date_Num', 'Qty. Sold', 'Amount']
X = final_data[features]
y = final_data['Purchase Next Month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
# SMOTE 
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

y_pred_rf = rf_model.predict(X_test)

# evaluate 
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, zero_division=1)
recall_rf = recall_score(y_test, y_pred_rf, zero_division=1)

print(f'Random Forest Accuracy: {accuracy_rf}')
print(f'Random Forest Precision: {precision_rf}')
print(f'Random Forest Recall: {recall_rf}')

# Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
cm_display_rf = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rf, display_labels=[0, 1])
cm_display_rf.plot()
#%%
# Rename columns
customers.rename(columns={'ID': 'Customer ID', 'Top Level Parent': 'Company'}, inplace=True)
#%%
# Function to predict purchases for the next month 
def predict_next_month(df, customers, year_month):

    df['Date_Num'] = df['Date'].astype(int) // 10**9  
    month_data = df[df['Date'].dt.to_period('M') == year_month]
    X_future = month_data[['Date_Num', 'Qty. Sold', 'Amount']]
    
    # Predict
    predictions = rf_model.predict(X_future)
    month_data['Prediction'] = predictions
    
    # Ensure each customer appears only once
    month_data = month_data.groupby('Customer ID').first().reset_index()
    
    all_customers = customers[['Customer ID', 'Email', 'Company']].copy()
    prediction_results = pd.merge(all_customers, month_data[['Customer ID', 'Prediction']], on='Customer ID', how='left')
    prediction_results['Prediction'].fillna(0, inplace=True)  # Fill NaN predictions with 0 (no purchase predicted)
    
    next_month = (pd.Period(year_month) + 1).strftime('%Y-%m')
    output_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions_{next_month}.csv'
    prediction_results.to_csv(output_path, index=False)
    
    return prediction_results

# Example usage: using year_month to predict next month
year_month = '2024-04'
predictions = predict_next_month(final_data, customers, year_month)


# %%
# Evaluation
def evaluate_predictions(df, predictions_path, year_month):
    """
    Evaluate the predictions by comparing them with the actual data for the specified month.
    
    Parameters:
    df (pd.DataFrame): The original dataframe containing the actual data.
    predictions_path (str): Path to the CSV file containing the predictions.
    year_month (str): The month and year for which the predictions are to be evaluated (format: 'YYYY-MM').
    
    Returns:
    pd.DataFrame: A dataframe with Customer ID, actual purchase next month, and prediction.
    """
    predictions = pd.read_csv(predictions_path)
    predictions['Customer ID'] = predictions['Customer ID'].astype(str)

    if 'Customer ID' not in predictions.columns or 'Prediction' not in predictions.columns:
        raise ValueError("Predictions file must contain 'Customer ID' and 'Prediction' columns")

    actual_data = df[df['Date'].dt.to_period('M') == year_month]
    actual_data['Customer ID'] = actual_data['Customer ID'].astype(str)

    if 'Customer ID' not in actual_data.columns or 'Purchase Next Month' not in actual_data.columns:
        raise ValueError("Actual data must contain 'Customer ID' and 'Purchase Next Month' columns")
    actual_data = actual_data.groupby('Customer ID').first().reset_index()

    evaluation_data = pd.merge(predictions, actual_data[['Customer ID', 'Purchase Next Month']], on='Customer ID', how='left')
    evaluation_data['Purchase Next Month'].fillna(0, inplace=True)  # Assume customers not present in the month didn't make a purchase

    evaluation_data = evaluation_data.groupby('Customer ID').first().reset_index()

    accuracy = accuracy_score(evaluation_data['Purchase Next Month'], evaluation_data['Prediction'])
    precision = precision_score(evaluation_data['Purchase Next Month'], evaluation_data['Prediction'], zero_division=1)
    recall = recall_score(evaluation_data['Purchase Next Month'], evaluation_data['Prediction'], zero_division=1)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(evaluation_data['Purchase Next Month'], evaluation_data['Prediction'])
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    cm_display.plot()
    
    return evaluation_data[['Customer ID', 'Prediction']]

# Example usage:
predictions_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions_2024-05.csv'
year_month = '2024-05'
evaluation_results = evaluate_predictions(final_data, predictions_path, year_month)
print(evaluation_results)

# %%
# Load the predictions CSV file
predictions_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions_2024-05.csv'
predictions_df = pd.read_csv(predictions_path)

# Filter and print rows where Prediction equals 1
print(predictions_df[predictions_df['Prediction'] == 1])
#%%
### evaluate each month in 2024 
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

prediction_month = 6  # Change this number to evaluate a different month

predictions_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions_2024-{prediction_month:02d}.csv'
salesdata_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SALESDATA.csv'
output_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/evaluation_2024-{prediction_month:02d}.csv'
#%%
# Load the predictions and actual data
predictions = pd.read_csv(predictions_path)
actual_data = pd.read_csv(salesdata_path)

actual_data['Date'] = pd.to_datetime(actual_data['Date'])
actual_month_2024 = actual_data[(actual_data['Date'].dt.year == 2024) & (actual_data['Date'].dt.month == prediction_month)]
actual_month_2024['Purchase'] = 1
actual_month_2024_agg = actual_month_2024.groupby(['ID', 'Email', 'Top Level Parent']).agg({'Purchase': 'sum'}).reset_index()
actual_month_2024_agg['Purchase'] = actual_month_2024_agg['Purchase'].apply(lambda x: 1 if x > 0 else 0)

merged_data = pd.merge(predictions, actual_month_2024_agg, left_on='Customer ID', right_on='ID', how='left', suffixes=('_pred', '_actual'))
merged_data['Purchase'].fillna(0, inplace=True)
merged_data.rename(columns={'Purchase': 'Actual', 'Prediction': 'Predicted'}, inplace=True)


output_data = merged_data[['Customer ID', 'Top Level Parent', 'Email_pred', 'Predicted', 'Actual']]
output_data.rename(columns={'Email_pred': 'Email'}, inplace=True)

output_data.dropna(inplace=True)
#%%
# Remove duplicates based on 'Customer ID' and 'Company', keeping the first occurrence
output_data = output_data.drop_duplicates(subset='Top Level Parent', keep='first')
#%%
# print output
output_data['Predicted'] = output_data['Predicted'].astype(int)
output_data['Actual'] = output_data['Actual'].astype(int)
output_data = output_data[output_data['Actual'] == 1]
output_data.to_csv(output_path, index=False)

# %%
