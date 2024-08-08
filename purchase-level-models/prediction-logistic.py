#%%
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Preprocessing
sales_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SALESDATA.csv'

# Read the sales data
sales = pd.read_csv(sales_path, dtype={'ID': str}, low_memory=False)
sales.columns = sales.columns.str.strip()

# Ensure the 'Date' column is in datetime format
sales['Date'] = pd.to_datetime(sales['Date'])

# Remove dollar signs, commas, and parentheses, then convert to numeric
sales['Amount'] = sales['Amount'].replace('[\$,()]', '', regex=True).replace('', '0').astype(float)
sales['Qty. Sold'] = sales['Qty. Sold'].replace('[\$,()]', '', regex=True).replace('', '0').astype(float)

# Remove rows with non-positive 'Amount' and 'Qty. Sold'
sales = sales[(sales['Amount'] > 0) & (sales['Qty. Sold'] > 0)]

# Check the resulting DataFrame
print("Resulting DataFrame after preprocessing:")
print(sales.head())
print(f"Total rows after preprocessing: {len(sales)}")
print(f"Total unique customers: {sales['ID'].nunique()}")

#%%
# Create target variable based on each customer
sales['YearMonth'] = sales['Date'].dt.to_period('M')

def create_target(df):
    df['Purchase Next Month'] = 0
    customers = df['ID'].unique()
    
    for customer in tqdm(customers, desc="Processing customers"):
        customer_data = df[df['ID'] == customer]
        purchase_dates = customer_data['YearMonth'].unique()
        
        for date in purchase_dates:
            next_month = date + 1
            if next_month in purchase_dates:
                df.loc[(df['ID'] == customer) & (df['YearMonth'] == date), 'Purchase Next Month'] = 1
    
    return df

final_data = create_target(sales)
#%%
final_data = final_data[['ID', 'Email', 'Top Level Parent', 'Item', 'Date', 'Qty. Sold', 'Amount', 'Purchase Next Month']]
final_data.to_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv', index=False)

# Check the first few rows of final_data to verify target variable creation
print(final_data.head())
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
#%%
# Logistic Regression Model with Extended Features
file_path = '/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/purchase_level_m1.csv'
data = pd.read_csv(file_path, dtype={'ID': str})
data['Date_Num'] = pd.to_datetime(data['Date']).astype(int) // 10**9 

# Define features and target variable
features = ['Date_Num', 'Qty. Sold', 'Amount']
X = data[features]
y = data['Purchase Next Month']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
#%%
# Initialize and train the logistic regression model
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_res, y_train_res)

# Make predictions
y_pred_log = log_model.predict(X_test)

# Evaluate the model
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log, zero_division=1)
recall_log = recall_score(y_test, y_pred_log, zero_division=1)

print(f'Logistic Regression Accuracy: {accuracy_log}')
print(f'Logistic Regression Precision: {precision_log}')
print(f'Logistic Regression Recall: {recall_log}')

# Confusion Matrix
conf_matrix_log = confusion_matrix(y_test, y_pred_log)
cm_display_log = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_log, display_labels=[0, 1])
cm_display_log.plot()
plt.show()
#%%
# prediction

def predict_and_evaluate(data, log_model, year, month):
    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Get all unique customer IDs
    unique_customers = data[['ID', 'Email', 'Top Level Parent']].drop_duplicates()

    # Predicting for specified month and year
    future_data = data[data['Date'].dt.year == year]
    month_data = future_data[future_data['Date'].dt.month == month]

    # Define features for future prediction
    features = ['Date_Num', 'Qty. Sold', 'Amount']
    X_future = month_data[features]

    # Make predictions for the specified month and year
    month_predictions = log_model.predict(X_future)

    # Add predictions to the DataFrame
    month_data['Prediction'] = month_predictions

    # Evaluate predictions for the specified month and year
    actual = month_data['Purchase Next Month'].values

    conf_matrix = confusion_matrix(actual, month_predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    cm_display.plot()
    plt.title(f'Confusion Matrix for {year}-{month:02d} Predictions')
    plt.show()

    accuracy = accuracy_score(actual, month_predictions)
    precision = precision_score(actual, month_predictions, zero_division=1)
    recall = recall_score(actual, month_predictions, zero_division=1)

    print(f'{year}-{month:02d} Prediction Accuracy: {accuracy}')
    print(f'{year}-{month:02d} Prediction Precision: {precision}')
    print(f'{year}-{month:02d} Prediction Recall: {recall}')

    # Merge predictions with all unique customers to ensure all are included
    month_data = month_data[['ID', 'Prediction', 'Purchase Next Month']].drop_duplicates()
    prediction_results = pd.merge(unique_customers, month_data, on='ID', how='left')
    prediction_results['Prediction'].fillna(0, inplace=True)  # Fill NaN predictions with 0 (no purchase predicted)
    prediction_results['Purchase Next Month'].fillna(0, inplace=True)  # Fill NaN actuals with 0 (no purchase made)
    prediction_results.rename(columns={'Purchase Next Month': 'Actual'}, inplace=True)
    
    # Convert 'Prediction' and 'Actual' columns to integers (0 or 1)
    prediction_results['Prediction'] = prediction_results['Prediction'].astype(int)
    prediction_results['Actual'] = prediction_results['Actual'].astype(int)

    # Save predictions to CSV
    output_path = f'/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/output/predictions_{year}-{month:02d}.csv'
    prediction_results.to_csv(output_path, index=False)

# Example usage:
predict_and_evaluate(data, log_model, 2024, 2)


# %%
