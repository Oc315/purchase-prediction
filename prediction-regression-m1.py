#%%
## Regression Model to predict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

#%%
#For Approach 1 (Logistic Regression)
#Names (Company/Customer Names) - Categorical feature
#Purchase Date - Converted to datetime and used to generate other features
#Product - Categorical feature
#Unit Volume - Numerical feature
#Unit Price - Numerical feature
#Days Since Last Purchase - Numerical feature derived from Purchase Date
#Email - may not be relevant for prediction
#Target: Binary variable indicating whether a purchase occurred within the last 180 days (1 if purchase occurred, 0 otherwise).

#%%
# Approach 1: Logistic Regression (Binary) - Predicts if a purchase will occur within a given period.
# Load your dataset
data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv')
# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Create a binary target variable: 1 if purchase in the last 180 days, 0 otherwise
data['target'] = (data['Purchase Date'] >= pd.to_datetime('today') - pd.DateOffset(days=180)).astype(int)
#%%
# Assume 'Names' and 'Product' as categorical that need encoding
data_encoded = pd.get_dummies(data, columns=['Names', 'Product', 'Email'])
#%%
# Generate features like 'days_since_last_purchase'
data_encoded['days_since_last_purchase'] = (pd.to_datetime('today') - data['Purchase Date']).dt.days
#%%
# Split the data into training and testing sets using the latest 6 months for testing
cutoff_date = pd.to_datetime('today') - pd.DateOffset(months=6)
train_data = data_encoded[data_encoded['Purchase Date'] < cutoff_date]
test_data = data_encoded[data_encoded['Purchase Date'] >= cutoff_date]
#%%
X_train = train_data.drop(columns=['target', 'Purchase Date'])
y_train = train_data['target']
#%%
X_test = test_data.drop(columns=['target', 'Purchase Date'])
y_test = test_data['target']
#%%
# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
for _ in tqdm(range(1), desc="Training Model"):
    model.fit(X_train, y_train)

#%%
# Predict on the test set
predictions = model.predict(X_test)

# Prepare the final output dataframe with person, product, and predicted target
output_df = X_test.copy()
output_df['predicted_target'] = predictions
output_df['Names'] = data.loc[X_test.index, 'Names']
output_df['Product'] = data.loc[X_test.index, 'Product']

# Select relevant columns to display
final_output = output_df[['Names', 'Product', 'predicted_target']]
print(final_output)

# Print out the classification report
print(classification_report(y_test, predictions))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

#%%
# Approach 2: Logistic Regression (Multi-Class) - Predicts the product a customer is most likely to purchase.

# Load your dataset
data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv')

# Convert 'Purchase Date' to datetime
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Split the data into training and testing sets using the latest 6 months for testing
cutoff_date = pd.to_datetime('today') - pd.DateOffset(months=6)
train_data = data[data['Purchase Date'] < cutoff_date]
test_data = data[data['Purchase Date'] >= cutoff_date]

# Assuming 'Product' is the target variable and 'Names' is another column in the dataset
X_train = train_data.drop(['Product'], axis=1)
y_train = train_data['Product']
X_test = test_data.drop(['Product'], axis=1)
y_test = test_data['Product']

# Stratify by 'Names' to ensure they appear in both training and test sets
# This requires that 'Names' be dropped after this step before training the model
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=train_data['Names'], random_state=42)

# Saving 'Names' information from X_test to display in final output
names_test = X_test['Names']

# Now drop 'Names' if it's not used as a feature in the model
X_train = X_train.drop(['Names'], axis=1)
X_test = X_test.drop(['Names'], axis=1)

# Train the Logistic Regression model 
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
for _ in tqdm(range(1), desc="Training Model"):
    model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Prepare the final output dataframe with Names and predicted product
final_output = pd.DataFrame({
    'Names': names_test,
    'predicted_product': predictions
})

# Display the predictions along with the person names
print(final_output)

#%%
# Print the classification report
print(classification_report(y_test, predictions))


#%%
# Approach 3: Random Forest
# Load your data
data = pd.read_csv('/Users/oceanuszhang/Documents/GitHub/purchase-prediction/sales-data/SalesData_Model1.csv')

# Convert 'Purchase Date' to datetime format
data['Purchase Date'] = pd.to_datetime(data['Purchase Date'])

# Additional feature engineering for temporal features
data['year'] = data['Purchase Date'].dt.year
data['month'] = data['Purchase Date'].dt.month
data['day'] = data['Purchase Date'].dt.day
data['dayofweek'] = data['Purchase Date'].dt.dayofweek

# Sorting data to facilitate repurchase identification
data.sort_values(by=['Names', 'Product', 'Purchase Date'], inplace=True)
data['previous_purchase_date'] = data.groupby(['Names', 'Product'])['Purchase Date'].shift(1)
data['is_repurchase'] = data['previous_purchase_date'].notna()

# Prepare features and target for modeling
features = data[['Unit Volume', 'Unit Price', 'month', 'day', 'dayofweek']] # Add other relevant features
target = data['is_repurchase']

# Split the data into training and testing sets using the latest 6 months for testing
cutoff_date = pd.to_datetime('today') - pd.DateOffset(months=6)
train_data = data[data['Purchase Date'] < cutoff_date]
test_data = data[data['Purchase Date'] >= cutoff_date]

X_train = train_data[features.columns]
y_train = train_data['is_repurchase']
X_test = test_data[features.columns]
y_test = test_data['is_repurchase']

# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
for _ in tqdm(range(1), desc="Training Model"):
    model.fit(X_train, y_train)

# Predicting and generating probabilities
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1] # Probability of repurchase

# Compiling results
results = pd.DataFrame({
    'Names': test_data['Names'],
    'Product': test_data['Product'],
    'repurchase_probability': probabilities
})

# Sorting results by the probability of repurchase
results_sorted = results.sort_values(by='repurchase_probability', ascending=False)
print(results_sorted.head(20))

