
import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
warnings.filterwarnings("ignore", category=DeprecationWarning)

data1 = pd.read_csv('./Data/supermarket_sales.csv')
data = data1[['Date', 'Branch', 'Total']]


df_n02ba = data[['Date','Total']]
df_n02ba

#from sklearn.preprocessing import LabelEncoder

#label = LabelEncoder()

#df_n02ba['Branch'] = label.fit_transform(data['Branch'])
# Reshape the dataframe
df_new3 = df_n02ba.melt(id_vars=['Date'],
             var_name='Total',
             value_name='Sales')

# Print the updated dataframe
df_new3

df_new3['Total'] = le.fit_transform(df_new3['Total']) + 200

# Display the modified DataFrame
print(df_new3)

# Create a new index column
df_new3['Index'] = range(len(df_new3))

# Set 'Index' as the index
df_new3.set_index('Index', inplace=True)

df_new3


# Function to prepare data
def prepare_data(time_data, n_features):
    X, y = [], []
    for i in range(len(time_data)):
        end_ix = i + n_features
        if end_ix > (len(time_data) - 1):
            break
        seq_x, seq_y = time_data[i:end_ix], time_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

split_index = 500
train3 = df_new3.iloc[:split_index]
test3 = df_new3.iloc[split_index:]
X_train3 = train3.drop(['Date'], axis=1).values
y_train3 = train3['Sales'].values
X_test3 = test3.drop(['Date'], axis=1).values
y_test3 = test3['Sales'].values

# Standardize features
scaler = StandardScaler()
X_train3 = scaler.fit_transform(X_train3)
X_test3 = scaler.transform(X_test3)

# Normalize target variable
scaler_y = StandardScaler()
y_train3 = scaler_y.fit_transform(y_train3.reshape(-1, 1)).flatten()
y_test3 = scaler_y.transform(y_test3.reshape(-1, 1)).flatten()

# Split the data
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train3, y_train3, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
gb = RandomForestRegressor(random_state=0)
gb.fit(X_train3, y_train3)


filename = 'model.pkl'
pickle.dump(gb, open(filename, 'wb'))


with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

train_loss2 = gb.score(X_train3, y_train3)
val_loss2 = gb.score(X_val3, y_val3)
print('Train Loss:', train_loss2)
print('Validation Loss:', val_loss2)





train_pred = gb.predict(X_train3)
val_pred = gb.predict(X_val3)

# Calculate MSE, RMSE, R², and MAE for training data
train_mse = mean_squared_error(y_train3, train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train3, train_pred)
train_mae = mean_absolute_error(y_train3, train_pred)

# Calculate MSE, RMSE, R², and MAE for validation data
val_mse = mean_squared_error(y_val3, val_pred)
val_rmse = np.sqrt(val_mse)
val_r2 = r2_score(y_val3, val_pred)
val_mae = mean_absolute_error(y_val3, val_pred)

# Print the results
print("Training Metrics:")
print('MSE (Train):', train_mse)
print('RMSE (Train):', train_rmse)
print('R² (Train):', train_r2)
print('MAE (Train):', train_mae)

print("\nValidation Metrics:")
print('MSE (Validation):', val_mse)
print('RMSE (Validation):', val_rmse)
print('R² (Validation):', val_r2)
print('MAE (Validation):', val_mae)


predictions = gb.predict(X_test3)
predicted_values = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_values = scaler_y.inverse_transform(y_test3.reshape(-1, 1)).flatten()

val_predictions = pd.DataFrame({
    'Actual': y_val3,  # Actual values from validation data
    'Predicted': val_pred  # Predicted values for validation data
})

# Save the DataFrames to CSV files
#
val_predictions.to_csv('val_predictions.csv', index=False)


# Select the first 50 records for plotting
predicted_values_50 = predicted_values[:50]
actual_values_50 = actual_values[:50]
dates_50 = test3['Date'][:50]
print(dates_50)

print(actual_values_50)


print("hai")
print(predicted_values_50)

plt.figure(figsize=(10, 6))
#plt.scatter(y_test, predicted_totals, alpha=0.7, color='blue')
plt.plot(dates_50, actual_values_50, marker='o', markersize=6,color='Red')
plt.plot(dates_50, predicted_values_50, marker='x', markersize=6,color='green')
plt.title('Actual vs Predicted Totals (First 50 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Total Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

data = np.array([[3, 2019]])
scaled_data = scaler.transform(data)
my_prediction = classifier.predict(data)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(my_prediction[0])


'''data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

data = data.sort_values(by='Date').reset_index(drop=True)
print(data)
print(data.head())

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

data['Branch'] = label.fit_transform(data['Branch'])

data['year'] = pd.DatetimeIndex(data['Date']).year
data['month'] = pd.DatetimeIndex(data['Date']).month
data['day'] = pd.DatetimeIndex(data['Date']).day
data.drop('Date', axis=1, inplace=True)

data['Total'].fillna(data['Total'].median(), inplace=True)

X = data.drop('Total', axis=1)
Y = data['Total']

print(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np

regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, random_state=0)

regressor.fit(X_train, np.ravel(y_train, order='C'))

predicted_totals = regressor.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predicted_totals)
mae = mean_absolute_error(y_test, predicted_totals)
rmse = np.sqrt(mse)

# Display the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
accuracy = regressor.score(X_test, y_test)
print(f"Accuracy (R^2 Score): {accuracy}")

# Creating a pickle file for the classifier
filename = 'Model/prediction-rfc-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': predicted_totals
})
print(comparison.head())

import matplotlib.pyplot as plt

# Scatter plot for Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicted_totals, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs Predicted Totals')
plt.xlabel('Actual Totals')
plt.ylabel('Predicted Totals')
plt.grid()
plt.show()

# Line plot for Actual vs Predicted (limited to first 50 for clarity)
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': predicted_totals})
comparison = comparison.head(50)

plt.figure(figsize=(12, 6))
plt.plot(comparison['Actual'], label='Actual', marker='o')
plt.plot(comparison['Predicted'], label='Predicted', marker='x')
plt.title('Actual vs Predicted Totals (First 50 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Total Sales')
plt.legend()
plt.grid()
plt.show()

filename = 'Model/prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

data = np.array([[1, 2019, 1, 1]])
my_prediction = classifier.predict(data)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(my_prediction[0])'''
