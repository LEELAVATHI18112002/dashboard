import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.graph_objs as go
import plotly.express as px

plt.style.use('seaborn-dark')
plt.style.context('grayscale')

from wordcloud import WordCloud, STOPWORDS

df = pd.read_csv("Data/supermarket_sales.csv")

print(df.head())

print(df.describe())

print(df.dtypes)

print('Number of rows are :', df.shape[0], ',and number of columns are :', df.shape[1])

df.rename(columns={"Customer type": "Customer_type", "Product line": "Product_line",
                   "Unit price": "Unit_price", "Tax 5%": "Tax_5", "gross margin percentage": "gross_margin_percentage",
                   "gross income": "gross_income"}, inplace="True")

msno.bar(df, figsize=(16, 5), color="magenta")
plt.show()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df['day'] = (df['Date']).dt.day
df['month'] = (df['Date']).dt.month
df['year'] = (df['Date']).dt.year

df.loc[(df['month'] == 1), 'month'] = 'January'
df.loc[(df['month'] == 2), 'month'] = 'February'
df.loc[(df['month'] == 3), 'month'] = 'March'

df.year.value_counts()

sns.countplot(x="year", data=df, color="cyan").set_title("Customer Type")
plt.show()

df.month.value_counts()

sns.countplot(x="month", data=df, palette="autumn").set_title("Months")
plt.show()

# fig = px.pie(df, values='Total', names='month')
# fig.show()

df.day.value_counts()

plt.figure(figsize=(8, 8))
sns.countplot(x="day", data=df, palette="Spectral").set_title("Days")
plt.show()

df.City.value_counts()

sns.countplot(x="City", data=df, palette="bone").set_title("Customer Type")
plt.show()

# fig = px.pie(df, values='Total', names='City')
# fig.show()

df.Customer_type.value_counts()
sns.countplot(x="Customer_type", data=df, palette="terrain_r").set_title("Customer Type")
plt.show()

# fig = px.pie(df, values='Total', names='Customer_type')
# fig.show()

df.Gender.value_counts()

sns.countplot(x="Gender", data=df, palette="ocean").set_title("Gender")
plt.show()
# fig = px.pie(df, values='Total', names='Gender')
# fig.show()

df.Product_line.value_counts()

sns.countplot(x="Product_line", data=df, palette="Pastel1").set_title("Product Line")
plt.xticks(rotation=45)
plt.show()

# fig = px.pie(df, values='Total', names='Product_line')
# fig.show()

df.Quantity.value_counts()

sns.countplot(x="Quantity", data=df, palette="viridis").set_title("Quantity")
plt.show()

# fig = px.pie(df, values='Total', names='Quantity')
# fig.show()

df.Branch.value_counts()

sns.countplot(x="Branch", data=df, palette="afmhot_r").set_title("Branch")
plt.show()

# fig = px.pie(df, values='Total', names='Branch')
# fig.show()

df.Payment.value_counts()
sns.countplot(x="Payment", data=df, palette="PuOr").set_title("Payment Channel")
plt.show()

# fig = px.pie(df, values='Total', names='Payment')
# fig.show()

sns.boxplot(x='Payment', y='Total', data=df, palette="PuOr", showmeans=True)
plt.show()

fig = plt.figure(figsize=(15, 4))
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0)
fig.text(0.120, 1.1, 'Payment Method ', fontfamily='serif', fontsize=15, fontweight='bold')
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

sns.countplot(x='Gender',
              data=df,
              palette='Blues',
              ax=ax0)
ax0.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
ax0.set_title('Distribution', fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1)
ax0.set_xticklabels(["Male", "Female"])
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.set_xlabel("")
ax0.set_ylabel("")

sns.countplot(x='Customer_type',
              data=df,
              hue='Payment',
              ax=ax1)
ax1.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
ax1.set_title('Distribution by Member or Normal', fontsize=12, fontfamily='serif', fontweight='bold', x=0.25, y=1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_legend().remove()
legend_labels, _ = ax1.get_legend_handles_labels()
ax1.legend(legend_labels, ['Ewallet', 'Cash', "Credit card"], ncol=3, bbox_to_anchor=(-0.30, 1.22))
ax1.set_xticklabels(["Member", "Normal"])
ax1.set_xlabel("")
ax1.set_ylabel("")

plt.show()

plt.subplots(figsize=(20, 8))
wordcloud = WordCloud(background_color='pink', width=1920, height=1080).generate(" ".join(df['Product_line']))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x='Rating', y='Unit_price', data=df)
plt.show()

plt.style.use("default")
plt.figure(figsize=(5, 5))
sns.barplot(x="Quantity", y="Unit_price", data=df)
plt.title("Rating vs Unit Price", fontsize=15)
plt.xlabel("Rating")
plt.ylabel("Unit Price")
plt.show()

# let's visualize gender, product line vs total first
'''from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/supermarket_sales.csv")
# create a new DataFrame with only the relevant columns
df_regression = df[
    ['Date', 'Total']]

# apply one-hot encoding to the categorical columns
df_regression = pd.get_dummies(df_regression)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_regression.drop('Total', axis=1), df_regression['Total'],
                                                    test_size=0.2)

# initialize the Random Forest model
rf = RandomForestRegressor()

# fit the model to the training data
rf.fit(X_train, y_train)

# predict the total sales for the test data
predicted_totals = rf.predict(X_test)

# calculate the accuracy of the predictions
accuracy = rf.score(X_test, y_test)
print('Accuracy:', accuracy)'''


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Data/supermarket_sales.csv")

# Create a new DataFrame with only the relevant columns
df_regression = df[['Date','Branch' ,'Total']]

# Apply one-hot encoding to the categorical columns (assuming 'Date' is categorical or needs encoding)
df_regression = pd.get_dummies(df_regression)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df_regression.drop('Total', axis=1),
    df_regression['Total'],
    test_size=0.2,
    random_state=42
)

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state=42)

# Fit the model to the training data
rf.fit(X_train, y_train)

# Predict the total sales for the test data
predicted_totals = rf.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predicted_totals)
mae = mean_absolute_error(y_test, predicted_totals)
rmse = np.sqrt(mse)

# Display the metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Calculate and print model accuracy (R^2 Score)
accuracy = rf.score(X_test, y_test)
print(f"Accuracy (R^2 Score): {accuracy}")

# Display actual vs predicted values
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



import pandas as pd
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the dataset
data1 = pd.read_csv('./Data/supermarket_sales.csv')
data = data1[['Date', 'Branch', 'Total']]

# Prepare the data (subset relevant columns and reshape)
df_n02ba = data[['Date', 'Total']]
df_new3 = df_n02ba.melt(id_vars=['Date'], var_name='Total', value_name='Sales')

# Label encode the 'Total' values
le = LabelEncoder()
df_new3['Total'] = le.fit_transform(df_new3['Total']) + 200

# Set an index for the dataframe
df_new3['Index'] = range(len(df_new3))
df_new3.set_index('Index', inplace=True)

# Split the data into training and testing sets
split_index = 500
train3 = df_new3.iloc[:split_index]
test3 = df_new3.iloc[split_index:]

X_train3 = train3.drop(['Date'], axis=1).values
y_train3 = train3['Sales'].values
X_test3 = test3.drop(['Date'], axis=1).values
y_test3 = test3['Sales'].values

# Standardize the features
scaler = StandardScaler()
X_train3 = scaler.fit_transform(X_train3)
X_test3 = scaler.transform(X_test3)

# Normalize the target variable
scaler_y = StandardScaler()
y_train3 = scaler_y.fit_transform(y_train3.reshape(-1, 1)).flatten()
y_test3 = scaler_y.transform(y_test3.reshape(-1, 1)).flatten()

# Train a Random Forest model
gb = RandomForestRegressor(random_state=0)
gb.fit(X_train3, y_train3)

# Make predictions
predictions = gb.predict(X_test3)

# Inverse transform the predictions to original scale
predicted_values = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()


# Save the model and scaler using pickle
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(gb, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('scaler_y.pkl', 'wb') as scaler_y_file:
    pickle.dump(scaler_y, scaler_y_file)

# Now you have saved the model and scalers
print("Model and scalers have been saved.")


