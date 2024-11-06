# Import libraries that we'll use.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file.
data = pd.read_csv('USDINRX.csv')

# Display first few rows of the data
print(data.head())

# Convert 'Date' column to datetime format and set it as index.
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Opening price chart
plt.figure(figsize=(14, 7))
plt.plot(data['Open'], label='Open Price')
plt.title('Historical Open Price of USD')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Descriptive statistics
print(data.describe())

# Data Preprocessing
# Remove the 'Volume' column which has no useful information
data.drop(columns=['Volume'], inplace=True)

# Check for missing values
missing_data = data.isnull().sum()
print(f"Missing data per column:\n{missing_data}")

# Remove rows with missing values
data_clean = data.dropna()

# Prepare data for splitting
X_clean = data_clean[['High', 'Low', 'Close', 'Adj Close']]
y_clean = data_clean['Open']

# Split clean data into training, validation, and test sets (70%, 15%, 15%)
X_train_clean, X_temp_clean, y_train_clean, y_temp_clean = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42)
X_val_clean, X_test_clean, y_val_clean, y_test_clean = train_test_split(
    X_temp_clean, y_temp_clean, test_size=0.5, random_state=42)

# Verify the new dimensions of the assemblies
print(f"Training set shape: {X_train_clean.shape}")
print(f"Validation set shape: {X_val_clean.shape}")
print(f"Test set shape: {X_test_clean.shape}")

# Training the models
model1 = LinearRegression()
model2 = Ridge(alpha=1.0)
model3 = Lasso(alpha=0.01, max_iter=10000)

model1.fit(X_train_clean, y_train_clean)
model2.fit(X_train_clean, y_train_clean)
model3.fit(X_train_clean, y_train_clean)

# Evaluate the models
def evaluate(model, X, y):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mae, r2

mae1, r2_1 = evaluate(model1, X_val_clean, y_val_clean)
mae2, r2_2 = evaluate(model2, X_val_clean, y_val_clean)
mae3, r2_3 = evaluate(model3, X_val_clean, y_val_clean)

# Comparing model performance
performance_data = {
    'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
    'MAE': [mae1, mae2, mae3],
    'R2 Score': [r2_1, r2_2, r2_3]
}
performance_table = pd.DataFrame(performance_data)
print("\nModel Performance Comparison:")
print(performance_table)

# Final evaluation on test data (if needed)
mae1_test, r2_1_test = evaluate(model1, X_test_clean, y_test_clean)
mae2_test, r2_2_test = evaluate(model2, X_test_clean, y_test_clean)
mae3_test, r2_3_test = evaluate(model3, X_test_clean, y_test_clean)

print("\nFinal Model Evaluation on Test Set:")
print(f"Linear Regression - MAE: {mae1_test}, R2: {r2_1_test}")
print(f"Ridge Regression - MAE: {mae2_test}, R2: {r2_2_test}")
print(f"Lasso Regression - MAE: {mae3_test}, R2: {r2_3_test}")