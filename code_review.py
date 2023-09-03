from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read file

#Read the CSV file (using relative path here)
file_path = 'CGGA_325_Read.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data for inspection
print("Initial data:")
print(data.head())

# Display and output the missing values before preprocessing
print("Missing values before preprocessing:")
print(data.isnull().sum())

# # Display rows with missing values before preprocessing
# print("\nRows with missing values before preprocessing:")
# print(data[data.isnull().any(axis=1)])

# Data preprocessing

# Fill missing values in features with the mode
for column in data.columns:
    if column != 'KLRB1':  
        mode_value = data[column].mode()[0]
        data[column].fillna(mode_value, inplace=True)

# Delete rows where the label (KLRB1) is missing
data.dropna(subset=['KLRB1'], inplace=True)

# Check and display the missing values in the processed data
print("\nMissing values after preprocessing:")
print(data.isnull().sum())

# Show the rows with missing values after processing
print("\nRows with missing values after preprocessing:")
print(data[data.isnull().any(axis=1)])

# Separate features and labels
X = data.drop(columns=['KLRB1'])
y = data['KLRB1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the split data
print("\nProcessed data:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Normalize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the normalized training data
model.fit(X_train, y_train)

# Make predictions using the normalized testing data
y_pred = model.predict(X_test)

# Evaluation

# Evaluate the model performance, using Mean Squared Error as the evaluation metric
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# Visualization

# Sort by y_test, and also sort y_pred accordingly
sorted_indices = np.argsort(y_test)
y_test_array = y_test.to_numpy()
y_test_sorted = y_test_array[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]  

# Create a plot to visualize the true labels and predicted values
plt.figure(figsize=(10, 6))

# Create indices for the x-axis
x = np.arange(len(y_test_sorted))

# Plot the sorted true labels
plt.scatter(x, y_test_sorted, label='True Labels', c='blue')

# Plot the sorted predicted values
plt.scatter(x, y_pred_sorted, label='Predictions', c='red')

# Add titles and labels
plt.title('True Labels vs Predictions (Sorted by True Labels)')
plt.xlabel('Index')
plt.ylabel('Gene Expression Level')
plt.legend()

# Show the plot
plt.show()
