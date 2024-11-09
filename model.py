import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
data = pd.read_csv('loan1.csv')

# Check for and remove any leading or trailing spaces in column names
data.columns = data.columns.str.strip()

# Print column names to verify
print("Column Names:", data.columns)

# Preprocessing
# Encode categorical variables
label_encoder = LabelEncoder()
data['education'] = label_encoder.fit_transform(data['education'])
data['self_employed'] = label_encoder.fit_transform(data['self_employed'])
data['loan_status'] = label_encoder.fit_transform(data['loan_status'])  # Approved/Rejected to 1/0

# Select features and target variable
X = data.drop(columns=['loan_id', 'loan_status'])
y = data['loan_status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler have been saved successfully.")
