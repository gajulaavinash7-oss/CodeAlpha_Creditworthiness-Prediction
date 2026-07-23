import pandas as pd

# Create Dataset
data = {
    "Income": [50000, 30000, 70000, 25000, 60000],
    "Debt": [10000, 25000, 5000, 30000, 7000],
    "PaymentHistory": [1, 0, 1, 0, 1],
    "Creditworthy": [1, 0, 1, 0, 1]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

print("Dataset")
print(df)

# Features (Input)
X = df[["Income", "Debt", "PaymentHistory"]]

# Target (Output)
y = df["Creditworthy"]

# Split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data")
print(X_train)

print("\nTesting Data")
print(X_test)

# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# Create Model
model = RandomForestClassifier()

# Train Model
model.fit(X_train, y_train)

# Predict
prediction = model.predict(X_test)

print("\nPrediction:", prediction)

print("Actual:", y_test.values)

# Check Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy:", accuracy)

# Predict for a New Customer
new_customer = pd.DataFrame({
    "Income": [65000],
    "Debt": [8000],
    "PaymentHistory": [1]
})

new_prediction = model.predict(new_customer)

print("\nNew Customer Prediction:", new_prediction)

if new_prediction[0] == 1:
    print("Loan Approved (Creditworthy)")
else:
    print("Loan Rejected (Not Creditworthy)")