# Financial Transaction Classification Project

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Create synthetic financial data
data = {
    "amount": [50, 120, 300, 1500, 70, 800, 90, 2000, 400, 60],
    "category": ["Food", "Food", "Food", "Rent", "Transport",
                 "Utilities", "Food", "Rent", "Utilities", "Transport"]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# 2. Encode the target variable
encoder = LabelEncoder()
df["category_encoded"] = encoder.fit_transform(df["category"])

# 3. Define features and target
X = df[["amount"]]
y = df["category_encoded"]

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 5. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_test)

# 7. Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("\nModel Accuracy:", accuracy)

# 8. Test with a new transaction
new_amount = [[100]]
predicted_category = model.predict(new_amount)
print(
    "Predicted category for amount 100:",
    encoder.inverse_transform(predicted_category)[0]
)
