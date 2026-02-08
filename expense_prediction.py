# Financial Expense Prediction Project

import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Create synthetic monthly expense data
data = {
    "month": [1, 2, 3, 4, 5, 6],
    "expenses": [4500, 4800, 5000, 5300, 5600, 5900]
}

df = pd.DataFrame(data)

print("Expense Data:")
print(df)

# 2. Prepare data
X = df[["month"]]
y = df["expenses"]

# 3. Train the model
model = LinearRegression()
model.fit(X, y)

# 4. Predict next month
future_month = [[7]]
prediction = model.predict(future_month)

print("\nPredicted expense for month 7:", prediction[0])
