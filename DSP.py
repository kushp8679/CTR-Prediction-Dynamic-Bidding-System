import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load CSV
df = pd.read_csv("Untitled - Sheet 1 (3).csv")

# Encode categorical columns
categorical_cols = ["domain", "size", "device_type", "os", "city"]
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Features (X) and label (y)
X = df.drop("clicked", axis=1)
y = df["clicked"]

# Train simple logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict CTR for a new request
new_request = pd.DataFrame([{
    "domain": encoders["domain"].transform(["news.com"])[0],
    "size": encoders["size"].transform(["300x250"])[0],
    "device_type": encoders["device_type"].transform(["mobile"])[0],
    "os": encoders["os"].transform(["Android"])[0],
    "city": encoders["city"].transform(["Delhi"])[0],
    "hour": 15
}])

pCTR = model.predict_proba(new_request)[:,1][0]

# Dynamic bid calculation
base_ecpm = 100
performance_modifier = 1 + (pCTR - 0.05)
pacing_modifier = 1.0

dynamic_bid = base_ecpm * performance_modifier * pacing_modifier
print("Predicted CTR:", round(pCTR, 4))
print("Final Bid:", round(dynamic_bid, 2))
