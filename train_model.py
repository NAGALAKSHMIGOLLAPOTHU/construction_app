import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/material_quantity_real.csv")

df["Building_Type"] = df.get("Building_Type", "Residential")
df["Construction_Type"] = df.get("Construction_Type", "RCC")

X = df[["Area_sqft", "Floors", "Building_Type", "Construction_Type"]]

targets = {
    "Cement_Bags": "cement_model.pkl",
    "Sand_CFT": "sand_model.pkl",
    "Bricks_Count": "brick_model.pkl",
    "Steel_Kg": "steel_model.pkl"
}

os.makedirs("models", exist_ok=True)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["Building_Type", "Construction_Type"]),
    ("num", "passthrough", ["Area_sqft", "Floors"])
])

for target, filename in targets.items():
    y = df[target]

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    model.fit(X, y)
    pickle.dump(model, open(f"models/{filename}", "wb"))

print("Models trained successfully")
