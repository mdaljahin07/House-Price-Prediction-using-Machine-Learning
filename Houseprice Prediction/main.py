import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
)

# Load dataset
dataset = pd.read_csv("housePrice.csv")

# Clean Area column
dataset["Area"] = (
    dataset["Area"]
    .astype(str)
    .str.replace(",", "", regex=False)
)
dataset["Area"] = pd.to_numeric(dataset["Area"])

# One-Hot Encode Address
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

encoded = ohe.fit_transform(dataset[["Address"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(["Address"])
)

dataset = pd.concat(
    [dataset.drop("Address", axis=1), encoded_df],
    axis=1
)

# Features and Target
X = dataset.drop("Price(USD)", axis=1)
y = dataset["Price(USD)"]

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluation
y_pred = model.predict(x_test)

print("R² Score :", r2_score(y_test, y_pred))
print("MSE      :", mean_squared_error(y_test, y_pred))
print("RMSE     :", root_mean_squared_error(y_test, y_pred))

# ---------------------------
# User Prediction
# ---------------------------

area = int(input("Area: "))
room = int(input("Room: "))
parking = int(input("Parking (0/1): "))
warehouse = int(input("Warehouse (0/1): "))
elevator = int(input("Elevator (0/1): "))
address = input("Address: ")

# Create empty row
input_df = pd.DataFrame(0, index=[0], columns=X.columns)

# Fill values
input_df["Area"] = area
input_df["Room"] = room
input_df["Parking"] = parking
input_df["Warehouse"] = warehouse
input_df["Elevator"] = elevator

address_col = f"Address_{address}"

if address_col in input_df.columns:
    input_df[address_col] = 1

prediction = model.predict(input_df)

print(f"\nPredicted House Price: ${prediction[0]:,.2f}")