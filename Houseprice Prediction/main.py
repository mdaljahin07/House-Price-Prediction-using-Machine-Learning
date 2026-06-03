import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error

ohe = OneHotEncoder(sparse_output=False)
lr = LinearRegression()

dataset = pd.read_csv("housePrice.csv")

dataset["Area"] = dataset["Area"].astype(str).str.replace(",","")
dataset["Area"] = pd.to_numeric(dataset["Area"])

encode = ohe.fit_transform(dataset[["Address"]])
encode_df = pd.DataFrame(encode, columns=ohe.get_feature_names_out(["Address"]))
dataset = pd.concat([dataset.drop("Address", axis=1), encode_df], axis=1)

X = dataset.drop(["Price(USD)"], axis=1)
Y = dataset["Price(USD)"]

x_train, x_test, y_trian, y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

lr.fit(x_train, y_trian)

lr.score(x_test, y_test)

y_pred = lr.predict(x_test)

print("R2: ", r2_score(y_test, y_pred))
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", root_mean_squared_error(y_test, y_pred))

ar = int(input("Area : "))
ro = int(input("Room : "))
pr = bool(input("Parking : "))
wh = bool(input("Warehouse: "))
el = bool(input("Elevator: "))
pr = int(input("Price: "))
ad = input("Address : ")

input_dict = {"Area" : ar,
              "Room" : ro,
              "Parking" : pr,
              "Warehouse" : wh,
              "Elevator" : el,
              "Price" : pr,
              f"Address_{ad}" : 1}


input_df = pd.DataFrame(columns=X.columns)

input_df.loc[0] = 0

for col, val in input_dict.items():
    input_df[col] = val

print("The house price in USD is:", lr.predict(input_df))