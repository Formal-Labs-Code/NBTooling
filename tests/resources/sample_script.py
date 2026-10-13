import pandas as pd

df = pd.read_csv("data.csv")
X = df.drop(columns=["y"])
y = df["y"]
model = fit_model(X, y)
