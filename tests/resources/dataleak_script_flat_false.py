import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = np.genfromtxt("data.csv", delimiter=",", dtype=None)
y = np.random.choice(2, size=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

# scaler fit on the training split only
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
accuracy = lr.score(X_test_scaled, y_test)
print(accuracy)
