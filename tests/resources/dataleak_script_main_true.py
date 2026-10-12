import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    X = np.genfromtxt("data.csv", delimiter=",", dtype=None)
    y = np.random.choice(2, size=10)
    X_scaled = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=2)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    accuracy = lr.score(X_test, y_test)
    print(accuracy)


if __name__ == "__main__":
    main()
