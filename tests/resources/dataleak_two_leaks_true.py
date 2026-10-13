import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def main():
    X = np.genfromtxt("data.csv", delimiter=",", dtype=None)
    y = np.random.choice(2, size=10)

    # leak 1: impute with statistics of all rows
    X = SimpleImputer(strategy="median").fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    # leak 2: a second model scales all rows before its own split
    X_all_scaled = StandardScaler().fit_transform(X)
    A_train, A_test, b_train, b_test = train_test_split(X_all_scaled, y, test_size=0.3, random_state=3)

    first = LogisticRegression()
    first.fit(X_train, y_train)
    print("first:", first.score(X_test, y_test))

    second = LogisticRegression()
    second.fit(A_train, b_train)
    print("second:", second.score(A_test, b_test))


if __name__ == "__main__":
    main()
