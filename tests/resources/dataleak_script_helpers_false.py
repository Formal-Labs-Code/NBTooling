import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_data(path, delimiter=","):
    if path.endswith(".npy"):
        return np.load(path)
    return np.genfromtxt(path, delimiter=delimiter, dtype=None)


def preprocess(X, scale=True):
    if not scale:
        return X
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def evaluate(model, X, y):
    print("accuracy:", model.score(X, y))


def main():
    X = load_data("data.csv")
    y = np.random.choice(2, size=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)  # fit separately per split: no leak into training
    model = LogisticRegression()
    model.fit(X_train, y_train)
    evaluate(model, X=X_test, y=y_test)


if __name__ == "__main__":
    main()
