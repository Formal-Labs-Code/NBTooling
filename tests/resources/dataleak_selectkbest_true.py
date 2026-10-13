import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

X = np.genfromtxt("data.csv", delimiter=",", dtype=None)
y = np.random.choice(2, size=10)

# feature selection looks at the labels of ALL rows before the split - the leak
selector = SelectKBest(f_classif, k=2)
X_selected = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, random_state=2)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("accuracy:", lr.score(X_test, y_test))
