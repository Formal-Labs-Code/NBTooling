#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% [markdown]
# Same notebook as dataleak_true.ipynb, in VS Code / Jupytext percent format.

# %%
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# random data:
X = np.genfromtxt("data.csv", delimiter=',', dtype=None)
y = np.random.choice(3, size=4)

# %%
min_max_scaler = MinMaxScaler()
# first select features
X_selected = min_max_scaler.fit_transform(X)

# %%
# then split
X_selected_train, X_selected_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.025, random_state=2)

# %%
# fit a simple logistic regression
lr = LogisticRegression()
a = lr.fit(X_selected_train, y_train)

# %%
# predict on the test set and get the test accuracy:
y_pred = lr.predict(X_selected_test)
accuracy_score(y_test, y_pred)
