#Author : Deepansh Dubey.
#Date   : 12/10/2021.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Fetching Dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
x,y = mnist['data'], mnist['target']
x.info

#Data extraction
some_digit = np.array(x.iloc[36001])
some_digit_image = np.reshape(some_digit, (28, 28))

#plotting
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis('off')

#Train-Test Set Slicing
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]
shuffled_index = np.random.permutation(60000)
x_train = x_train.sample(frac=1).reset_index(drop=True)

#Creating a 3 detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_3 = (y_train==3)
y_test_3 = (y_test==3)

#model training
clf = LogisticRegression(tol = 0.1)
clf.fit(x_train, y_train_3)
clf.predict([some_digit])
a=cross_val_score(clf, x_train, y_train_3, cv=3, scoring='accuracy')
print(a.mean())