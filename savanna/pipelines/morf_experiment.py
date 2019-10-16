from rerf.rerfClassifier import rerfClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import numpy as np

HEIGHT = 28
WIDTH = 28
n_estimators = 50

def process_data(X, y):
	random_state = check_random_state(0)
	permutation = random_state.permutation(X.shape[0])
	X = X[permutation]
	y = y[permutation]
	X = X.reshape((X.shape[0], -1))
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, train_size=5000, test_size=5000) # train size is 60k
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	return X_train, X_test, y_train, y_test

def test(clf, X_train, X_test, y_train, y_test):
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def avg_probs(clf1, clf2, clf3, X_train, X_test, y_train, y_test):
	clf1.fit(X_train, y_train)
	y_pred1 = clf1.predict(X_test)
	probs1 = clf1.predict_proba(X_test)
	print("Accuracy A:", metrics.accuracy_score(y_test, y_pred1))
	clf2.fit(X_train, y_train)
	y_pred2 = clf2.predict(X_test)
	probs2 = clf2.predict_proba(X_test)
	print("Accuracy B:", metrics.accuracy_score(y_test, y_pred2))
	clf3.fit(X_train, y_train)
	y_pred3 = clf3.predict(X_test)
	probs3 = clf3.predict_proba(X_test)
	print("Accuracy C:", metrics.accuracy_score(y_test, y_pred3))
	return np.mean((probs1, probs2, probs3), axis = 0).argmax(axis = 1)

naive_RF = rerfClassifier(n_estimators=n_estimators, projection_matrix = 'Base')
RerF = rerfClassifier(n_estimators=n_estimators)
MORF = rerfClassifier(n_estimators=n_estimators, projection_matrix = 'S-RerF', 
	image_height = HEIGHT, image_width = WIDTH)

MORF_2 = rerfClassifier(n_estimators=n_estimators, projection_matrix = 'S-RerF', 
	image_height = HEIGHT, image_width = WIDTH, patch_height_min = 2, patch_height_max = 2,
	patch_width_min = 2, patch_width_max = 2)
MORF_3 = rerfClassifier(n_estimators=n_estimators, projection_matrix = 'S-RerF', 
	image_height = HEIGHT, image_width = WIDTH, patch_height_min = 3, patch_height_max = 3,
	patch_width_min = 3, patch_width_max = 3)
MORF_4 = rerfClassifier(n_estimators=n_estimators, projection_matrix = 'S-RerF', 
	image_height = HEIGHT, image_width = WIDTH, patch_height_min = 4, patch_height_max = 4,
	patch_width_min = 4, patch_width_max = 4)

# RF w/ same # trees as aggregate
naive_RF_300 = rerfClassifier(n_estimators=n_estimators*3)
RerF_300 = rerfClassifier(n_estimators=n_estimators*3, projection_matrix = 'Base')
MORF_300 = rerfClassifier(n_estimators=n_estimators*3, projection_matrix = 'S-RerF', 
	image_height = HEIGHT, image_width = WIDTH)

# input is probabilities
# multinomial logistic regression
logistic = LogisticRegression(multi_class='multinomial',
                         penalty='l1', solver='saga', tol=0.1)

print("MNIST")
# load MNIST
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X_train, X_test, y_train, y_test = process_data(X, y)
print("Naive RF")
test(naive_RF, X_train, X_test, y_train, y_test)
print("ReRF")
test(RerF, X_train, X_test, y_train, y_test)
print("MORF")
test(MORF, X_train, X_test, y_train, y_test)
print("MORF Patch Size 2")
#test(MORF_2, X_train, X_test, y_train, y_test)
print("MORF Patch Size 4")
#test(MORF_4, X_train, X_test, y_train, y_test)
print("MORF Patch Size 3")
#test(MORF_3, X_train, X_test, y_train, y_test)
print("MORF Aggregate")
probs = avg_probs(MORF_2, MORF_3, MORF_4, X_train, X_test, y_train, y_test)
print("Accuracy:", metrics.accuracy_score(y_test, probs))
print("Naive RF 150 trees")
test(naive_RF_300, X_train, X_test, y_train, y_test)
print("ReRF 150 trees")
test(RerF_300, X_train, X_test, y_train, y_test)
print("MORF 150 trees")
test(MORF_300, X_train, X_test, y_train, y_test)

# Run on Probabilities
MORF_300.fit(X_train, y_train)
y_pred = MORF_300.predict_proba(X_test)
train_size = 4000
print("Logistic on Probabilities")
test(logistic, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])
print("Naive RF on Probabilities")
test(naive_RF, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])
print("RerF on Probabilities")
test(RerF, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])

print("FashionMNIST")
# load fashion MNIST
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
y = y.astype(int)
X_train, X_test, y_train, y_test = process_data(X, y)
print("Naive RF")
test(naive_RF, X_train, X_test, y_train, y_test)
print("ReRF")
test(RerF, X_train, X_test, y_train, y_test)
print("MORF")
test(MORF, X_train, X_test, y_train, y_test)
print("MORF Patch Size 2")
#test(MORF_2, X_train, X_test, y_train, y_test)
print("MORF Patch Size 4")
#test(MORF_4, X_train, X_test, y_train, y_test)
print("MORF Patch Size 3")
#test(MORF_3, X_train, X_test, y_train, y_test)
print("MORF Aggregate")
probs = avg_probs(MORF_2, MORF_3, MORF_4, X_train, X_test, y_train, y_test)
print("Accuracy:", metrics.accuracy_score(y_test, probs))
print("Naive RF 150 trees")
test(naive_RF_300, X_train, X_test, y_train, y_test)
print("ReRF 150 trees")
test(RerF_300, X_train, X_test, y_train, y_test)
print("MORF 150 trees")
test(MORF_300, X_train, X_test, y_train, y_test)

# Run on Probabilities
MORF_300.fit(X_train, y_train)
y_pred = MORF_300.predict_proba(X_test)
train_size = 4000
print("Logistic on Probabilities")
test(logistic, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])
print("Naive RF on Probabilities")
test(naive_RF, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])
print("RerF on Probabilities")
test(RerF, y_pred[:train_size], y_pred[train_size:], y_test[:train_size], y_test[train_size:])
