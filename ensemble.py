# -----------------------------------
# 1. Import Libraries
# -----------------------------------
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# -----------------------------------
# 2. Load Dataset
# -----------------------------------
data = load_breast_cancer()
X = data.data
y = data.target


# -----------------------------------
# 3. Split Dataset
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------------
# 4. Train Base Model
# -----------------------------------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))


# -----------------------------------
# 5. Bagging (Random Forest)
# -----------------------------------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))


# -----------------------------------
# 6. Boosting Models
# -----------------------------------

# AdaBoost
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))


# -----------------------------------
# 7. Voting Ensemble
# -----------------------------------
lr = LogisticRegression(max_iter=2000)
svc = SVC(probability=True)

voting = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('svc', svc)],
    voting='soft'
)

voting.fit(X_train, y_train)
voting_acc = accuracy_score(y_test, voting.predict(X_test))


# -----------------------------------
# 8. Print Results
# -----------------------------------
print("\nModel Accuracy Results:\n - ensemble.py:82")
print("Decision Tree: - ensemble.py:83", dt_acc)
print("Random Forest: - ensemble.py:84", rf_acc)
print("AdaBoost: - ensemble.py:85", ada_acc)
print("Gradient Boosting: - ensemble.py:86", gb_acc)
print("Voting Classifier: - ensemble.py:87", voting_acc)


# -----------------------------------
# 9. Visualization
# -----------------------------------
models = ['DT', 'RF', 'Ada', 'GB', 'Vote']
scores = [dt_acc, rf_acc, ada_acc, gb_acc, voting_acc]

plt.bar(models, scores)
plt.title("Ensemble Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()