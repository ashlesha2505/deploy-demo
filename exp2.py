# Experiment 2: Bagging vs Boosting

# Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Base Model (Decision Tree)
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))

# 2. Bagging (Random Forest)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# 3. Boosting (AdaBoost)
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, y_train)
ada_acc = accuracy_score(y_test, ada.predict(X_test))

# Print results
print("\nExperiment 2 Results:\n - exp2.py:38")
print("Decision Tree (Base): - exp2.py:39", dt_acc)
print("Bagging (Random Forest): - exp2.py:40", rf_acc)
print("Boosting (AdaBoost): - exp2.py:41", ada_acc)

# Visualization
models = ['Base', 'Bagging', 'Boosting']
scores = [dt_acc, rf_acc, ada_acc]

plt.bar(models, scores)
plt.title("Bagging vs Boosting Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()