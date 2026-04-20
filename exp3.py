# Experiment 3: Random Forest Algorithm

# -----------------------------------
# 1. Import Libraries
# -----------------------------------
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
# 4. Decision Tree (Base Model)
# -----------------------------------
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# -----------------------------------
# 5. Random Forest Model
# -----------------------------------
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -----------------------------------
# 6. Results
# -----------------------------------
print("\nExperiment 3: Random Forest Results\n - exp3.py:47")
print("Decision Tree Accuracy: - exp3.py:48", dt_acc)
print("Random Forest Accuracy: - exp3.py:49", rf_acc)

# -----------------------------------
# 7. Confusion Matrix
# -----------------------------------
print("\nConfusion Matrix (Random Forest):\n - exp3.py:54")
print(confusion_matrix(y_test, rf_pred))

# -----------------------------------
# 8. Visualization
# -----------------------------------
models = ['Decision Tree', 'Random Forest']
scores = [dt_acc, rf_acc]

plt.bar(models, scores)
plt.title("Random Forest vs Decision Tree")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()