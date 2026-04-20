# Experiment 4: Bayesian Learning (with Graph)

# -----------------------------------
# 1. Import Libraries
# -----------------------------------
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("\n Bayesian Learning \n - exp4bayes.py:12")

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
# 4. Train Model
# -----------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -----------------------------------
# 5. Prediction
# -----------------------------------
y_pred = model.predict(X_test)

# -----------------------------------
# 6. Accuracy
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy: - exp4bayes.py:43", accuracy)

# -----------------------------------
# 7. Graph (Accuracy Plot)
# -----------------------------------
models = ['Naive Bayes']
scores = [accuracy]

plt.bar(models, scores)
plt.title("Bayesian Learning Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)   # scale from 0 to 1
plt.show()