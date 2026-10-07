from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os


# -------------------------
# 1. Load dataset
# -------------------------

iris = load_iris()

X = iris.data
y = iris.target


# -------------------------
# 2. Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# 3. Train model
# -------------------------

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)


# -------------------------
# 4. Evaluate model
# -------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Model Accuracy: {accuracy:.4f}")


# -------------------------
# 5. Model quality gate
# -------------------------

THRESHOLD = 0.85

if accuracy < THRESHOLD:
    raise ValueError(
        f"Model accuracy {accuracy:.4f} "
        f"is below required threshold {THRESHOLD}"
    )


# -------------------------
# 6. Save model
# -------------------------

os.makedirs("artifacts", exist_ok=True)

joblib.dump(
    model,
    "artifacts/model.pkl"
)

print("Model saved successfully!")