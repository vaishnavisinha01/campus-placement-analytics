import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. LOAD DATA


df = pd.read_csv("data/placement_data.csv")



# 2. SELECT FEATURES


features = [
    "cgpa",
    "dsa_problems",
    "projects",
    "internships",
    "certifications",
    "aptitude_score",
    "communication_score",
    "technical_score"
]

X = df[features]

y = df["placement_status"]



# 3. SPLIT DATA


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)



# 4. CREATE MODEL


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)



# 5. TRAIN MODEL


model.fit(X_train, y_train)



# 6. MAKE PREDICTIONS


y_pred = model.predict(X_test)


# 7. EVALUATE MODEL


accuracy = accuracy_score(y_test, y_pred)

print("=" * 60)
print("CAMPUS PLACEMENT PREDICTION MODEL")
print("=" * 60)

print("\nModel: Random Forest Classifier")

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# 8. FEATURE IMPORTANCE


importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)



# 9. SAVE MODEL


joblib.dump(
    model,
    "placement_model.pkl"
)

print("\nModel saved as: placement_model.pkl")

print("=" * 60)