import joblib

feature_names = joblib.load("feature_names.pkl")

print(feature_names)
print("\nNumber of features:", len(feature_names))