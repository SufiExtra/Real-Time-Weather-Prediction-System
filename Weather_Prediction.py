import pandas as pd
import requests
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("seattle-weather.csv")
df = df.dropna()


# ============================================================
# 2. DATE FEATURE ENGINEERING
# ============================================================

df['date'] = pd.to_datetime(df['date'])

df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


# ============================================================
# 3. CREATE TOMORROW WEATHER TARGET
# ============================================================

df['tomorrow_weather'] = df['weather'].shift(-1)
df = df.dropna()


# ============================================================
# 4. SELECT FEATURES AND TARGET
# ============================================================

X = df[['precipitation', 'temp_max', 'temp_min', 'wind', 'month', 'day']]
y = df['tomorrow_weather']


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 6. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model_accuracies = {}
trained_models = {}


# ============================================================
# 7. MODEL 1: KNN
# ============================================================

knn_model = KNeighborsClassifier(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_predictions)

model_accuracies["KNN"] = knn_accuracy
trained_models["KNN"] = knn_model

print("\n================ KNN MODEL ================")
print(f"Accuracy: {knn_accuracy:.2f}")


# ============================================================
# 8. MODEL 2: DECISION TREE
# ============================================================

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

model_accuracies["Decision Tree"] = dt_accuracy
trained_models["Decision Tree"] = dt_model

print("\n============= DECISION TREE MODEL =============")
print(f"Accuracy: {dt_accuracy:.2f}")


# ============================================================
# 9. MODEL 3: RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

model_accuracies["Random Forest"] = rf_accuracy
trained_models["Random Forest"] = rf_model

print("\n============= RANDOM FOREST MODEL =============")
print(f"Accuracy: {rf_accuracy:.2f}")


# ============================================================
# 10. MODEL 4: LOGISTIC REGRESSION
# ============================================================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_predictions = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_predictions)

model_accuracies["Logistic Regression"] = lr_accuracy
trained_models["Logistic Regression"] = lr_model

print("\n=========== LOGISTIC REGRESSION MODEL ===========")
print(f"Accuracy: {lr_accuracy:.2f}")


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = trained_models[best_model_name]

print("\n================ MODEL COMPARISON ================")

for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(model_accuracies[best_model_name], 2))


# ============================================================
# 12. MODEL ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(9, 5))
plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.title("Weather Prediction Model Accuracy Comparison")
plt.show()


# ============================================================
# 13. FETCH TODAY'S WEATHER FROM OPEN-METEO API
# ============================================================

def get_today_weather(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        "&forecast_days=1"
        "&timezone=auto"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("API request failed. Please check internet connection or coordinates.")

    data = response.json()

    today_weather = {
        "precipitation": data["daily"]["precipitation_sum"][0],
        "temp_max": data["daily"]["temperature_2m_max"][0],
        "temp_min": data["daily"]["temperature_2m_min"][0],
        "wind": data["daily"]["windspeed_10m_max"][0],
        "month": datetime.today().month,
        "day": datetime.today().day
    }

    return today_weather


# ============================================================
# 14. USER ENTERS LOCATION ONLY
# ============================================================

print("\nEnter your location coordinates:")

latitude = float(input("Enter Latitude: "))
longitude = float(input("Enter Longitude: "))


# ============================================================
# 15. PREDICT TOMORROW WEATHER
# ============================================================

today_weather = get_today_weather(latitude, longitude)

print("\nToday's Weather Data From API:")
print(today_weather)

today_data = pd.DataFrame([{
    "precipitation": today_weather["precipitation"],
    "temp_max": today_weather["temp_max"],
    "temp_min": today_weather["temp_min"],
    "wind": today_weather["wind"],
    "month": today_weather["month"],
    "day": today_weather["day"]
}])

if best_model_name in ["KNN", "Logistic Regression"]:
    today_data = scaler.transform(today_data)

tomorrow_weather = best_model.predict(today_data)

print("\n======================================")
print("Predicted Tomorrow Weather:", tomorrow_weather[0])
print("Model Used:", best_model_name)
print("======================================")

