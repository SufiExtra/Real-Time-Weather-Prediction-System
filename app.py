import streamlit as st
import pandas as pd
import requests
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


st.title("🌦 Weather Prediction System")
st.write("Predict tomorrow's weather using Machine Learning and Weather API")


# =============================
# Load Dataset
# =============================

df = pd.read_csv("seattle-weather.csv")
df = df.dropna()

df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df["tomorrow_weather"] = df["weather"].shift(-1)
df = df.dropna()

X = df[["precipitation", "temp_max", "temp_min", "wind", "month", "day"]]
y = df["tomorrow_weather"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================
# Models
# =============================

models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

accuracies = {}
trained_models = {}

for name, model in models.items():
    if name in ["KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    accuracies[name] = accuracy_score(y_test, pred)
    trained_models[name] = model

best_model_name = max(accuracies, key=accuracies.get)
best_model = trained_models[best_model_name]


# =============================
# API Function
# =============================

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
    data = response.json()

    return {
        "precipitation": data["daily"]["precipitation_sum"][0],
        "temp_max": data["daily"]["temperature_2m_max"][0],
        "temp_min": data["daily"]["temperature_2m_min"][0],
        "wind": data["daily"]["windspeed_10m_max"][0],
        "month": datetime.today().month,
        "day": datetime.today().day
    }


# =============================
# User Input
# =============================

st.subheader("📍 Enter Location Coordinates")

latitude = st.number_input("Enter Latitude", value=33.91)
longitude = st.number_input("Enter Longitude", value=72.49)

if st.button("Predict Tomorrow Weather"):

    if latitude < -90 or latitude > 90:
        st.error("Invalid latitude. Latitude must be between -90 and 90.")

    elif longitude < -180 or longitude > 180:
        st.error("Invalid longitude. Longitude must be between -180 and 180.")

    else:
        today_weather = get_today_weather(latitude, longitude)

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

        prediction = best_model.predict(today_data)[0]

        st.success(f"🌤 Predicted Tomorrow Weather: {prediction}")
        st.info(f"Best Model Used: {best_model_name}")
        

        st.subheader("Today's Weather Data From API")
        st.write(today_weather)


st.subheader("📊 Model Accuracy Comparison")
st.bar_chart(pd.DataFrame.from_dict(accuracies, orient="index", columns=["Accuracy"]))