# Smart Energy Forecasting and Optimization System using AI

# --- Dependencies ---
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import streamlit as st

# --- Config ---
API_KEY = "f06e05470d401cf1885e8667247b744e"
CITY = "Hyderabad"

# --- Function to Fetch Real-time Weather Data ---
def get_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        return None

    weather = {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "clouds": data["clouds"]["all"],
        "wind_speed": data["wind"]["speed"],
        "solar_radiation": 1000 - data["clouds"]["all"] * 10  # Approximation
    }
    return weather

# --- Simulated Dataset Generation ---
def generate_dummy_data():
    np.random.seed(42)
    data = pd.DataFrame({
        "temperature": np.random.uniform(20, 45, 100),
        "humidity": np.random.uniform(10, 90, 100),
        "clouds": np.random.uniform(0, 100, 100),
        "wind_speed": np.random.uniform(0, 15, 100),
    })
    data["solar_output"] = 1000 - data["clouds"] * 5 + data["temperature"] * 2 - data["humidity"] * 1.5 + data["wind_speed"] * 2
    return data

# --- Model Training ---
def train_model():
    data = generate_dummy_data()
    X = data.drop("solar_output", axis=1)
    y = data["solar_output"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "solar_model.pkl")
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions)**0.5
    return rmse

# --- Prediction App ---
def run_app():
    st.title("Smart Energy Forecasting and Optimization System ✨")
    st.write("Predict real-time solar energy output using live weather data")

    city_input = st.text_input("Enter your city:", CITY)
    current_usage = st.number_input("Enter current energy usage (in watts):", min_value=0)
    battery_capacity = st.number_input("Enter battery storage capacity (in watts):", min_value=0)


    if st.button("Predict Energy Output"):
        weather = get_weather_data(city_input, API_KEY)
        if not weather:
            st.error("Failed to fetch weather data. Check city name or API key.")
            return

        st.subheader("Current Weather")
        st.json(weather)

        features = [[weather["temperature"], weather["humidity"], weather["clouds"], weather["wind_speed"]]]
        model = joblib.load("solar_model.pkl")
        prediction = model.predict(features)[0]

        st.success(f"Predicted Solar Energy Output: {prediction:.2f} watts/m²")
        st.write(f"Battery Capacity: {battery_capacity:.2f} watts")
        st.write(f"Current Usage: {current_usage:.2f} watts")
        energy_surplus = prediction - current_usage
        if energy_surplus > 0:
            st.info(f"You will generate {energy_surplus:.2f} watts more than your usage.")
            st.subheader("🔎 Smart Energy Tips")
            if energy_surplus <= battery_capacity:
                st.success("Suggestion: Store excess energy in battery for later use.")
            else:
                st.warning("Suggestion: Battery might overflow. Consider selling to grid or reduce solar panel load.")
        else:
            st.warning(f"Deficit of {-energy_surplus:.2f} watts. Solar energy may not meet your needs.")

            if battery_capacity >= abs(energy_surplus):
                st.info("Suggestion: Use stored battery power to meet the deficit.")
            else:
                st.error("Suggestion: Use backup power or reduce appliance load.")
        #recs = generate_recommendations(prediction, current_usage, battery_capacity)
        #st.subheader("🔎 Smart Energy Tips")
        #for rec in recs:
            #st.write(rec)

def generate_recommendations(prediction, current_usage, battery_capacity):
    energy_deficit = prediction - current_usage
    recommendations = []
    st.subheader("Recommendations")
    if prediction > 800:
        recommendations.append("⚡ Great solar potential today! Run high-energy appliances during daylight hours.")
    
    if energy_deficit < 0:
        recommendations.append("🔋 Reduce usage or use battery backup during low solar production.")
    elif energy_deficit > 0 and energy_deficit > battery_capacity:
        recommendations.append("💡 Suggestion: Sell excess energy to the grid or store in a larger battery.")
    
    if prediction < 400:
        recommendations.append("🌥️ Low solar forecast. Schedule essential usage for nighttime if needed.")
    
    return recommendations

# --- Execution ---
if __name__ == "__main__":
    # Train model only once
    train_model()
    run_app()
    generate_recommendations()



