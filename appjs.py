import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------- UI ----------------
st.set_page_config(page_title="Traffic Optimizer", page_icon="🚦", layout="wide")

st.markdown("""
<style>
.main {
    background: linear-gradient(120deg, #1f4037, #99f2c8);
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Light Optimization System")
st.write("AI-based system to predict optimal traffic signal timing")

# ---------------- DATA GENERATION ----------------
@st.cache_data
def generate_data(n=5000):
    np.random.seed(42)

    vehicles = np.random.randint(20, 1000, n)
    hour = np.random.randint(0, 24, n)
    lane_length = np.random.randint(100, 800, n)
    speed = np.random.randint(5, 80, n)
    weather = np.random.choice(['Clear', 'Rain', 'Fog'], n)
    day = np.random.choice(['Weekday', 'Weekend'], n)
    traffic_type = np.random.choice(['Low', 'Medium', 'High'], n)

    peak_factor = np.where(((hour >= 8) & (hour <= 11)) | ((hour >= 17) & (hour <= 20)), 20, 0)

    weather_effect = np.where(weather == 'Rain', 15,
                      np.where(weather == 'Fog', 20, 0))

    traffic_effect = np.where(traffic_type == 'High', 25,
                      np.where(traffic_type == 'Medium', 10, 0))

    green_time = (
        vehicles * 0.15 +
        lane_length * 0.04 -
        speed * 0.25 +
        peak_factor +
        weather_effect +
        traffic_effect +
        np.where(day == 'Weekend', -5, 5)
    )

    df = pd.DataFrame({
        'vehicles': vehicles,
        'hour': hour,
        'lane_length': lane_length,
        'weather': weather,
        'speed': speed,
        'day': day,
        'traffic_type': traffic_type,
        'green_time': green_time
    })

    return df

df = generate_data()

# ---------------- SUMMARY ----------------
st.subheader("📊 Dataset Summary")
st.write(f"Total Records: {len(df)}")
st.write(f"Average Vehicles: {int(df['vehicles'].mean())}")
st.write(f"Average Green Time: {round(df['green_time'].mean(),2)} sec")

with st.expander("📂 View Dataset (Optional)"):
    st.dataframe(df.head(), use_container_width=True)

# ---------------- VISUALIZATION ----------------
st.subheader("📈 Traffic Insights")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['vehicles'], df['green_time'])
    ax1.set_xlabel("Vehicles")
    ax1.set_ylabel("Green Time")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.hist(df['green_time'], bins=30)
    ax2.set_xlabel("Green Time")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

# ---------------- PREPROCESS ----------------
le_weather = LabelEncoder()
le_day = LabelEncoder()
le_traffic = LabelEncoder()

df['weather'] = le_weather.fit_transform(df['weather'])
df['day'] = le_day.fit_transform(df['day'])
df['traffic_type'] = le_traffic.fit_transform(df['traffic_type'])

X = df[['vehicles', 'hour', 'lane_length', 'weather', 'speed', 'day', 'traffic_type']]
y = df['green_time']

# ---------------- TRAIN MODEL ----------------
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

st.success("✅ Model trained successfully!")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

importance = model.feature_importances_
features = X.columns

fig3, ax3 = plt.subplots()
ax3.bar(features, importance)
ax3.set_ylabel("Importance")
st.pyplot(fig3)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Predict Signal Timing")

col1, col2, col3 = st.columns(3)

with col1:
    vehicles = st.number_input("🚗 Vehicles", 0, 2000, 100)
    hour = st.slider("⏰ Hour", 0, 23, 12)

with col2:
    lane_length = st.number_input("🛣 Lane Length", 50, 1000, 200)
    speed = st.number_input("⚡ Speed", 0, 120, 40)

with col3:
    weather = st.selectbox("🌦 Weather", le_weather.classes_)
    day = st.selectbox("📅 Day", le_day.classes_)
    traffic_type = st.selectbox("🚦 Traffic Type", le_traffic.classes_)

if st.button("🚀 Optimize Signal Timing"):
    weather_encoded = le_weather.transform([weather])[0]
    day_encoded = le_day.transform([day])[0]
    traffic_encoded = le_traffic.transform([traffic_type])[0]

    input_data = [[vehicles, hour, lane_length, weather_encoded, speed, day_encoded, traffic_encoded]]

    green_time = model.predict(input_data)[0]

    # ---------------- SIGNAL LOGIC ----------------
    # Yellow based on speed
    if speed < 20:
        yellow_time = 5
    elif speed < 50:
        yellow_time = 4
    else:
        yellow_time = 3

    # Total cycle time
    total_cycle = max(60, min(180, green_time * 2))

    # Red time
    red_time = total_cycle - green_time - yellow_time

    # ---------------- DISPLAY ----------------
    st.markdown(f"""
        <div style="
            background-color:#000000;
            padding:25px;
            border-radius:15px;
            text-align:center;
            font-size:22px;
            color:white;">
            
            🚦 <b>Optimized Signal Timing</b><br><br>
            
            🟢 Green Time: <b>{round(green_time,2)} sec</b><br>
            🟡 Yellow Time: <b>{yellow_time} sec</b><br>
            🔴 Red Time: <b>{round(red_time,2)} sec</b><br><br>
            
            ⏱ Total Cycle Time: <b>{round(total_cycle,2)} sec</b>
        </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Smart Traffic Optimization System | Built with Machine Learning")