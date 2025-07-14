import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Airline Satisfaction App", layout="centered")
st.title("✈️ Airline Customer Satisfaction Predictor")

# --- Sidebar for Model Selection ---
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "Decision Tree", "Logistic Regression"))

# Load selected model
if model_choice == "Random Forest":
    model = pickle.load(open('random_forest_model.pkl', 'rb'))
elif model_choice == "Decision Tree":
    model = pickle.load(open('decision_tree_model.pkl', 'rb'))
else:
    model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

# Load pre-fitted scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.subheader("Passenger Information")
# Manual mappings from LabelEncoder
gender_map = {'Female': 0, 'Male': 1}
customer_type_map = {'Disloyal Customer': 0, 'Loyal Customer': 1}
travel_type_map = {'Business travel': 0, 'Personal Travel': 1}
class_map = {'Business': 0, 'Eco': 1, 'Eco Plus': 2}

# --- User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.slider("Age", 7, 85, 30)
travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
flight_distance = st.number_input("Flight Distance", min_value=50, max_value=5000, value=1000)

wifi = st.slider("Inflight wifi service", 0, 5, 3)
dep_arr_time = st.slider("Departure/Arrival time convenient", 0, 5, 3)
ease_booking = st.slider("Ease of Online booking", 0, 5, 3)
gate_location = st.slider("Gate location", 0, 5, 3)
food_drink = st.slider("Food and drink", 0, 5, 3)
online_boarding = st.slider("Online boarding", 0, 5, 3)
seat_comfort = st.slider("Seat comfort", 0, 5, 3)
entertainment = st.slider("Inflight entertainment", 0, 5, 3)
onboard_service = st.slider("On-board service", 0, 5, 3)
leg_room = st.slider("Leg room service", 0, 5, 3)
baggage = st.slider("Baggage handling", 0, 5, 3)
checkin_service = st.slider("Checkin service", 0, 5, 3)
inflight_service = st.slider("Inflight service", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)
departure_delay = st.number_input("Departure Delay in Minutes", 0, 1440, 0)

# --- Prepare Input for Prediction ---
input_data = pd.DataFrame([{
    'Gender': gender_map[gender],
    'Customer Type': customer_type_map[customer_type],
    'Age': age,
    'Type of Travel': travel_type_map[travel_type],
    'Class': class_map[travel_class],
    'Flight Distance': flight_distance,
    'Inflight wifi service': wifi,
    'Departure/Arrival time convenient': dep_arr_time,
    'Ease of Online booking': ease_booking,
    'Gate location': gate_location,
    'Food and drink': food_drink,
    'Online boarding': online_boarding,
    'Seat comfort': seat_comfort,
    'Inflight entertainment': entertainment,
    'On-board service': onboard_service,
    'Leg room service': leg_room,
    'Baggage handling': baggage,
    'Checkin service': checkin_service,
    'Inflight service': inflight_service,
    'Cleanliness': cleanliness,
    'Departure Delay in Minutes': departure_delay
}])

# --- Predict ---
if st.button("Predict Satisfaction"):
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"Prediction: {'Satisfied' if prediction == 1 else 'Dissatisfied'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
