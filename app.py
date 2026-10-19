"""Streamlit app: London Airbnb price recommendation for hosts.

Uses the 10-feature app model (app_model.joblib) trained in the notebook.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

# Charing Cross, used as "central London" (same point as in the notebook)
CENTER_LAT, CENTER_LON = 51.5074, -0.1278
# Typical test error of the app model (MAE in pounds, from the notebook)
APP_MAE_GBP = 65


@st.cache_resource
def load_assets():
    """Load the model and the helper files once."""
    model = joblib.load(BASE / "app_model.joblib")
    model.steps[-1][1].set_params(n_jobs=1)
    centers = json.load(open(BASE / "neighbourhood_centers.json"))
    options = json.load(open(BASE / "options.json"))
    return model, centers, options


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    r = 6371
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


model, centers, options = load_assets()

st.set_page_config(page_title="London Airbnb Price Advisor", page_icon="🏠")
st.title("🏠 London Airbnb price advisor")
st.caption("Enter your listing details to see the market price hosts set for similar listings.")

col1, col2 = st.columns(2)
with col1:
    neighbourhood = st.selectbox("Neighbourhood", options["neighbourhood"])
    room_type = st.selectbox("Room type", options["room_type"])
    property_type = st.selectbox("Property type", options["property_type"])
    accommodates = st.slider("Guests (accommodates)", 1, 10, 2)
with col2:
    bedrooms = st.slider("Bedrooms", 1, 5, 1)
    bathrooms = st.slider("Bathrooms", 0.5, 3.5, 1.0, step=0.5)
    bath_shared = st.checkbox("Shared bathroom")
    minimum_nights = st.slider("Minimum nights", 1, 30, 2)
    n_amenities = st.slider("Number of amenities", 4, 65, 30)

if st.button("Predict price", type="primary"):
    c = centers[neighbourhood]
    dist = haversine_km(c["latitude"], c["longitude"], CENTER_LAT, CENTER_LON)

    # One-row table with the exact column names the model was trained on
    row = pd.DataFrame([{
        "neighbourhood_cleansed": neighbourhood,
        "room_type": room_type,
        "property_type": property_type,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "bath_shared": int(bath_shared),
        "minimum_nights": minimum_nights,
        "n_amenities": n_amenities,
        "dist_center_km": dist,
    }])

    # Model predicts log(1 + price), so convert back to pounds
    price = float(np.expm1(model.predict(row)[0]))

    st.metric("Estimated nightly price", f"£{price:,.0f}")
    st.write(f"Typical error of this model is about **±£{APP_MAE_GBP}**.")
    st.write(f"Distance to central London: **{dist:.1f} km** (calculated from the neighbourhood).")

st.divider()
st.caption(
    "The model learned prices that hosts currently set on Airbnb (London, £15-£1500 range). "
    "It shows the market level, not the optimal price. Less accurate for very expensive listings."
)
