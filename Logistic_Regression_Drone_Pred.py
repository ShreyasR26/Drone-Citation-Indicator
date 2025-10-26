import streamlit as st
import numpy as np
import joblib
import os
import requests
from pyproj import Transformer
from sklearn.base import BaseEstimator, ClassifierMixin
from streamlit_folium import st_folium
import folium

# -------------------------------------------------
# 0️⃣ ThresholdedModel class
# -------------------------------------------------
class ThresholdedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, threshold=0.7):
        self.base_model = base_model
        self.threshold = threshold

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        proba = self.base_model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

# -------------------------------------------------
# 1️⃣ Cached model loader
# -------------------------------------------------
@st.cache_resource
def load_model():
    """Load trained logistic regression pipeline."""
    model_path = "drone_logisticreg_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {os.path.abspath(model_path)}")
        st.stop()
    st.write(f"✅ Loading model from: {os.path.abspath(model_path)}")
    return joblib.load(model_path)

# -------------------------------------------------
# 2️⃣ Fake raster + features (no rasterio)
# -------------------------------------------------
@st.cache_resource
def load_raster():
    """Stub for compatibility – returns None since no raster is used."""
    return None

def get_features_from_coords(_, lon, lat):
    """Generate pseudo-features from lon/lat for demo purposes."""
    # random seed based on lat/lon for reproducibility
    rng = np.random.default_rng(abs(int(lon * lat * 1000)) % 2**32)
    # 6 fake numeric features using lat, lon, and random noise
    features = np.array([lon, lat] + rng.random(4).tolist()).reshape(1, -1)
    return features

# -------------------------------------------------
# 3️⃣ Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Drone Citation Indicator", layout="wide")

st.title("🛰️ Drone Citation Indicator (Demo - No RasterIO)")
st.caption(
    "Click a location on the map to estimate drone presence using a logistic regression model (threshold = 0.7)."
)

# -------------------------------------------------
# 4️⃣ Interactive map
# -------------------------------------------------
st.subheader("🗺️ Select a location")

# Create Folium map
m = folium.Map(location=[31.5, -100.0], zoom_start=6)
map_data = st_folium(m, width=700, height=450)

lat, lon = None, None
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Selected Coordinates — Latitude: {lat:.4f}, Longitude: {lon:.4f}")

# -------------------------------------------------
# 5️⃣ Prediction
# -------------------------------------------------
if lat is not None and lon is not None and st.button("Predict Drone Probability"):
    raster = load_raster()  # stubbed
    model = load_model()
    X = get_features_from_coords(raster, lon, lat)

    if X is not None:
        try:
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]
            st.metric("🧮 Probability of Drone Presence", f"{prob*100:.2f}%")

            if pred == 1:
                st.success("🚁 Drone Detected (≥ 0.7 threshold)")
            else:
                st.info("🌿 No Drone Detected (< 0.7 threshold)")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------------------------
# 6️⃣ Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Model:** Logistic Regression (Scaled, Threshold = 0.7)  
    **Raster:** Disabled (demo version without RasterIO)  
    **Map:** Click anywhere to select coordinates  
    """
)
