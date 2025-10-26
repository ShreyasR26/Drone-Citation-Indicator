import streamlit as st
import rasterio
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
# 1️⃣ Cached model & raster loaders
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

@st.cache_resource
def load_raster():
    """Download large raster from Dropbox if not found locally."""
    raster_path = "border_us_only_stacked.tif"
    dropbox_url = (
        "https://www.dropbox.com/scl/fi/6rvu91g7t03s3biyqwmta/"
        "border_us_only_stacked.tif?rlkey=9u4jztfaxnv5fxmqniyvwxk8b&st=tmkqhqsf&dl=1"
    )

    if not os.path.exists(raster_path):
        st.info("⬇️ Downloading raster file (~2.6 GB)… this may take several minutes.")
        with requests.get(dropbox_url, stream=True) as r:
            r.raise_for_status()
            with open(raster_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("✅ Raster downloaded successfully.")
    return rasterio.open(raster_path)

# -------------------------------------------------
# 2️⃣ Extract features
# -------------------------------------------------
def get_features_from_coords(raster, lon, lat):
    """Extract feature vector for a given lon/lat (auto CRS conversion)."""
    try:
        transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        bounds = raster.bounds
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            st.warning("⚠️ Coordinates are outside raster bounds (after projection).")
            return None
        row, col = raster.index(x, y)
        window = rasterio.windows.Window(col, row, 1, 1)
        data = raster.read(window=window)
        features = data[:, 0, 0]
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# -------------------------------------------------
# 3️⃣ Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Drone Citation Indicator", layout="wide")

st.title("🛰️ Drone Citation Indicator")
st.caption(
    "Click a location on the map to estimate drone presence "
    "using a logistic regression model (threshold = 0.7)."
)

# -------------------------------------------------
# 4️⃣ Interactive map
# -------------------------------------------------
st.subheader("🗺️ Select a location")

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
    raster = load_raster()
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
    **Raster CRS:** EPSG: 6350 (NAD83 / Texas Centric Albers Equal Area)  
    **Map:** Click anywhere to select coordinates  
    """
)
