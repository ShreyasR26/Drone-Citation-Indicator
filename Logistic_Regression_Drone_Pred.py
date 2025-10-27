import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import numpy as np
import joblib
import os
from pyproj import Transformer
from sklearn.base import BaseEstimator, ClassifierMixin
from streamlit_folium import st_folium
import folium
import shap
import matplotlib.pyplot as plt
import pandas as pd
import requests

# -------------------------------------------------
# 0️⃣ Model + Raster Configuration
# -------------------------------------------------
MODEL_PATH = "/Users/shreyasramani/Downloads/drone_logisticreg_model.pkl"

# 19-band TIFF locations
LOCAL_TIF = "/Users/shreyasramani/Downloads/border_us_only_stacked.tif"
DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/msgh3wqd6z4ky4o5a7jkq/"
    "border_us_only_stacked_cog.tif?rlkey=k1xfi4hw1zmkggfyqttx3pob1&st=g5etjgw5&dl=1"
)

USED_BANDS = [
    "dem",
    "slope_riserun",
    "aspect",
    "curvature",
    "planform_curvature",
    "profile_curvature",
    "ndvi",
    "landscan",
    "nighttime lights",
    "precip",
    "aero",
    "utilities",
    "medical",
    "military",
    "events",
    "parks",
    "schools",
    "university",
    "all",
]

# -------------------------------------------------
# 1️⃣ Model Wrapper
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
# 2️⃣ Cached loaders
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found: {MODEL_PATH}")
        st.stop()
    st.write(f"✅ Loading model from: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_raster():
    """
    Try opening a local TIFF lazily (fast).
    If not found, open directly from Dropbox (COG streaming).
    """
    try:
        if os.path.exists(LOCAL_TIF):
            st.write(f"📂 Using local TIFF: {LOCAL_TIF}")
            raster = rasterio.open(LOCAL_TIF)
        else:
            st.info("🌐 Local TIFF not found. Accessing directly from Dropbox (COG streaming)...")
            raster = rasterio.open(DROPBOX_URL)
        st.success(f"✅ Raster opened successfully with {raster.count} bands.")
        return raster
    except Exception as e:
        st.error(f"Failed to open raster: {e}")
        st.stop()

# -------------------------------------------------
# 3️⃣ Extract features
# -------------------------------------------------
def get_features_from_coords(raster, lon, lat):
    """Extract 19 features from raster lazily (no full read)."""
    try:
        transformer = Transformer.from_crs("EPSG:4326", raster.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        bounds = raster.bounds
        if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
            st.warning("⚠️ Coordinates outside raster bounds.")
            return None

        row, col = raster.index(x, y)
        window = rasterio.windows.Window(col, row, 1, 1)
        values = [raster.read(i, window=window)[0, 0] for i in range(1, raster.count + 1)]
        return np.array(values, dtype="float32").reshape(1, -1)

    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# -------------------------------------------------
# 4️⃣ Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Drone Citation Indicator", layout="wide")

st.title("🛰️ Drone Citation Indicator")
st.caption(
    "Estimate drone presence probability using a logistic regression model "
    "and a 19-band geospatial feature raster (local or Dropbox COG)."
)

st.sidebar.subheader("Raster Source")
use_local = st.sidebar.radio(
    "Choose raster source:",
    ["Local (fast, lazy load)", "Dropbox COG (streamed)"],
    index=0,
)

# Load raster
if use_local == "Local (fast, lazy load)":
    raster_path = LOCAL_TIF
else:
    raster_path = DROPBOX_URL

# Map
st.subheader("🗺️ Select a location")
m = folium.Map(location=[31.5, -100.0], zoom_start=6)
map_data = st_folium(m, width=700, height=450)

lat, lon = None, None
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    st.success(f"📍 Selected — Lat: {lat:.4f}, Lon: {lon:.4f}")

# -------------------------------------------------
# 5️⃣ Prediction + SHAP
# -------------------------------------------------
if lat is not None and lon is not None and st.button("Predict Drone Probability"):
    model = load_model()
    raster = rasterio.open(raster_path)

    X = get_features_from_coords(raster, lon, lat)
    if X is not None:
        try:
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            st.metric(label="🧮 Probability of Drone Presence", value=f"{prob*100:.2f}%")

            if pred == 1:
                st.success("🚁 Drone Likely Detected (≥ 0.7 threshold)")
            else:
                st.info("🌿 Low Drone Likelihood (< 0.7 threshold)")

            # SHAP explainability
            st.subheader("📊 Top Factors Influencing This Prediction")
            try:
                lr = model.base_model[-1]
                scaler = model.base_model[0]

                background = np.random.normal(0, 1, (50, X.shape[1]))
                explainer = shap.LinearExplainer(
                    lr, background, feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(scaler.transform(X))

                contributions = list(zip(USED_BANDS, shap_values[0]))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                top_features = contributions[:5]

                st.write("These features most influenced this prediction:")
                for feat, val in top_features:
                    emoji = "🔺" if val > 0 else "🔻"
                    st.write(f"{emoji} **{feat}**: {val:+.3f}")

                top_df = pd.DataFrame(top_features, columns=["Feature", "SHAP Value"])
                top_df["Color"] = top_df["SHAP Value"].apply(lambda x: "red" if x > 0 else "blue")

                fig, ax = plt.subplots()
                ax.barh(top_df["Feature"], top_df["SHAP Value"],
                        color=top_df["Color"], edgecolor="black")
                ax.set_xlabel("Impact on Drone Probability (SHAP Value)")
                ax.set_ylabel("Feature")
                ax.set_title("Top Factors Influencing This Location")
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Could not compute SHAP explanation: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------------------------
# 6️⃣ Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Model:** Logistic Regression (Scaled, Threshold = 0.7)  
    **Raster:** 19-band GeoTIFF (local or Dropbox COG)  
    **Explainability:** SHAP-based local feature importance  
    """
)
