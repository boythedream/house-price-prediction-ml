import textwrap

import joblib
import pandas as pd
import streamlit as st

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def html(s: str) -> str:
    """Remove indentation/blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in textwrap.dedent(s).splitlines() if line.strip())


# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


MODEL_ERROR = ""
try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ---------------------------
# Custom CSS
# ---------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 10% 10%, #1e1b4b 0%, #0f172a 40%, #0b1220 100%);
    background-attachment: fixed;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Hero */
.hero { text-align: center; padding: 10px 0 25px 0; }
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white; padding: 6px 18px; border-radius: 999px;
    font-size: 13px; font-weight: 600; letter-spacing: 0.5px;
    margin-bottom: 14px; box-shadow: 0 4px 20px rgba(99,102,241,0.4);
}
h1.hero-title {
    font-family: 'Poppins', sans-serif; font-weight: 800;
    font-size: clamp(28px, 5vw, 48px);
    background: linear-gradient(90deg, #ffffff 20%, #a5b4fc 60%, #f0abfc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; padding: 0;
}
.hero-subtitle {
    color: #94a3b8; font-size: clamp(14px, 2vw, 17px);
    margin: 10px auto 0 auto; max-width: 560px;
}

/* Glass card = bordered container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30, 41, 59, 0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    padding: 18px 16px;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

.section-label {
    color: #c4b5fd; font-family: 'Poppins', sans-serif;
    font-weight: 700; font-size: 15px; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.8px;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput input {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border-color: rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
label, label p { color: #cbd5e1 !important; font-weight: 500 !important; font-size: 14px !important; }

/* Button */
div.stButton > button {
    width: 100%; height: 56px; border-radius: 14px;
    font-size: 18px; font-weight: 700; font-family: 'Poppins', sans-serif;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white; border: none;
    box-shadow: 0 8px 24px rgba(139,92,246,0.35);
    transition: all 0.25s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 12px 30px rgba(139,92,246,0.5);
    color: white; border: none;
}
div.stButton > button:active { transform: scale(0.99); }

/* Result */
.result-card {
    background: linear-gradient(135deg, #059669 0%, #047857 100%);
    padding: 34px; border-radius: 20px; text-align: center; color: white;
    box-shadow: 0 10px 40px rgba(5,150,105,0.35);
    animation: fadeUp 0.5s ease; margin-top: 16px;
}
.result-label { font-size: 15px; text-transform: uppercase; letter-spacing: 1.5px; opacity: 0.85; font-weight: 600; }
.result-value { font-family: 'Poppins', sans-serif; font-size: clamp(32px, 6vw, 52px); font-weight: 800; margin: 10px 0 4px 0; }
.result-sub { font-size: 13px; opacity: 0.75; }
@keyframes fadeUp { from {opacity:0; transform:translateY(15px);} to {opacity:1; transform:translateY(0);} }

.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; justify-content: center; }
.chip {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    color: #fff; padding: 6px 14px; border-radius: 999px; font-size: 13px;
}

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Header
# ---------------------------
st.markdown(
    html(
        """
        <div class="hero">
            <div class="hero-badge">✨ ML-Powered Valuation</div>
            <h1 class="hero-title">AI House Price Predictor</h1>
            <p class="hero-subtitle">
                Get an instant, data-driven estimate of any property's market value
                using a trained Gradient Boosting model.
            </p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

if not MODEL_LOADED:
    st.error(
        "⚠️ Could not load the model file `house_price_model.pkl`. "
        "Make sure it's in the same folder as this app and that scikit-learn "
        "has the same version used for training.\n\n"
        f"Details: {MODEL_ERROR}"
    )
    st.stop()


# ---------------------------
# Inputs
# ---------------------------
with st.container(border=True):
    st.markdown('<div class="section-label">📐 Property Basics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
    with c2:
        bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    with c3:
        bathrooms = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4], index=1)
    with c4:
        stories = st.selectbox("🏢 Stories", [1, 2, 3, 4], index=0)

st.write("")

with st.container(border=True):
    st.markdown('<div class="section-label">🧱 Amenities & Features</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        mainroad = st.selectbox("🛣 Main Road Access", ["yes", "no"])
        guestroom = st.selectbox("🛋 Guest Room", ["no", "yes"])
    with c6:
        basement = st.selectbox("🏚 Basement", ["no", "yes"])
        hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["no", "yes"])
    with c7:
        airconditioning = st.selectbox("❄️ Air Conditioning", ["no", "yes"])
        parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3])
    with c8:
        prefarea = st.selectbox("⭐ Preferred Area", ["no", "yes"])
        furnishingstatus = st.selectbox(
            "🪑 Furnishing", ["furnished", "semi-furnished", "unfurnished"]
        )

st.write("")


# ---------------------------
# Prediction helpers
# ---------------------------
def prepare_input(df: pd.DataFrame, model) -> pd.DataFrame:
    """Make the input match what the model was trained on."""
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return df  # e.g. a Pipeline that handles everything itself

    expected = list(expected)
    if set(expected) == set(df.columns):
        return df[expected]  # same columns, just fix the order

    # Model was probably trained on one-hot encoded data
    encoded = pd.get_dummies(df)
    return encoded.reindex(columns=expected, fill_value=0)


# ---------------------------
# Predict
# ---------------------------
if st.button("🚀 Predict House Price"):
    input_data = pd.DataFrame(
        {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "hotwaterheating": [hotwaterheating],
            "airconditioning": [airconditioning],
            "parking": [parking],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishingstatus],
        }
    )

    try:
        with st.spinner("Crunching the numbers..."):
            prediction = float(model.predict(prepare_input(input_data, model))[0])

        st.session_state["result"] = html(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated House Price</div>
                <div class="result-value">💰 {prediction:,.0f} PKR</div>
                <div class="result-sub">Based on the features you provided</div>
                <div class="chip-row">
                    <div class="chip">{area:,} sq ft</div>
                    <div class="chip">{bedrooms} bed · {bathrooms} bath</div>
                    <div class="chip">{stories} storey(s)</div>
                    <div class="chip">{parking} parking</div>
                    <div class="chip">{furnishingstatus}</div>
                </div>
            </div>
            """
        )
        st.session_state.pop("error", None)
    except Exception as e:
        st.session_state["error"] = str(e)
        st.session_state.pop("result", None)

if "result" in st.session_state:
    st.markdown(st.session_state["result"], unsafe_allow_html=True)
if "error" in st.session_state:
    st.error(f"Something went wrong while predicting: {st.session_state['error']}")