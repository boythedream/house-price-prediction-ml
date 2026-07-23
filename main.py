import streamlit as st
import pandas as pd
import joblib


# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------
# Load Model
# ---------------------------

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ---------------------------
# Custom CSS
# ---------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #1e1b4b 0%, #0f172a 40%, #0b1220 100%);
        background-attachment: fixed;
    }

    /* Hide default streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .hero-badge {
        display: inline-block;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        padding: 6px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }

    h1.hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: clamp(28px, 5vw, 48px);
        background: linear-gradient(90deg, #ffffff 20%, #a5b4fc 60%, #f0abfc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: clamp(14px, 2vw, 17px);
        margin-top: 10px;
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Glass card */
    .glass-card {
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 28px 26px;
        border-radius: 20px;
        margin-bottom: 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    .section-label {
        color: #c4b5fd;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }

    label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        height: 56px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
        transition: all 0.25s ease;
        letter-spacing: 0.3px;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.5);
        color: white;
        border: none;
    }

    div.stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 34px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(5, 150, 105, 0.35);
        animation: fadeUp 0.5s ease;
        margin-top: 10px;
    }

    .result-label {
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.85;
        font-weight: 600;
    }

    .result-value {
        font-family: 'Poppins', sans-serif;
        font-size: clamp(32px, 6vw, 52px);
        font-weight: 800;
        margin: 10px 0 4px 0;
    }

    .result-sub {
        font-size: 13px;
        opacity: 0.75;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Summary chips */
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
        justify-content: center;
    }

    .chip {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #e2e8f0;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
    }

    /* Responsive tweaks */
    @media (max-width: 768px) {
        .glass-card {
            padding: 20px 16px;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Header
# ---------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">✨ ML-Powered Valuation</div>
        <h1 class="hero-title">AI House Price Predictor</h1>
        <p class="hero-subtitle">
            Get an instant, data-driven estimate of any property's market value
            using a trained Gradient Boosting model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_LOADED:
    st.error(
        f"⚠️ Could not load the model file `house_price_model.pkl`. "
        f"Make sure it's in the same directory as this app.\n\nDetails: {MODEL_ERROR}"
    )
    st.stop()


# ---------------------------
# Input Section
# ---------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">📐 Property Basics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)

with col2:
    bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5, 6], index=2)

with col3:
    bathrooms = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4], index=1)

with col4:
    stories = st.selectbox("🏢 Stories", [1, 2, 3, 4], index=0)

st.markdown("</div>", unsafe_allow_html=True)


st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">🧱 Amenities & Features</div>', unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4)

with col5:
    mainroad = st.selectbox("🛣 Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("🛋 Guest Room", ["yes", "no"])

with col6:
    basement = st.selectbox("🏚 Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["yes", "no"])

with col7:
    airconditioning = st.selectbox("❄️ Air Conditioning", ["yes", "no"])
    parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3])

with col8:
    prefarea = st.selectbox("⭐ Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "🪑 Furnishing",
        ["furnished", "semi-furnished", "unfurnished"],
    )

st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Prediction
# ---------------------------

predict_clicked = st.button("🚀 Predict House Price")

if predict_clicked:
    with st.spinner("Crunching the numbers..."):
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
            prediction = model.predict(input_data)[0]

            st.markdown(
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
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Something went wrong while predicting: {e}")