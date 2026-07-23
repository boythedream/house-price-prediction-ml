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
    initial_sidebar_state="expanded",
)

HERO_IMAGE_URL = "https://images.unsplash.com/photo-1760067537293-6b30141d6a52?auto=format&fit=crop&w=1600&q=80"


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
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 10% 0%, #1e1b4b 0%, #0f172a 45%, #0b1220 100%);
        background-attachment: fixed;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }}

    /* ---------- Hero banner with image ---------- */
    .hero-banner {{
        position: relative;
        width: 100%;
        height: 260px;
        border-radius: 24px;
        overflow: hidden;
        margin-bottom: 26px;
        background-image:
            linear-gradient(120deg, rgba(15,23,42,0.92) 10%, rgba(30,27,75,0.55) 55%, rgba(15,23,42,0.35) 100%),
            url('{HERO_IMAGE_URL}');
        background-size: cover;
        background-position: center;
        display: flex;
        align-items: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.4);
        border: 1px solid rgba(148,163,184,0.15);
    }}

    .hero-content {{
        padding: 0 40px;
        max-width: 640px;
    }}

    .hero-badge {{
        display: inline-block;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
        box-shadow: 0 4px 18px rgba(139, 92, 246, 0.45);
    }}

    .hero-title {{
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: clamp(24px, 3.6vw, 38px);
        color: #ffffff;
        margin: 0;
        line-height: 1.15;
        text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}

    .hero-sub {{
        color: #cbd5e1;
        font-size: clamp(13px, 1.6vw, 15px);
        margin-top: 10px;
        text-shadow: 0 1px 8px rgba(0,0,0,0.5);
    }}

    @media (max-width: 768px) {{
        .hero-banner {{ height: 210px; }}
        .hero-content {{ padding: 0 22px; }}
    }}

    /* ---------- Cards ---------- */
    .glass-card {{
        background: rgba(30, 41, 59, 0.55);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 24px 24px;
        border-radius: 18px;
        margin-bottom: 20px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.22);
    }}

    .section-label {{
        color: #c4b5fd;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #14102b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148,163,184,0.12);
    }}

    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        color: #e9d5ff;
        font-family: 'Poppins', sans-serif;
    }}

    /* Inputs */
    div[data-baseweb="select"] > div, .stNumberInput input {{
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        border-radius: 10px !important;
        color: #f1f5f9 !important;
    }}

    label {{
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
    }}

    /* Predict button */
    div.stButton > button {{
        width: 100%;
        height: 54px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
        transition: all 0.25s ease;
    }}

    div.stButton > button:hover {{
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 12px 30px rgba(139, 92, 246, 0.5);
        color: white;
        border: none;
    }}

    /* Result card */
    .result-card {{
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(5, 150, 105, 0.35);
        animation: fadeUp 0.5s ease;
    }}

    .result-label {{
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        opacity: 0.85;
        font-weight: 600;
    }}

    .result-value {{
        font-family: 'Poppins', sans-serif;
        font-size: clamp(30px, 5.5vw, 46px);
        font-weight: 800;
        margin: 8px 0 4px 0;
    }}

    .result-sub {{
        font-size: 12.5px;
        opacity: 0.75;
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Metric mini cards */
    .mini-stat {{
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 14px 16px;
        text-align: center;
    }}

    .mini-stat .val {{
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 20px;
        color: #f1f5f9;
    }}

    .mini-stat .lbl {{
        font-size: 11.5px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 3px;
    }}

    .chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
        justify-content: center;
    }}

    .chip {{
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        color: #ffffff;
        padding: 5px 13px;
        border-radius: 999px;
        font-size: 12.5px;
    }}

    .empty-state {{
        text-align: center;
        padding: 50px 20px;
        color: #94a3b8;
    }}

    .empty-state .emoji {{
        font-size: 46px;
        margin-bottom: 10px;
    }}

    @media (max-width: 768px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .glass-card {{ padding: 18px 16px; }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Hero Banner
# ---------------------------

st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-content">
            <div class="hero-badge">✨ ML-POWERED VALUATION</div>
            <h1 class="hero-title">Know Your Property's True Worth</h1>
            <p class="hero-sub">
                Instant, data-driven price estimates powered by a trained
                Gradient Boosting model — enter your property details in the
                sidebar to get started.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_LOADED:
    st.error(
        f"⚠️ Could not load `house_price_model.pkl`. Make sure it's in the same "
        f"directory as this app.\n\nDetails: {MODEL_ERROR}"
    )
    st.stop()


# ---------------------------
# Sidebar — Inputs
# ---------------------------

with st.sidebar:
    st.markdown("## 🧾 Property Details")
    st.caption("Fill these in, then hit **Predict**.")

    st.markdown("**📐 Basics**")
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
    bedrooms = st.selectbox("🛏 Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4], index=1)
    stories = st.selectbox("🏢 Stories", [1, 2, 3, 4], index=0)

    st.divider()

    st.markdown("**🧱 Amenities**")
    mainroad = st.selectbox("🛣 Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("🛋 Guest Room", ["yes", "no"])
    basement = st.selectbox("🏚 Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("🔥 Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("❄️ Air Conditioning", ["yes", "no"])
    parking = st.selectbox("🚗 Parking Spaces", [0, 1, 2, 3])
    prefarea = st.selectbox("⭐ Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "🪑 Furnishing", ["furnished", "semi-furnished", "unfurnished"]
    )

    st.divider()
    predict_clicked = st.button("🚀 Predict Price")


# ---------------------------
# Main Area
# ---------------------------

left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Quick Overview</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="mini-stat"><div class="val">{area:,}</div><div class="lbl">Sq Ft</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mini-stat"><div class="val">{bedrooms}</div><div class="lbl">Bedrooms</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="mini-stat"><div class="val">{bathrooms}</div><div class="lbl">Bathrooms</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="mini-stat"><div class="val">{stories}</div><div class="lbl">Stories</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    features = {
        "Main Road": mainroad, "Guest Room": guestroom, "Basement": basement,
        "Hot Water Heating": hotwaterheating, "Air Conditioning": airconditioning,
        "Preferred Area": prefarea,
    }
    feature_chips = "".join(
        f'<span class="chip">{("✅" if v == "yes" else "▫️")} {k}</span>'
        for k, v in features.items()
    )
    st.markdown(f'<div class="chip-row">{feature_chips}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">💰 Estimated Price</div>', unsafe_allow_html=True)

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
                price_per_sqft = prediction / area

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Price</div>
                        <div class="result-value">💰 {prediction:,.0f} PKR</div>
                        <div class="result-sub">≈ {price_per_sqft:,.0f} PKR / sq ft · {furnishingstatus}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Something went wrong while predicting: {e}")
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="emoji">🏡</div>
                Fill in the property details on the left<br>and click <b>Predict Price</b> to see your estimate.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)