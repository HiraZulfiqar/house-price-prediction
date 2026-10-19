
import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Noir Estate | Property Valuation",
    page_icon="🏡",
    layout="wide"
)

# ---------------- LUXURY DARK THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

/* Main background */
.stApp {
    background-color: #171717;
    color: #F2F2F2;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #F5F5F5 !important;
}

p, li, label {
    color: #D0D0D0;
}

/* Brand */
.brand {
    color: #D4AF70;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 5px;
    margin-bottom: 20px;
}

/* Hero section */
.hero {
    background: linear-gradient(125deg, #101010, #303030);
    padding: 42px 38px;
    border: 1px solid #48433A;
    border-bottom: 3px solid #C5A467;
    border-radius: 12px;
    margin: 15px 0 30px 0;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 42px !important;
    line-height: 1.2;
    margin: 12px 0;
}

.hero p {
    color: #C7C7C7;
    font-size: 16px;
    line-height: 1.8;
}

.gold-label {
    color: #D4AF70;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}

.section-label {
    color: #D4AF70;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Input labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #E8E8E8 !important;
    font-weight: 600 !important;
}

/* Number input boxes and typed text */
div[data-testid="stNumberInput"] input {
    background-color: #292929 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid #555555 !important;
    caret-color: #D4AF70 !important;
}

/* Number input buttons */
div[data-testid="stNumberInput"] button {
    background-color: #333333 !important;
    color: #D4AF70 !important;
    border-color: #555555 !important;
}

/* Dropdown boxes */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #292929 !important;
    border-color: #555555 !important;
    color: #FFFFFF !important;
}

/* Dropdown selected text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="menu"] {
    background-color: #292929 !important;
}

div[role="option"] {
    background-color: #292929 !important;
    color: #FFFFFF !important;
}

/* ---------------- BLACK & GOLD BUTTON ---------------- */
div.stButton > button {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1.5px solid #C58B55 !important;
    border-radius: 8px !important;
    padding: 13px 20px !important;
    min-height: 48px;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    box-shadow: none !important;
    transition: all 0.25s ease !important;
}

/* Button hover */
div.stButton > button:hover {
    background-color: #242424 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1.5px solid #E0AD78 !important;
    box-shadow: 0 0 10px rgba(197, 139, 85, 0.20) !important;
}

/* Button focus */
div.stButton > button:focus,
div.stButton > button:focus-visible {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border-color: #E0AD78 !important;
    outline: 2px solid rgba(224, 173, 120, 0.35) !important;
    outline-offset: 2px !important;
}

/* Button text */
div.stButton > button p {
    color: #FFFFFF !important;
}

/* Form panel */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #202020;
    border-color: #414141 !important;
    border-radius: 10px;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #242424, #111111);
    color: #FFFFFF;
    padding: 30px;
    border: 1px solid #514630;
    border-left: 4px solid #C5A467;
    border-radius: 10px;
    margin-top: 20px;
}

.result-label {
    color: #D4AF70;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 700;
}

.result-price {
    color: #FFFFFF;
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 600;
    margin: 14px 0;
    overflow-wrap: anywhere;
}

.result-note {
    color: #C5C5C5;
    font-size: 13px;
    line-height: 1.7;
}

/* Divider */
hr {
    border-color: #424242 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #888888;
    border-top: 1px solid #383838;
    margin-top: 45px;
    padding-top: 22px;
    font-size: 11px;
    letter-spacing: 2px;
}

/* Hide Streamlit menu/footer */
#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD TRAINED MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


model = load_model()


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="brand">NOIR ESTATE &nbsp; | &nbsp; PROPERTY VALUATION</div>

<div class="hero">
    <div class="gold-label">INTELLIGENT PROPERTY ESTIMATES</div>
    <h1>Where elegance meets<br>property intelligence.</h1>
    <p>
        Discover an estimated value for your property using
        machine learning. Enter the details below to begin
        your valuation.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- PROPERTY DETAILS ----------------
st.markdown(
    '<div class="section-label">01 — PROPERTY PROFILE</div>',
    unsafe_allow_html=True
)

st.header("Property Details")
st.write("Complete the information below to prepare your estimate.")

with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area",
            min_value=1,
            value=5000,
            step=100
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            value=2,
            step=1
        )

        stories = st.number_input(
            "Stories",
            min_value=1,
            value=2,
            step=1
        )

    with col2:
        mainroad = st.selectbox(
            "Main road access", ["yes", "no"]
        )

        guestroom = st.selectbox(
            "Guest room", ["yes", "no"]
        )

        basement = st.selectbox(
            "Basement", ["yes", "no"]
        )

        hotwaterheating = st.selectbox(
            "Hot water heating", ["yes", "no"]
        )

    with col3:
        airconditioning = st.selectbox(
            "Air conditioning", ["yes", "no"]
        )

        parking = st.number_input(
            "Parking spaces",
            min_value=0,
            value=1,
            step=1
        )

        prefarea = st.selectbox(
            "Preferred area", ["yes", "no"]
        )

        furnishingstatus = st.selectbox(
            "Furnishing status",
            ["furnished", "semi-furnished", "unfurnished"]
        )


# ---------------- VALUATION ----------------
st.markdown("---")

st.markdown(
    '<div class="section-label">02 — YOUR VALUATION</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Ready to discover the value?")
    st.write(
        "When you're ready, request an estimate based on "
        "the property details you entered."
    )

    predict_clicked = st.button(
        "✦  ESTIMATE PROPERTY VALUE",
        use_container_width=True
    )

with right:
    st.markdown("""
    <div class="result-card">
        <div class="result-label">ESTIMATED MARKET VALUE</div>
        <div class="result-price">Awaiting valuation</div>
        <div class="result-note">
            Your estimated value will appear below after
            you request a prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAKE PREDICTION ----------------
if predict_clicked:

    house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    try:
        prediction = model.predict(house)[0]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">ESTIMATED MARKET VALUE</div>
            <div class="result-price">PKR {prediction:,.0f}</div>
            <div class="result-note">
                Generated by your trained Linear Regression model.
                This is an estimate, not a guaranteed selling price.
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    NOIR ESTATE &nbsp; • &nbsp; PROPERTY VALUATION
    &nbsp; • &nbsp; MACHINE LEARNING PROJECT
</div>
""", unsafe_allow_html=True)
