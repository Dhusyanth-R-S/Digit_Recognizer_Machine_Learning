import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("digit_lr_model.pkl")

model = load_model()

# Initialize Session States
if "show_canvas" not in st.session_state:
    st.session_state.show_canvas = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"

# Custom CSS
st.markdown("""
<style>
/* 1. Landing Page Styles */
.draw-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 80px;
    text-align: center;
}

.draw-text {
    font-size: 22px;
    margin-bottom: 25px;
    color: white;
}

/* 2. Target ONLY the DRAW button using a specific div class */
/* This prevents the Clear/Predict buttons from changing */
.draw-container div.stButton > button {
    width: 180px !important;
    height: 180px !important;
    border-radius: 50% !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    background-color: #2ecc71 !important;
    color: white !important;
    border: none !important;
}

/* 3. Normal buttons for the canvas page (just in case) */
[data-testid="stHorizontalBlock"] button {
    width: auto !important;
    height: auto !important;
    border-radius: 4px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# --- FIRST PAGE (Landing) ---
if not st.session_state.show_canvas:
    st.markdown("<div class='draw-container'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='draw-text'>Click draw to make me recognize what you draw!</div>",
        unsafe_allow_html=True
    )
    if st.button("DRAW"):
        st.session_state.show_canvas = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECOND PAGE (Canvas & Prediction) ---
else:
    # Display Prediction Result
    if st.session_state.prediction is not None:
        if isinstance(st.session_state.prediction, (int, np.integer)):
            st.markdown(
                f"""
                <div style="text-align:center; color:white; font-size:24px; margin-bottom: 10px;">
                    You drew: 
                    <span style="font-size:55px; font-weight:900; color:#2ecc71; display:block;">
                        {st.session_state.prediction}
                    </span>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='text-align:center;color:#f39c12;'>{st.session_state.prediction}</h2>",
                unsafe_allow_html=True
            )

    # Drawing Canvas
    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key
    )

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Clear", use_container_width=True):
            st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
            st.session_state.prediction = None
            st.rerun()

    with c2:
        if st.button("Predict", use_container_width=True):
            if canvas.image_data is not None:
                img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)
                digit_28 = Image.fromarray(img).resize((28, 28), resample=Image.BILINEAR)
                X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)

                probs = model.predict_proba(X)[0]
                best_prob = probs.max()
                pred = probs.argmax()

                if best_prob < 0.6:
                    st.session_state.prediction = "Please redraw"
                else:
                    st.session_state.prediction = int(pred)
                st.rerun()