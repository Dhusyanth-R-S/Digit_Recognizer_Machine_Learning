import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Page configuration
st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

@st.cache_resource
def load_model():
    # Ensure this filename matches your joblib dump
    return joblib.load("digit_lr_model.pkl")

model = load_model()

# Initialize Session States
if "show_canvas" not in st.session_state:
    st.session_state.show_canvas = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"

# Custom CSS for the Large Circular Button and Layout
st.markdown("""
<style>
/* Center draw section */
.draw-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 70vh; /* Adjusts vertical position */
    text-align: center;
}

/* Text above the button */
.draw-text {
    font-size: clamp(20px, 5vw, 28px);
    margin-bottom: 40px;
    font-weight: 500;
    color: white;
}

/* BIG circular responsive button */
div.stButton > button {
    width: clamp(250px, 60vw, 400px) !important;
    height: clamp(250px, 60vw, 400px) !important;
    border-radius: 50% !important;
    font-size: clamp(36px, 8vw, 54px) !important;
    font-weight: 900 !important;
    background-color: #2ecc71 !important;
    color: white !important;
    border: 8px solid #27ae60 !important;
    box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
}

/* Hover effect */
div.stButton > button:hover {
    transform: scale(1.05) !important;
    border-color: #ffffff !important;
    box-shadow: 0 15px 25px rgba(46, 204, 113, 0.4) !important;
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
                <div style="text-align:center; color:white; font-size:28px; font-weight:400; margin-bottom: 20px;">
                    You drew: 
                    <span style="font-size:70px; font-weight:900; color:#2ecc71; display:block;">
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

    # Center the Canvas
    col_canvas, _ = st.columns([1, 1]) # Placeholder columns can help centering if needed
    
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

    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("Clear"):
            st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
            st.session_state.prediction = None
            st.rerun()

    with c2:
        if st.button("Predict"):
            if canvas.image_data is not None:
                # Preprocessing
                img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)
                digit_28 = Image.fromarray(img).resize((28, 28), resample=Image.BILINEAR)
                X = np.array(digit_28, dtype=np.uint8).reshape(1, -1)

                # Inference
                probs = model.predict_proba(X)[0]
                best_prob = probs.max()
                pred = probs.argmax()

                # Logic check for clear drawings
                if best_prob < 0.6:
                    st.session_state.prediction = "Please redraw"
                else:
                    st.session_state.prediction = int(pred)

                st.rerun()
            else:
                st.warning("Please draw something first!")