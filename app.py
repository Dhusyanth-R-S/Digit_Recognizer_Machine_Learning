import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("digit_lr_model.pkl")

model = load_model()

if "show_canvas" not in st.session_state:
    st.session_state.show_canvas = False
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"

st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.hint {
    text-align: center;
    font-size: 18px;
    color: #cfcfcf;
    margin-bottom: 20px;
}

.draw-wrap {
    display: flex;
    justify-content: center;
}

.draw-wrap button {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    font-size: 30px !important;
    font-weight: 800;
}

.prediction {
    font-size: 80px;
    font-weight: 900;
    color: #2ecc71;
    animation: pop 0.4s ease-out;
}

@keyframes pop {
    from { transform: translateY(-15px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

.warning {
    font-size: 28px;
    font-weight: 700;
    color: #f39c12;
}

.clear-btn button {
    height: 36px;
    width: 90px;
    font-size: 14px !important;
}

.predict-btn button {
    height: 78px;
    width: 100%;
    font-size: 28px !important;
    font-weight: 900;
    background-color: #7bed9f !important;
    color: #000 !important;
}

@media (min-width: 768px) {
    .desktop-layout {
        display: flex;
        gap: 40px;
        align-items: center;
    }
    .prediction {
        font-size: 96px;
    }
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.show_canvas:
    st.markdown("<div class='hint'>Click draw to make me recognize what you draw!</div>", unsafe_allow_html=True)
    st.markdown("<div class='draw-wrap'>", unsafe_allow_html=True)
    if st.button("DRAW"):
        st.session_state.show_canvas = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='desktop-layout'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.session_state.prediction is not None:
            if isinstance(st.session_state.prediction, int):
                st.markdown(f"<div class='prediction'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning'>{st.session_state.prediction}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='prediction'>&nbsp;</div>", unsafe_allow_html=True)

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

        b1, b2 = st.columns([1, 3])

        with b1:
            st.markdown("<div class='clear-btn'>", unsafe_allow_html=True)
            if st.button("Clear"):
                st.session_state.canvas_key = f"canvas_{np.random.randint(1_000_000)}"
                st.session_state.prediction = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with b2:
            st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
            if st.button("Predict"):
                if canvas.image_data is None:
                    st.stop()
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
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if st.session_state.prediction is not None and isinstance(st.session_state.prediction, int):
            st.markdown(f"<div class='prediction'>{st.session_state.prediction}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
