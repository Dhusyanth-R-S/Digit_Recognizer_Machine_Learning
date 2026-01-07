import streamlit as st
import numpy as np
import csv
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Data Collector",
    page_icon="📝",
    layout="wide"
)

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "real_streamlit_samples.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Create CSV if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"p{i}" for i in range(784)])

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "collector_canvas"

st.title("📝 Digit Data Collection App")
st.caption("Draw a digit → enter the correct label → save")

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

true_label = st.number_input(
    "Which digit did you draw?",
    min_value=0,
    max_value=9,
    step=1
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Save Sample"):
        if canvas.image_data is None:
            st.warning("Draw something first.")
        else:
            img = canvas.image_data[:, :, :3].mean(axis=2).astype(np.uint8)

            digit_28 = Image.fromarray(img).resize(
                (28, 28),
                resample=Image.BILINEAR
            )

            X = np.array(digit_28, dtype=np.uint8).reshape(-1)

            with open(DATA_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([true_label] + X.tolist())

            st.success("Sample saved ✔️")

with col2:
    if st.button("Clear"):
        st.session_state.canvas_key = f"collector_{np.random.randint(1_000_000)}"
        st.rerun()
