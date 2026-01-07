# ✍️ Handwritten Digit Recognizer  
### Classical Machine Learning • Streamlit Deployment
 
Users can draw digits on an interactive canvas and receive **real-time predictions**.

---

## 🚀 Project Overview

This project demonstrates a complete **end-to-end ML workflow** — from model training to live deployment — using a **Logistic Regression classifier** wrapped inside a **Scikit-learn Pipeline**.

Instead of deep learning, this project intentionally uses **classical ML** to highlight:
- strong fundamentals  
- preprocessing correctness  
- deployment reliability  
- CPU-efficient inference  

---

## ⭐ Key Features & Speciality

- ✅ **Classical ML (Logistic Regression)** — fast, lightweight, interpretable  
- ✅ **Pipeline-based architecture** (StandardScaler + model)  
- ✅ Prevents **training–inference preprocessing mismatch**  
- ✅ **Interactive drawing canvas** for user input  
- ✅ Clean UI with **instant predict & clear** (no page reloads)  
- ✅ Carefully aligned **stroke width, resolution, and normalization**  
- ✅ Deployed directly on **Streamlit Cloud** (no Docker / backend)

---

## 🧠 Model Details

- **Algorithm:** Logistic Regression (multiclass)
- **Framework:** Scikit-learn
- **Input Shape:** 28 × 28 grayscale image
- **Features:** 784 flattened pixel values
- **Preprocessing:**
  - Resize to 28×28  
  - Pixel normalization  
  - Standard scaling handled inside the pipeline  

---

## 🖥️ Tech Stack

- **Python**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **streamlit-drawable-canvas**
- **Joblib**

---

## 📂 Project Structure

