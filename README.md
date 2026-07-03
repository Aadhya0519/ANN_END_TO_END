# 🔋 EV Battery Cell Quality Prediction using ANN

## 📌 Project Overview

This project predicts the quality grade of electric vehicle battery cells using an Artificial Neural Network (ANN). It classifies battery cells into one of three quality categories based on manufacturing and performance parameters.

---

## 🎯 Objective

To build and deploy an ANN model that predicts battery quality as:

- 🟢 Grade A
- 🟡 Grade B
- 🔴 Scrap

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

## 📂 Project Structure

```
ANN_END_TO_END/
│── app.py
│── requirements.txt
│── EV_Battery_QC_ANN.keras
│── preprocessor.pkl
│── label_encoder.pkl
│── feature_names.pkl
│── README.md
```

---

## 🚀 Features

- Battery quality prediction using ANN
- Data preprocessing with ColumnTransformer
- One-Hot Encoding and MinMax Scaling
- Interactive Streamlit web application
- User-friendly interface

---

## 📈 Model Performance

- Training Accuracy: **95.64%**
- Validation Accuracy: **94.87%**
- Test Accuracy: **94.33%**

---

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Manisharaj K**