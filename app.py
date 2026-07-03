import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="EV Battery QC Prediction",
    page_icon="🔋",
    layout="wide"
)

# -------------------- LOAD FILES --------------------
model = load_model("EV_Battery_QC_ANN.keras")
preprocessor = joblib.load("preprocessor.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -------------------- CSS --------------------
st.markdown("""
<style>

.main{
background: linear-gradient(to right,#eef2f3,#d9e7ff);
}

.title{
    text-align:center;
    color:white;
    background:linear-gradient(90deg,#1565C0,#42A5F5);
    padding:20px;
    border-radius:15px;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    margin-top:-10px;
    margin-bottom:10px;
}

.prediction{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
}

</style>
""",unsafe_allow_html=True)

# -------------------- HEADER --------------------

st.markdown("""
<div class='title'>
🔋 EV Battery Cell Quality Prediction
</div>
<div class='subtitle'>
Artificial Neural Network | Streamlit Deployment
</div>
""",unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 Model", "ANN")

with col2:
    st.metric("🎯 Accuracy", "94.33%")

with col3:
    st.metric("📂 Classes", "3")

# -------------------- SIDEBAR --------------------

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png",width=120)

st.sidebar.title("About")

st.sidebar.success("Project Details")

st.sidebar.metric("Model Accuracy", "94.33%")

st.sidebar.write("""
### 🤖 Model
Artificial Neural Network

### 📂 Dataset
EV Battery QC Dataset

### 🎯 Target
QC Grade Prediction

### Classes

🟢 Grade A

🟡 Grade B

🔴 Scrap
""")


# -------------------- INPUTS --------------------

st.header("Enter Battery Details")

col1,col2=st.columns(2)

with col1:

    batch_id=st.text_input("Batch ID","BATCH_0001")

    production_line=st.selectbox(
        "Production Line",
        ["Line_1","Line_2","Line_3"]
    )

    shift=st.selectbox(
        "Shift",
        ["Morning","Evening","Night"]
    )

    supplier=st.selectbox(
        "Supplier",
        ["ChemCorp","LithioMat","VoltIndustries"]
    )

    ambient_temp=st.number_input(
        "Ambient Temperature (°C)",
        value=25.0
    )

    anode_overhang=st.number_input(
        "Anode Overhang (mm)",
        value=1.5
    )

with col2:

    electrolyte_volume=st.number_input(
        "Electrolyte Volume (ml)",
        value=5.0
    )

    internal_resistance=st.number_input(
        "Internal Resistance (mOhm)",
        value=15.0
    )

    capacity=st.number_input(
        "Capacity (mAh)",
        value=3000.0
    )

    retention=st.number_input(
        "Retention After 50 Cycles (%)",
        value=95.0
    )

    defect=st.selectbox(
        "Defect Type",
        [
            "Poor Retention",
            "High Internal Resistance",
            "Severe Capacity Fade",
            "Critical Resistance",
            "Short Circuit Risk (Overhang)",
            "Low Capacity"
        ]
    )

st.write("")

predict = st.button(
    "🚀 Predict Battery Quality",
    use_container_width=True
)

# -------------------- PREDICTION --------------------

if predict:

    input_df=pd.DataFrame({

        "Batch_ID":[batch_id],
        "Production_Line":[production_line],
        "Shift":[shift],
        "Supplier":[supplier],
        "Ambient_Temp_C":[ambient_temp],
        "Anode_Overhang_mm":[anode_overhang],
        "Electrolyte_Volume_ml":[electrolyte_volume],
        "Internal_Resistance_mOhm":[internal_resistance],
        "Capacity_mAh":[capacity],
        "Retention_50Cycle_Pct":[retention],
        "Defect_Type":[defect]

    })

    processed=preprocessor.transform(input_df)

    prediction=model.predict(processed)

    pred_class=np.argmax(prediction,axis=1)

    pred_label=label_encoder.inverse_transform(pred_class)[0]

    probabilities = prediction[0]

    confidence = np.max(probabilities)*100

    st.divider()

    if pred_label=="Grade A":
        st.balloons()

        color="#2E7D32"
        emoji="🟢"

    elif pred_label=="Grade B":

        color="#FB8C00"
        emoji="🟡"

    else:

        color="#C62828"
        emoji="🔴"

    st.markdown(f"""
    <div class='prediction'
    style='background-color:{color};color:white;'>

    {emoji}<br>

    Predicted Grade

    <br><br>

    {pred_label}

    </div>

    """,unsafe_allow_html=True)

    st.write("")

    st.subheader("Prediction Confidence")

    st.progress(float(confidence/100))

    st.success(f"Confidence : {confidence:.2f}%")
    
    if pred_label=="Grade A":
        st.success("🟢 Excellent Battery Cell")

    elif pred_label=="Grade B":
        
        st.warning("🟡 Moderate Quality Battery")

    else:
        st.error("🔴 Reject Battery Cell")
    st.write("")

    st.subheader("📊 Prediction Probability")

    prob_df = pd.DataFrame({

        "QC Grade": label_encoder.classes_,
        "Probability (%)": np.round(probabilities * 100, 2)

    })

    st.bar_chart(prob_df.set_index("QC Grade"))

    with st.expander("📋 View Battery Input Details"):

        st.dataframe(input_df,use_container_width=True)

st.markdown("""

---

<center>

### 🔋 EV Battery Cell Quality Prediction

Built using TensorFlow • Keras • Streamlit

Artificial Neural Network for Battery Quality Classification

</center>

""",unsafe_allow_html=True)