import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="VoltWise AI",
    page_icon="⚡",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html,body,[class*="css"]{
font-family:'Inter',sans-serif;
}

.stApp{
background:#0D1117;
}

.block-container{
padding-top:1rem;
padding-left:3rem;
padding-right:3rem;
padding-bottom:2rem;
}

h1,h2,h3,h4,h5,h6,p{
color:white;
}

/* Metric Cards */

div[data-testid="stMetric"]{

background:#161B22;

padding:20px;

border-radius:18px;

border:1px solid #30363D;

box-shadow:0px 5px 20px rgba(0,0,0,0.3);

}

/* Metric Labels */

div[data-testid="stMetricLabel"]{

font-size:16px;

font-weight:600;

color:#9CA3AF;

}

/* Metric Values */

div[data-testid="stMetricValue"]{

font-size:34px;

font-weight:700;

color:#00D4FF;

}

/* Buttons */

div[data-testid="stButton"] > button{

width:100%;

height:60px;

border:none;

border-radius:15px;

background:#00D4FF !important;

color:#0D1117 !important;

font-size:20px !important;

font-weight:700 !important;

}

/* Divider */

hr{

border:1px solid #30363D;

}

</style>

""",unsafe_allow_html=True)

# ---------------- MODEL ----------------

class ANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.model=nn.Sequential(

            nn.Linear(4,6),

            nn.ReLU(),

            nn.Linear(6,6),

            nn.ReLU(),

            nn.Linear(6,1)

        )

    def forward(self,x):

        return self.model(x)

# ---------------- LOAD MODEL ----------------

MODEL_PATH=os.path.join(
    os.path.dirname(__file__),
    "best_model.pth"
)

@st.cache_resource
def load_model():

    model=ANN()

    state_dict=torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model

try:

    model=load_model()

    model_loaded=True

except Exception as e:

    model_loaded=False

    load_error=str(e)

# ---------------- FEATURE STATS ----------------

FEATURE_MEANS=np.array(
[19.6513,54.3058,1013.2590,73.3089],
dtype=np.float32
)

FEATURE_STDS=np.array(
[7.4523,12.7075,5.9388,14.5994],
dtype=np.float32
)

# ---------------- HEADER ----------------

st.markdown("""

<h1 style='text-align:center;color:#00D4FF;'>

⚡ VoltWise AI

</h1>

<h3 style='text-align:center;color:#9CA3AF;'>

Smart Power Plant Energy Predictor

</h3>

""",unsafe_allow_html=True)

st.write("")

# ---------------- TOP CARDS ----------------

col1,col2,col3=st.columns(3)

with col1:

    st.metric(
        "🟢 Model",
        "Online" if model_loaded else "Offline"
    )

with col2:

    st.metric(
        "🧠 Technology",
        "PyTorch ANN"
    )

with col3:

    st.metric(
        "📊 Dataset",
        "CCPP"
    )

st.divider()

# ---------------- INPUTS ----------------

st.subheader("⚙ Input Parameters")

col1,col2=st.columns(2)

with col1:

    at=st.slider(
        "🌡 Ambient Temperature (°C)",
        1.81,
        37.11,
        19.65
    )

    ap=st.slider(
        "🌍 Ambient Pressure (mbar)",
        992.89,
        1033.30,
        1013.26
    )

with col2:

    v=st.slider(
        "💨 Exhaust Vacuum (cm Hg)",
        25.36,
        81.56,
        54.31
    )

    rh=st.slider(
        "💧 Relative Humidity (%)",
        25.56,
        100.16,
        73.31
    )

st.write("")

# ---------------- PREDICTION ----------------

if model_loaded:

    if st.button("🚀 Predict Energy Output"):

        raw=np.array(
            [[at,v,ap,rh]],
            dtype=np.float32
        )

        scaled=(
            raw-FEATURE_MEANS
        )/FEATURE_STDS

        tensor=torch.tensor(
            scaled,
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction=model(
                tensor
            ).item()

        st.divider()

        st.subheader("⚡ Prediction")

        st.metric(
            "Predicted Electrical Output",
            f"{prediction:.2f} MW"
        )

        st.write("")

        st.subheader("📊 Current Input Values")

        c1,c2,c3,c4=st.columns(4)

        with c1:

            st.metric(
                "🌡 Temperature",
                f"{at:.2f} °C"
            )

        with c2:

            st.metric(
                "💨 Vacuum",
                f"{v:.2f}"
            )

        with c3:

            st.metric(
                "🌍 Pressure",
                f"{ap:.2f}"
            )

        with c4:

            st.metric(
                "💧 Humidity",
                f"{rh:.2f}%"
            )

        st.write("")

        st.subheader("📋 Input Summary")

        summary=pd.DataFrame({

            "Feature":[

                "Temperature",

                "Vacuum",

                "Pressure",

                "Humidity"

            ],

            "Value":[

                at,

                v,

                ap,

                rh

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

else:

    st.error(load_error)

# ---------------- FOOTER ----------------

st.markdown("""

<hr>

<center>

⚡ VoltWise AI | Built with Streamlit + PyTorch

</center>

""",unsafe_allow_html=True)