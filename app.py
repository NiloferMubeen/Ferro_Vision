import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Ferro Vision", layout="wide")

# ---------- File locations (all relative to this app.py) ----------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE, "train_data.csv")
ANN_MODEL = os.path.join(BASE, "ann_model.h5")       # ANN for RUL
CNN_MODEL = os.path.join(BASE, "model.h5")           # MobileNet (ImageNet) for defects
IMG_DIR = os.path.join(BASE, "casting_data")

# Same 25 input columns, in the same order, used to train the ANN
FEATURES = ["cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [f"sensor_{i}" for i in range(1, 22)]

# Same class order as flow_from_directory in the CNN notebook
CLASS_NAMES = ["defective", "normal"]

# ---------- Sidebar menu ----------
with st.sidebar:
    page = option_menu(
        "Ferro Vision",
        ["EDA", "Regression", "Image Classification"],
        icons=["bar-chart", "speedometer2", "camera"],
    )


# =====================================================
# PAGE 1 - EDA
# =====================================================
if page == "EDA":
    st.title("Exploratory Data Analysis")
    dataset = st.selectbox("Choose a dataset", ["Turbofan sensor data", "Casting image data"])

    # ---------- Turbofan ----------
    if dataset == "Turbofan sensor data":
        df = pd.read_csv(DATA_CSV)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Engines", df["engine_id"].nunique())
        c3.metric("Columns", df.shape[1])

        st.subheader("Data preview")
        st.dataframe(df.head(100), width="stretch")

        st.subheader("Summary statistics")
        st.dataframe(df.describe().T, width="stretch")

        st.subheader("RUL distribution")
        st.plotly_chart(px.histogram(df, x="RUL", nbins=50), width="stretch")

        st.subheader("Sensor trend over cycles")
        col1, col2 = st.columns(2)
        engine = col1.selectbox("Engine ID", sorted(df["engine_id"].unique()))
        sensors = col2.multiselect("Sensors", FEATURES[4:], default=["sensor_2", "sensor_3"])
        engine_df = df[df["engine_id"] == engine]
        if sensors:
            st.plotly_chart(px.line(engine_df, x="cycle", y=sensors), width="stretch")

        st.subheader("Correlation of each feature with RUL")
        corr = df[FEATURES + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
        st.plotly_chart(px.bar(corr, labels={"value": "correlation", "index": "feature"}), width="stretch")

    # ---------- Casting images ----------
    else:
        # class balance
        rows = []
        for split in ["train", "test"]:
            for cls in CLASS_NAMES:
                folder = os.path.join(IMG_DIR, split, cls)
                rows.append({"split": split, "class": cls, "images": len(os.listdir(folder))})
        counts = pd.DataFrame(rows)

        st.subheader("Class balance")
        st.plotly_chart(px.bar(counts, x="split", y="images", color="class", barmode="group", text="images"),
                        width="stretch")

        st.subheader("Sample images")
        for cls in CLASS_NAMES:
            st.write(f"**{cls.title()}**")
            folder = os.path.join(IMG_DIR, "train", cls)
            files = sorted(os.listdir(folder))[:5]
            cols = st.columns(5)
            for col, f in zip(cols, files):
                col.image(os.path.join(folder, f), width="stretch")


# =====================================================
# PAGE 2 - REGRESSION (ANN)
# =====================================================
elif page == "Regression":
    st.title("RUL Prediction (ANN)")
    df = pd.read_csv(DATA_CSV)

    mode = st.radio("Input method", ["Pick a row from the dataset", "Enter values manually"], horizontal=True)
    actual = None

    if mode == "Pick a row from the dataset":
        col1, col2 = st.columns(2)
        engine = col1.selectbox("Engine ID", sorted(df["engine_id"].unique()))
        cycles = df[df["engine_id"] == engine]["cycle"].tolist()
        cycle = col2.selectbox("Cycle", cycles)
        row = df[(df["engine_id"] == engine) & (df["cycle"] == cycle)].iloc[0]
        values = row[FEATURES].astype(float).to_dict()
        actual = row["RUL"]
        st.dataframe(row[FEATURES].to_frame().T, width="stretch")
    else:
        cols = st.columns(5)
        values = {}
        for i, name in enumerate(FEATURES):
            values[name] = cols[i % 5].number_input(name, value=float(df[name].median()))

    if st.button("Predict RUL"):
        # The ANN was trained on scaled data, so rebuild the same scaler:
        # same engine split (random_state=42) and fit on the training engines only
        engine_ids = df["engine_id"].unique()
        train_engines, _ = train_test_split(engine_ids, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(df[df["engine_id"].isin(train_engines)][FEATURES])

        X = scaler.transform(pd.DataFrame([values])[FEATURES])
        model = load_model(ANN_MODEL, compile=False)
        predicted = max(0.0, float(model.predict(X, verbose=0)[0][0]))

        c1, c2 = st.columns(2)
        c1.metric("Predicted RUL (cycles)", round(predicted, 1))
        if actual is not None:
            c2.metric("Actual RUL (cycles)", int(actual))


# =====================================================
# PAGE 3 - IMAGE CLASSIFICATION (MobileNet)
# =====================================================
else:
    st.title("Casting Defect Detection")
    st.write("Model: **MobileNet** (pretrained on ImageNet, fine-tuned on casting images) - test accuracy 98.88%")

    file = st.file_uploader("Upload a casting component image", type=["jpg", "jpeg", "png"])

    if file is not None:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded image", width=300)

        # Same preprocessing as training: 224x224 and pixel values / 255
        x = np.array(img.resize((224, 224))) / 255.0
        x = np.expand_dims(x, axis=0)

        model = load_model(CNN_MODEL, compile=False)
        probs = model.predict(x, verbose=0)[0]
        label = CLASS_NAMES[int(np.argmax(probs))]
        confidence = float(np.max(probs)) * 100

        if label == "defective":
            st.error(f"DEFECTIVE - confidence {confidence:.2f}%")
        else:
            st.success(f"NORMAL - confidence {confidence:.2f}%")

        for name, p in zip(CLASS_NAMES, probs):
            st.write(f"{name.title()}: {p * 100:.2f}%")
            st.progress(float(p))