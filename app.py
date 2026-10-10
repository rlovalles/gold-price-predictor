import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Gold Price Predictor", layout="wide")

# Streamlit dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #FFD700; }
    .stMetric { border: 1px solid #FFD700; border-radius: 8px; padding: 10px; }
    .block-container { padding-top: 2rem; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# Data loading
df = pd.read_csv("XAU_15m_data.csv", sep=";")

# Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Target variable engineering (1 = price goes up, 0 = price goes down)
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Date parsing
df["Date"] = pd.to_datetime(df["Date"])

# Return calculation
df["Return"] = df["Close"].pct_change()

df.dropna(inplace=True)

# Latest volume for prediction
latest_volume = df["Volume"].iloc[-1]

# Feature selection
features = ["Open", "High", "Low", "Close", "Volume"]
X = df[features]
y = df["Target"]

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Macchine learning (logistic regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sidebar (prediction widget)
with st.sidebar:
    st.markdown("## Make a Prediction")
    st.write("Enter current gold price values to reveal a direction!")
    open_val = st.number_input("Open", min_value=0.0, value=1800.0, step=0.1)
    high_val = st.number_input("High", min_value=0.0, value=1805.0, step=0.1)
    low_val = st.number_input("Low", min_value=0.0, value=1795.0, step=0.1)
    close_val = st.number_input("Close", min_value=0.0, value=1802.0, step=0.1)
    # adding errors if high is lower than low and low is higher than high
    if st.button("Predict Direction"):
        if high_val < low_val:
            st.error("High must be greater than Low.")
        elif open_val < low_val or open_val > high_val:
            st.error("Open must be between Low and High.")
        elif close_val < low_val or close_val > high_val:
            st.error("Close must be between Low and High.")
        else:
            input_data = pd.DataFrame(
                [[open_val, high_val, low_val, close_val, latest_volume]],
                columns=["Open", "High", "Low", "Close", "Volume"]
            )
            prediction = model.predict(input_data)[0]
            if prediction == 1:
                st.success("Price predicted to go UP")
            else:
                st.error("Price predicted to go DOWN")

# Dashboard header
st.title("Gold Price Predictor")
st.write("Predicting whether gold prices (XAU/USD) will go up or down in the next 15-minute interval.")

st.metric("Logistic Regression Model Accuracy", f"{accuracy:.2%}", help="Accuracy score after training on 80% of the dataset and testing on the remaining 20%. This value is fixed and does not change with the date filter.")

st.markdown("---")

# Interactive query (date range filter)
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

st.subheader("Explore the Data")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

filtered_df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
st.write(f"{len(filtered_df):,} records found from {start_date} to {end_date}.")
st.dataframe(filtered_df[["Date", "Open", "High", "Low", "Close", "Volume"]].head(50), use_container_width=True)

st.markdown("---")

# Visualization 1 (line chart)
st.subheader("XAU/USD Closing Price Over Time")
fig1, ax1 = plt.subplots(figsize=(12, 3))
fig1.patch.set_facecolor("#0E0E0E")
ax1.set_facecolor("#0E0E0E")
ax1.plot(filtered_df["Date"].values, filtered_df["Close"].values, color="#FFD700", linewidth=1)
ax1.set_xlabel("Date", color="#F0F0F0")
ax1.set_ylabel("Close Price (USD)", color="#F0F0F0")
ax1.tick_params(colors="#F0F0F0")
ax1.spines["bottom"].set_color("#333333")
ax1.spines["left"].set_color("#333333")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Visualization 2 (histogram)
st.subheader("Distribution of 15-Minute Returns")
fig2, ax2 = plt.subplots(figsize=(12, 3))
fig2.patch.set_facecolor("#0E0E0E")
ax2.set_facecolor("#0E0E0E")
ax2.hist(filtered_df["Return"].dropna(), bins=100, color="#FFD700", edgecolor="none")
ax2.set_xlabel("Return", color="#F0F0F0")
ax2.set_ylabel("Frequency", color="#F0F0F0")
ax2.tick_params(colors="#F0F0F0")
ax2.spines["bottom"].set_color("#333333")
ax2.spines["left"].set_color("#333333")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)

st.markdown("---")

# Visualization 3 (confusion matrix)
st.subheader("Confusion Matrix")
_, col_cm, _ = st.columns([1.2, 2, 1.2])
with col_cm:
    fig3, ax3 = plt.subplots(figsize=(3.5, 3))
    fig3.patch.set_facecolor("#0E0E0E")
    ax3.set_facecolor("#0E0E0E")
    for i in range(2):
        for j in range(2):
            ax3.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#1A1A1A", ec="#333333", lw=1))
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="#FFD700", fontsize=14, fontweight="bold")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Down (0)", "Up (1)"], color="#F0F0F0", fontsize=9)
    ax3.set_yticklabels(["Down (0)", "Up (1)"], color="#F0F0F0", fontsize=9)
    ax3.set_xlabel("Predicted", color="#F0F0F0", fontsize=10, labelpad=20)
    ax3.set_ylabel("Actual", color="#F0F0F0", fontsize=10, labelpad=20)
    ax3.tick_params(axis='x', pad=12)
    ax3.tick_params(axis='y', pad=12)
    ax3.set_xlim(-0.5, 1.5)
    ax3.set_ylim(-0.5, 1.5)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3)

# footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#666; font-size:0.8rem;'>Built by Rebecca Ovalles for Ovalles Trading · 2026</div>", unsafe_allow_html=True)