import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Part C: Data loading
df = pd.read_csv("XAU_15m_data.csv", sep=";")

# Part C: Data cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Part C: Target variable engineering (1 = price goes up, 0 = price goes down next 15 minutes)
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df.dropna(inplace=True)

# Part C: Date parsing
df["Date"] = pd.to_datetime(df["Date"])

# Part C: Return calculation
df["Return"] = df["Close"].pct_change()

# Part C: Feature selection
features = ["Open", "High", "Low", "Close", "Volume"]
X = df[features]
y = df["Target"]

# Part C: Train and test split (80/20, no shuffle to preserve time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Part C: Machine learning (logistic regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Part C: Accuracy evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Part C: Streamlit dashboard
st.title("Gold Price Direction Predictor")
st.write("Predicting whether XAU/USD will go up or down in the next 15-minute period.")

st.subheader("Model Accuracy")
st.metric("Accuracy", f"{accuracy:.2%}")

# Part C: Interactive query (date range filter)
st.subheader("Explore the Data")
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

filtered_df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]
st.write(f"Showing {len(filtered_df):,} records from {start_date} to {end_date}")
st.dataframe(filtered_df[["Date", "Open", "High", "Low", "Close", "Volume", "Target"]].head(20))

# Part C: Visualization 1 (line chart)
st.subheader("XAU/USD Closing Price Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 3))
ax1.plot(filtered_df["Date"].values, filtered_df["Close"].values, color="gold", linewidth=1)
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price (USD)")
ax1.set_title("Gold Closing Price")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# Part C: Visualization 2 (histogram)
st.subheader("Distribution of 15-Minute Returns")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.hist(filtered_df["Return"].dropna(), bins=100, color="steelblue", edgecolor="none")
ax2.set_xlabel("Return")
ax2.set_ylabel("Frequency")
ax2.set_title("Distribution of 15-Minute Price Returns")
st.pyplot(fig2)

# Part C: Visualization 3 (Confusion matrix)
st.subheader("Confusion Matrix")
fig3, ax3 = plt.subplots(figsize=(4, 3))
ax3.imshow(cm, interpolation="nearest", cmap="Blues")
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(["Down (0)", "Up (1)"])
ax3.set_yticklabels(["Down (0)", "Up (1)"])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
ax3.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
st.pyplot(fig3)

# Part C: Security features
st.subheader("Security Notes")
st.info(
    "This application does not collect or store any user data. "
    "All data is loaded locally from a static CSV file. "
    "No authentication is required as this is a read-only decision-support tool."
)

# Part C: Monitoring and maintenance
st.subheader("Monitoring and Maintenance")
st.write(f"**Dataset size:** {len(df):,} records")
st.write(f"**Date range covered:** {df['Date'].min().date()} to {df['Date'].max().date()}")
st.write(f"**Model type:** Logistic Regression (scikit-learn)")
st.write("**Recommended maintenance:** Retrain the model periodically as new price data becomes available to maintain prediction relevance.")