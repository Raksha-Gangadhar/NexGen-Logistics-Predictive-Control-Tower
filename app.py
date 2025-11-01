# app.py
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="NexGen Predictive Control Tower", layout="wide")

# ---------- Load data & model ----------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_all.csv")
    # Defensive types
    for col in df.columns:
        if col.endswith("_Date"):
            with pd.option_context('mode.chained_assignment', None):
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_resource
def load_model():
    bundle = joblib.load("model_bundle.pkl")
    return bundle["model"], bundle["features"]

df = load_data()
model, model_features = load_model()

# ---------- Helpers ----------
def safe_mean(series):
    try:
        return pd.to_numeric(series, errors="coerce").dropna().mean()
    except Exception:
        return np.nan

def get_col(*names):
    """Return the first existing column from names or a full-NaN series."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([np.nan] * len(df))

# KPI columns that might exist with variant names
delay_hours = get_col("Delay_Hours")
is_delayed  = get_col("Is_Delayed")
status      = get_col("Delivery_Status")
rating      = get_col("Customer_Rating", "Rating")
cost        = get_col("Total_Cost_INR")
order_value = get_col("Order_Value_INR")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
dest = st.sidebar.multiselect("Destination", sorted(df["Destination"].dropna().unique().tolist()), default=None)
orig = st.sidebar.multiselect("Origin", sorted(df["Origin"].dropna().unique().tolist()), default=None)
prio = st.sidebar.multiselect("Priority", sorted(df["Priority"].dropna().unique().tolist()), default=None)

mask = pd.Series(True, index=df.index)
if dest: mask &= df["Destination"].isin(dest)
if orig: mask &= df["Origin"].isin(orig)
if prio: mask &= df["Priority"].isin(prio)

view = df.loc[mask].copy()

# ---------- KPIs ----------
total_orders = len(view)
sla_rate = None
if "Is_Delayed" in view.columns:
    sla_rate = (1 - pd.to_numeric(view["Is_Delayed"], errors="coerce").fillna(0)).mean()
elif "Delivery_Status" in view.columns:
    sla_rate = view["Delivery_Status"].astype(str).str.lower().eq("delivered_on_time").mean()

avg_delay = pd.to_numeric(view.get("Delay_Hours", pd.Series(dtype=float)), errors="coerce")
avg_delay = avg_delay[avg_delay > 0].mean() if not avg_delay.empty else np.nan

avg_cost = safe_mean(view.get("Total_Cost_INR", pd.Series(dtype=float)))
avg_rating = safe_mean(view.get("Customer_Rating", view.get("Rating", pd.Series(dtype=float))))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orders in View", f"{total_orders}")
c2.metric("SLA (On-time %)", f"{(sla_rate*100):.1f}%" if sla_rate is not None and pd.notna(sla_rate) else "—")
c3.metric("Avg Delay (hrs, >0)", f"{avg_delay:.1f}" if pd.notna(avg_delay) else "—")
c4.metric("Avg Cost / Order (₹)", f"{avg_cost:,.0f}" if pd.notna(avg_cost) else "—")

st.markdown("---")

# ---------- Prediction section ----------
st.subheader("Delay Risk Prediction")

# Ensure the model feature frame exists with the right columns
X = pd.DataFrame(index=view.index)
for col in model_features:
    if col in view.columns:
        X[col] = view[col]
    else:
        # create missing columns as NaN so the pipeline imputers can handle them
        X[col] = np.nan

# Predict
proba = model.predict_proba(X)[:, 1]
view["Delay_Prob"] = proba

# Threshold + table
th = st.slider("Flag threshold (delay probability)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
flagged = view[view["Delay_Prob"] >= th].copy()

# Useful columns to show
show_cols = [c for c in [
    "Order_ID", "Priority", "Origin", "Destination", "Route",
    "Distance_KM", "Traffic_Delay_Minutes", "Weather_Impact",
    "Assigned_Vehicle_Type", "Assigned_Vehicle_FuelEff", "Assigned_Vehicle_Age",
    "Total_Cost_INR", "Customer_Rating", "Delay_Hours", "Is_Delayed", "Delivery_Status",
    "CO2_Est_kg", "Order_Value_INR", "Delay_Prob"
] if c in view.columns]

st.write(f"Orders predicted at risk (≥ {th:.2f}): **{len(flagged)}**")
st.dataframe(flagged[show_cols].sort_values("Delay_Prob", ascending=False), use_container_width=True)

csv = flagged[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Download flagged orders (CSV)", csv, file_name="flagged_orders.csv", mime="text/csv")

st.markdown("---")

# ---------- Simple prescriptions (rule-based v1) ----------
st.subheader("Prescriptions (Rule-based v1)")

def prescribe(row):
    recs = []
    # If express and long distance, suggest faster vehicle
    if str(row.get("Priority","")).lower() == "express" and pd.to_numeric(row.get("Distance_KM", np.nan), errors="coerce") > 300:
        if str(row.get("Assigned_Vehicle_Type","")).lower().find("bike") >= 0:
            recs.append("Swap to van/truck for long Express.")
    # If traffic delay high
    if pd.to_numeric(row.get("Traffic_Delay_Minutes", np.nan), errors="coerce") >= 45:
        recs.append("Avoid high-traffic route; try alternate lane.")
    # Weather impact
    if str(row.get("Weather_Impact","")).lower() in {"high","severe"}:
        recs.append("Weather risk high: add buffer or choose sheltered route.")
    # Cost high but short distance
    dist = pd.to_numeric(row.get("Distance_KM", np.nan), errors="coerce")
    cost = pd.to_numeric(row.get("Total_Cost_INR", np.nan), errors="coerce")
    if pd.notna(dist) and pd.notna(cost) and dist < 50 and cost > np.nanmedian(df.get("Total_Cost_INR", pd.Series(dtype=float))):
        recs.append("Short haul but costly: check carrier/fuel efficiency.")
    return "; ".join(recs) if recs else "No change"

if not flagged.empty:
    flagged["Prescription"] = flagged.apply(prescribe, axis=1)
    pres_cols = [c for c in ["Order_ID","Priority","Origin","Destination","Assigned_Vehicle_Type","Distance_KM","Traffic_Delay_Minutes","Weather_Impact","Total_Cost_INR","Delay_Prob","Prescription"] if c in flagged.columns]
    st.dataframe(flagged[pres_cols].sort_values("Delay_Prob", ascending=False), use_container_width=True)
    pres_csv = flagged[pres_cols].to_csv(index=False).encode("utf-8")
    st.download_button("Download prescriptions (CSV)", pres_csv, file_name="prescriptions.csv", mime="text/csv")
else:
    st.info("No orders cross the current risk threshold.")
