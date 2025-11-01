import pandas as pd
import numpy as np
from typing import Optional

# ---------- Load ----------
orders   = pd.read_csv("orders.csv")
perf     = pd.read_csv("delivery_performance.csv")
routes   = pd.read_csv("routes_distance.csv")
costs    = pd.read_csv("cost_breakdown.csv")
feedback = pd.read_csv("customer_feedback.csv")
fleet    = pd.read_csv("vehicle_fleet.csv")
inv      = pd.read_csv("warehouse_inventory.csv")

# ---------- Clean / derive labels ----------
# Delivery delay label
if {"Promised_Delivery_Days","Actual_Delivery_Days"}.issubset(perf.columns):
    perf["Delay_Hours"] = (perf["Actual_Delivery_Days"] - perf["Promised_Delivery_Days"]) * 24.0
else:
    # If they were hours already, try those:
    if {"Promised_Delivery_Hours","Actual_Delivery_Hours"}.issubset(perf.columns):
        perf["Delay_Hours"] = perf["Actual_Delivery_Hours"] - perf["Promised_Delivery_Hours"]
    else:
        perf["Delay_Hours"] = np.nan

perf["Is_Delayed"] = perf["Delay_Hours"] > 0

# ---------- Base merge on Order_ID ----------
df = (
    orders.merge(perf, on="Order_ID", how="left")
          .merge(routes, on="Order_ID", how="left")
          .merge(costs,  on="Order_ID", how="left")
          .merge(feedback, on="Order_ID", how="left")
)

# ---------- Inventory assignment (use per-category best/nearest warehouse) ----------
# For each Product_Category, pick the warehouse row with the highest stock (tie-breaker: lowest storage cost)
inv_ranked = (
    inv.sort_values(["Product_Category","Current_Stock_Units","Storage_Cost_per_Unit"],
                    ascending=[True, False, True])
       .groupby("Product_Category", as_index=False)
       .first()
       .rename(columns={
           "Warehouse_ID":"Assigned_Warehouse_ID",
           "Location":"Assigned_Warehouse_Location",
           "Current_Stock_Units":"Assigned_Warehouse_Stock",
           "Reorder_Level":"Assigned_Warehouse_Reorder_Level",
           "Storage_Cost_per_Unit":"Assigned_Warehouse_Storage_Cost",
           "Last_Restocked_Date":"Assigned_Warehouse_Last_Restocked"
       })
)
df = df.merge(inv_ranked, on="Product_Category", how="left")

# ---------- Vehicle assignment (turn fleet into a prescriptive join) ----------
# We don't have Vehicle_ID in orders, so we "assign" the best available vehicle by rules:
#  - match vehicle type to product category needs
#  - if Priority == Express, allow bikes/express types
#  - prefer Status == 'Available' if present
#  - rank by higher fuel efficiency, lower age
fleet2 = fleet.copy()
fleet2.columns = [c.strip() for c in fleet2.columns]
for col in ["Vehicle_Type","Status"]:
    if col not in fleet2.columns:
        fleet2[col] = ""

if "Fuel_Efficiency_KM_per_L" not in fleet2.columns: fleet2["Fuel_Efficiency_KM_per_L"] = np.nan
if "Age_Years" not in fleet2.columns: fleet2["Age_Years"] = np.nan

# Simple need map (regex-like tokens) by product category
need_map = {
    "Food & Beverage": "refriger",             # refrigerated units
    "Healthcare": "refriger",
    "Electronics": "truck|van",
    "Industrial goods": "truck",
    "Books": "van",
    "Fashion": "van|bike",
    "Home Goods": "truck|van",
}

def pick_vehicle(row) -> Optional[str]:
    cat = str(row.get("Product_Category", "")).strip()
    pri = str(row.get("Priority", "")).strip().lower()
    origin = str(row.get("Origin","")).strip()

    pat = None
    for k, v in need_map.items():
        if k.lower() == cat.lower():
            pat = v
            break
    # Express can also consider bikes/express
    if "express" in pri:
        pat = "bike|express|van" if pat is None else f"{pat}|bike|express"

    candidates = fleet2.copy()
    # Prefer available
    if "Status" in candidates.columns and "Available" in candidates["Status"].astype(str).unique():
        candidates = pd.concat([
            candidates[candidates["Status"].astype(str).str.lower().eq("available")],
            candidates[~candidates["Status"].astype(str).str.lower().eq("available")]
        ], ignore_index=True)

    if pat is not None and "Vehicle_Type" in candidates.columns:
        mask = candidates["Vehicle_Type"].astype(str).str.lower().str.contains(pat, regex=True)
        cands = candidates[mask]
        if cands.empty:
            cands = candidates
    else:
        cands = candidates

    # Rank: higher fuel efficiency, then newer vehicle (lower age)
    cands = cands.copy()
    if "Fuel_Efficiency_KM_per_L" in cands.columns:
        cands["__fe"] = pd.to_numeric(cands["Fuel_Efficiency_KM_per_L"], errors="coerce")
    else:
        cands["__fe"] = np.nan
    if "Age_Years" in cands.columns:
        cands["__age"] = pd.to_numeric(cands["Age_Years"], errors="coerce")
    else:
        cands["__age"] = np.nan

    cands = cands.sort_values(["__fe","__age"], ascending=[False, True])
    return None if cands.empty else cands.iloc[0]["Vehicle_ID"]

df["Assigned_Vehicle_ID"] = df.apply(pick_vehicle, axis=1)

# Merge back fleet attributes using Assigned_Vehicle_ID
veh_cols_map = {
    "Vehicle_ID":"Assigned_Vehicle_ID",
    "Vehicle_Type":"Assigned_Vehicle_Type",
    "Capacity_KG":"Assigned_Vehicle_Capacity_KG",
    "Fuel_Efficiency_KM_per_L":"Assigned_Vehicle_FuelEff",
    "Current_Location":"Assigned_Vehicle_Current_Location",
    "Status":"Assigned_Vehicle_Status",
    "Age_Years":"Assigned_Vehicle_Age",
    "CO2_Emissions_Kg_per_KM":"Assigned_Vehicle_CO2_per_KM"
}
fleet_for_join = fleet2.rename(columns=veh_cols_map)
df = df.merge(fleet_for_join[list(veh_cols_map.values())].drop_duplicates(veh_cols_map["Vehicle_ID"]),
              on="Assigned_Vehicle_ID", how="left")

# ---------- Cost total ----------
cost_cols = ["Fuel_Cost","Labor_Cost","Vehicle_Maintenance","Insurance",
             "Packaging_Cost","Technology_Platform_Fee","Other_Overhead"]
use_costs = [c for c in cost_cols if c in df.columns]
df["Total_Cost_INR"] = df[use_costs].sum(axis=1, skipna=True) if use_costs else np.nan

# ---------- CO2 estimate per order ----------
if "CO2_Emissions_Kg_per_KM" in fleet.columns and "Distance_KM" in df.columns:
    # Prefer assigned vehicle's rate
    if "Assigned_Vehicle_CO2_per_KM" in df.columns:
        df["CO2_Est_kg"] = pd.to_numeric(df["Distance_KM"], errors="coerce") * \
                           pd.to_numeric(df["Assigned_Vehicle_CO2_per_KM"], errors="coerce")
    else:
        avg_co2 = pd.to_numeric(fleet["CO2_Emissions_Kg_per_KM"], errors="coerce").mean()
        df["CO2_Est_kg"] = pd.to_numeric(df["Distance_KM"], errors="coerce") * avg_co2
else:
    df["CO2_Est_kg"] = np.nan

# ---------- Save ----------
df.to_csv("merged_all.csv", index=False)

# ---------- Quick report ----------
def pct(x): 
    return f"{100.0* x:.1f}%"
used = {
    "orders": len(orders),
    "perf":   perf["Order_ID"].nunique(),
    "routes": routes["Order_ID"].nunique(),
    "costs":  costs["Order_ID"].nunique(),
    "feedback": feedback["Order_ID"].nunique(),
}
print("\n✅ Build complete -> merged_all.csv")
print("Rows:", len(df), " | Columns:", df.shape[1])
print("Delay label coverage:", pct(df["Is_Delayed"].notna().mean() if "Is_Delayed" in df else 0))
print("Inventory attached (by category):", pct(df["Assigned_Warehouse_ID"].notna().mean()))
print("Vehicle assigned:", pct(df["Assigned_Vehicle_ID"].notna().mean()))
print("Datasets merged (row-key = Order_ID):", used)
