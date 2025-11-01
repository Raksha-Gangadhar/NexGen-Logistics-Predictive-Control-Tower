import pandas as pd

# Load the datasets
orders = pd.read_csv("orders.csv")
perf = pd.read_csv("delivery_performance.csv")
routes = pd.read_csv("routes_distance.csv")
costs = pd.read_csv("cost_breakdown.csv")
feedback = pd.read_csv("customer_feedback.csv")  # Optional
fleet = pd.read_csv("vehicle_fleet.csv")  # Optional
inventory = pd.read_csv("warehouse_inventory.csv")  # Optional

# Show the first few rows of each dataset
print("Orders Dataset:")
print(orders.head())
print("\nDelivery Performance Dataset:")
print(perf.head())
print("\nRoutes Distance Dataset:")
print(routes.head())
print("\nCost Breakdown Dataset:")
print(costs.head())
print("\nCustomer Feedback Dataset:")
print(feedback.head())
print("\nVehicle Fleet Dataset:")
print(fleet.head())
print("\nWarehouse Inventory Dataset:")
print(inventory.head())
