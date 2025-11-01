# NexGen Logistics Predictive Control Tower

## Project Overview
This project aims to **predict delivery delays** and optimize logistics operations using **data analytics** and **machine learning**. By using real-time data, the solution can predict potential delivery delays, suggest vehicle assignments, and improve operational efficiency.

## Problem Statement
The logistics company **NexGen Logistics Pvt. Ltd.** faces critical challenges such as:
- **Delivery Performance Issues**: Inconsistent on-time deliveries.
- **Operational Inefficiencies**: High costs and inefficient route planning.
- **Limited Innovation**: Lack of predictive data-driven decisions.
- **Sustainability Concerns**: Environmental impact of inefficient logistics operations.

## Approach
To solve these challenges, we created a **Predictive Control Tower** using:
- **Data-Driven Decision-Making**: Analyze historical data (orders, delivery performance, costs) to predict potential delays.
- **Machine Learning Models**: Built a **classification model** to predict delays based on various input features (distance, traffic delays, vehicle type, etc.).
- **Streamlit Dashboard**: Built an interactive web dashboard using Streamlit to visualize predictions and suggest corrective actions.

## Solution
- **Predictive Model**: A machine learning model was built to predict whether an order will be delayed based on past data.
- **Real-Time Dashboard**: An interactive dashboard allows users to filter by **origin**, **destination**, and **priority** to view predicted delays and get vehicle suggestions.

## Features
- **Real-Time Predictions**: Predicts delays and provides insights into possible corrective actions.
- **Filter Options**: Filter data based on origin, destination, and priority for customized insights.
- **Download Reports**: Users can download flagged orders and prescriptive suggestions in CSV format.

## Requirements
To run this project locally, you will need the following dependencies:
- Python 3.x
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib

To install the dependencies, run the following:

```bash
pip install -r requirements.txt
