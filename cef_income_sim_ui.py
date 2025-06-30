# Streamlit-based CEF Income Simulation UI

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app simulates long-term growth and income of a closed-end fund (CEF) portfolio based on your investment philosophy.

*Adjust the inputs below and explore the results of a Monte Carlo simulation based on price growth, yield, and interest rates.*
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

st.sidebar.header("Portfolio Composition")
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

st.sidebar.header("Assumed Yield Ranges (Annual)")
equity_yield = st.sidebar.slider("Equity CEF Yield Range (%)", 4, 12, (6, 10))
taxable_yield = st.sidebar.slider("Taxable Income CEF Yield Range (%)", 6, 20, (8, 15))

st.sidebar.header("Assumed Price Growth Ranges (Annual)")
equity_growth = st.sidebar.slider("Equity CEF Growth Range (%)", -5, 15, (4, 10))
taxable_growth = st.sidebar.slider("Taxable CEF Growth Range (%)", -5, 5, (-2, 2))

# --- Simulation Logic ---
np.random.seed(42)
months = years * 12
monthly_records = []

for _ in range(simulations):
    value = initial_investment
    monthly_values = []
    for _ in range(months):
        equity_alloc = value * (1 - taxable_income_ratio)
        taxable_alloc = value * taxable_income_ratio

        # Monthly yields and returns
        equity_yield_month = np.random.uniform(*equity_yield) / 12
        equity_growth_month = np.random.uniform(*equity_growth) / 12
        taxable_yield_month = np.random.uniform(*taxable_yield) / 12
        taxable_growth_month = np.random.uniform(*taxable_growth) / 12

        # Monthly compounding
        equity_value = equity_alloc * (1 + equity_yield_month + equity_growth_month)
        taxable_value = taxable_alloc * (1 + taxable_yield_month + taxable_growth_month)

        value = equity_value + taxable_value
        monthly_values.append(value)
    monthly_records.append(monthly_values)

# --- Output Summary ---
portfolio_df = pd.DataFrame(monthly_records).T
portfolio_df.columns = [f"Sim {i+1}" for i in range(simulations)]

st.subheader("📊 Simulation Results")
st.write(f"Final Median Portfolio Value: **${portfolio_df.iloc[-1].median():,.0f}**")
st.write(f"Final 5th Percentile: **${portfolio_df.iloc[-1].quantile(0.05):,.0f}**, Final 95th Percentile: **${portfolio_df.iloc[-1].quantile(0.95):,.0f}**")

st.line_chart(portfolio_df[[f"Sim {i+1}" for i in range(min(10, simulations))]])

percentiles = portfolio_df.quantile([0.05, 0.5, 0.95], axis=1).T
st.area_chart(percentiles)

st.caption("Note: Simulation reflects approximate monthly compounding of yield + growth assumptions. Use as planning tool, not forecast.")
