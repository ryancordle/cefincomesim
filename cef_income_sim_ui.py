# Streamlit-based CEF Income Simulation UI (Full Monte Carlo Version)

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app simulates long-term growth and income of a closed-end fund (CEF) portfolio using a Monte Carlo method based on detailed assumptions.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")
run_sim = st.sidebar.button("▶️ Run Simulation")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)
reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", 0, 100, 100) / 100
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

# CEF Yield and Growth Settings
equity_yield_range = (0.06, 0.10)
taxable_yield_range = (0.08, 0.15)
equity_growth_range = (0.04, 0.10)
taxable_growth_range = (-0.02, 0.02)

# Run simulation only when button pressed
if run_sim:
    np.random.seed(42)
    months = years * 12
    portfolio_vals, working_caps, incomes = [], [], []

    for _ in range(simulations):
        value = initial_investment
        cost_basis = initial_investment
        month_vals, month_caps, month_incomes = [], [], []

        for _ in range(months):
            equity_val = value * (1 - taxable_income_ratio)
            taxable_val = value * taxable_income_ratio

            eq_yield = np.random.uniform(*equity_yield_range) / 12
            tx_yield = np.random.uniform(*taxable_yield_range) / 12
            eq_growth = np.random.uniform(*equity_growth_range) / 12
            tx_growth = np.random.uniform(*taxable_growth_range) / 12

            income = equity_val * eq_yield + taxable_val * tx_yield
            reinvest = income * reinvestment_pct
            value = equity_val * (1 + eq_growth) + taxable_val * (1 + tx_growth) + reinvest

            cost_basis += reinvest

            month_vals.append(value)
            month_caps.append(cost_basis)
            month_incomes.append(income * 12)

        portfolio_vals.append(month_vals)
        working_caps.append(month_caps)
        incomes.append(month_incomes)

    # Create DataFrames
    portfolio_df = pd.DataFrame(portfolio_vals).T
    working_cap_df = pd.DataFrame(working_caps).T
    income_df = pd.DataFrame(incomes).T

    final_value = portfolio_df.iloc[-1]
    final_cap = working_cap_df.iloc[-1]
    final_income = income_df.iloc[-1]

    summary = pd.DataFrame({
        "Metric": ["Median", "Mean", "5th Percentile", "95th Percentile"],
        "Final Portfolio Value": [
            f"${final_value.median():,.0f}",
            f"${final_value.mean():,.0f}",
            f"${final_value.quantile(0.05):,.0f}",
            f"${final_value.quantile(0.95):,.0f}"
        ],
        "Final Working Capital": [
            f"${final_cap.median():,.0f}",
            f"${final_cap.mean():,.0f}",
            f"${final_cap.quantile(0.05):,.0f}",
            f"${final_cap.quantile(0.95):,.0f}"
        ],
        "Final Annual Income": [
            f"${final_income.median():,.0f}",
            f"${final_income.mean():,.0f}",
            f"${final_income.quantile(0.05):,.0f}",
            f"${final_income.quantile(0.95):,.0f}"
        ]
    })

    st.subheader("📊 Simulation Summary Table")
    st.dataframe(summary.set_index("Metric"))
    st.caption("Note: This reflects a simplified Monte Carlo model. Portfolio behavior depends heavily on initial parameters.")
else:
    st.info("Adjust the parameters and click **Run Simulation** to begin.")
