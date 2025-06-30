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
run_sim = st.sidebar.button("▶️ Run Simulation")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

st.sidebar.header("Portfolio Composition")
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

st.sidebar.header("Assumed Yield Ranges (Annual %)")
equity_yield = st.sidebar.slider("Equity CEF Yield Range (%)", 4, 12, (6, 10))
taxable_yield = st.sidebar.slider("Taxable Income CEF Yield Range (%)", 6, 20, (8, 15))

st.sidebar.header("Assumed Price Growth Ranges (Annual %)")
equity_growth = st.sidebar.slider("Equity CEF Growth Range (%)", -5, 15, (4, 10))
taxable_growth = st.sidebar.slider("Taxable CEF Growth Range (%)", -5, 5, (-2, 2))

st.sidebar.header("Income Reinvestment")
income_reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", 0, 100, 100) / 100

if run_sim:
    np.random.seed(42)
    months = years * 12

    monthly_records = []
    working_cap_records = []
    income_records = []

    for _ in range(simulations):
        principal = initial_investment
        cash = 0.0

        monthly_values = []
        monthly_working_cap = []
        monthly_income = []

        for _ in range(months):
            equity_alloc = principal * (1 - taxable_income_ratio)
            taxable_alloc = principal * taxable_income_ratio

            equity_yield_month = np.random.uniform(*equity_yield) / 12 / 100
            equity_growth_month = np.random.uniform(*equity_growth) / 12 / 100
            taxable_yield_month = np.random.uniform(*taxable_yield) / 12 / 100
            taxable_growth_month = np.random.uniform(*taxable_growth) / 12 / 100

            income = equity_alloc * equity_yield_month + taxable_alloc * taxable_yield_month

            reinvested_income = income * income_reinvestment_pct
            withdrawn_income = income * (1 - income_reinvestment_pct)

            cash += withdrawn_income  # Withdrawn income accumulates in cash, no compounding

            equity_total_growth = equity_yield_month + equity_growth_month
            taxable_total_growth = taxable_yield_month + taxable_growth_month

            overall_growth = (equity_total_growth * (1 - taxable_income_ratio)) + (taxable_total_growth * taxable_income_ratio)

            # Grow principal by overall growth rate
            principal = principal * (1 + overall_growth)

            # Add reinvested income after growth
            principal += reinvested_income

            total_value = principal + cash

            monthly_values.append(total_value)
            monthly_working_cap.append(total_value)  # working capital assumed as total here
            monthly_income.append(income * 12)  # annualized income

        monthly_records.append(monthly_values)
        working_cap_records.append(monthly_working_cap)
        income_records.append(monthly_income)

    # DataFrames
    portfolio_df = pd.DataFrame(monthly_records).T
    working_cap_df = pd.DataFrame(working_cap_records).T
    income_df = pd.DataFrame(income_records).T

    portfolio_df.columns = [f"Sim {i+1}" for i in range(simulations)]
    working_cap_df.columns = [f"Sim {i+1}" for i in range(simulations)]
    income_df.columns = [f"Sim {i+1}" for i in range(simulations)]

    final_values = portfolio_df.iloc[-1]
    final_working_cap = working_cap_df.iloc[-1]
    final_income = income_df.iloc[-1]

    summary_df = pd.DataFrame({
        "Metric": ["Median", "Mean", "5th Percentile", "95th Percentile"],
        "Final Portfolio Value": [
            f"${final_values.median():,.0f}",
            f"${final_values.mean():,.0f}",
            f"${final_values.quantile(0.05):,.0f}",
            f"${final_values.quantile(0.95):,.0f}"
        ],
        "Final Working Capital": [
            f"${final_working_cap.median():,.0f}",
            f"${final_working_cap.mean():,.0f}",
            f"${final_working_cap.quantile(0.05):,.0f}",
            f"${final_working_cap.quantile(0.95):,.0f}"
        ],
        "Final Annual Income": [
            f"${final_income.median():,.0f}",
            f"${final_income.mean():,.0f}",
            f"${final_income.quantile(0.05):,.0f}",
            f"${final_income.quantile(0.95):,.0f}"
        ]
    })

    st.subheader("📊 Simulation Summary Table")
    st.dataframe(summary_df.set_index("Metric"))

    st.caption("Note: Simulation reflects approximate monthly compounding of yield + growth assumptions. Use as planning tool, not forecast.")
else:
    st.info("Adjust the parameters and click **Run Simulation** to begin.")
