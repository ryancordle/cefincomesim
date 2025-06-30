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
        equity_yield_month = np.random.uniform(*equity_yield) / 12 / 100
        equity_growth_month = np.random.uniform(*equity_growth) / 12 / 100
        taxable_yield_month = np.random.uniform(*taxable_yield) / 12 / 100
        taxable_growth_month = np.random.uniform(*taxable_growth) / 12 / 100

        # Monthly compounding
        equity_value = equity_alloc * (1 + equity_yield_month + equity_growth_month)
        taxable_value = taxable_alloc * (1 + taxable_yield_month + taxable_growth_month)

        value = equity_value + taxable_value
        monthly_values.append(value)
    monthly_records.append(monthly_values)

# Convert to DataFrame
portfolio_df = pd.DataFrame(monthly_records).T
portfolio_df.columns = [f"Sim {i+1}" for i in range(simulations)]

# --- Improved Results Section ---

st.subheader("📊 Simulation Results Summary")

final_values = portfolio_df.iloc[-1]

median_final = final_values.median()
p5_final = final_values.quantile(0.05)
p95_final = final_values.quantile(0.95)

st.markdown(f"""
**Final Portfolio Value (after {years} years):**

- Median: **${median_final:,.0f}**  
- 5th Percentile (Conservative): **${p5_final:,.0f}**  
- 95th Percentile (Optimistic): **${p95_final:,.0f}**  

These numbers represent the range of possible portfolio outcomes after {years} years, based on your assumptions.
""")

st.subheader("Distribution of Final Portfolio Values")
hist_values = final_values.value_counts(bins=30).sort_index()
st.bar_chart(hist_values)

st.subheader("Portfolio Value Over Time (Selected Percentiles)")
percentiles = portfolio_df.quantile([0.05, 0.5, 0.95], axis=1).T
percentiles.index.name = "Month"
percentiles = percentiles.rename(columns={0.05: "5th Percentile", 0.5: "Median", 0.95: "95th Percentile"})
st.line_chart(percentiles)

st.subheader("Key Summary Statistics")
summary_df = pd.DataFrame({
    "5th Percentile": [p5_final],
    "Median": [median_final],
    "95th Percentile": [p95_final]
}, index=[f"End of Year {years}"])

summary_df = summary_df.style.format("${:,.0f}")
st.dataframe(summary_df, use_container_width=True)

st.caption("Interpret these results as a probabilistic range of outcomes based on your investment assumptions.")
