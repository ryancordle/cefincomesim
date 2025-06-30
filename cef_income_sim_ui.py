import streamlit as st
import numpy as np
import pandas as pd

st.title("CEF Portfolio Simulator with Cost Basis, Market Value, and Cash Tracking")

# Sliders for user inputs
initial_investment = st.slider("Initial Investment ($)", 50000, 500000, 180000, step=5000)
years = st.slider("Years to Simulate", 5, 30, 19, step=1)
income_reinvestment_pct = st.slider("Income Reinvestment Percentage", 0, 100, 70, step=5) / 100
taxable_income_ratio = st.slider("Taxable Income CEF Ratio", 0, 100, 50, step=5) / 100
simulations = st.slider("Number of Simulations", 100, 2000, 1000, step=100)

# Parameters (simplified & fixed for demo)
equity_yield_annual = 0.08
equity_growth_annual = 0.08
taxable_yield_annual = 0.11
taxable_growth_annual = 0.00

# Convert annual to monthly
equity_yield_month = equity_yield_annual / 12
equity_growth_month = equity_growth_annual / 12
taxable_yield_month = taxable_yield_annual / 12
taxable_growth_month = taxable_growth_annual / 12

months = years * 12

np.random.seed(42)

cost_basis_results = []
market_value_results = []
cash_results = []
annual_income_results = []

progress_text = "Running simulations..."
my_bar = st.progress(0)

for sim in range(simulations):
    cost_basis = initial_investment
    cash = 0.0
    market_value = initial_investment

    monthly_cost_basis = []
    monthly_market_value = []
    monthly_cash = []
    monthly_annual_income = []

    for month in range(months):
        equity_cb = cost_basis * (1 - taxable_income_ratio)
        taxable_cb = cost_basis * taxable_income_ratio

        equity_growth = np.random.normal(equity_growth_month, 0.02)
        taxable_growth = np.random.normal(taxable_growth_month, 0.01)
        equity_yield = np.random.normal(equity_yield_month, 0.005)
        taxable_yield = np.random.normal(taxable_yield_month, 0.005)

        equity_mv = equity_cb * (1 + equity_growth)
        taxable_mv = taxable_cb * (1 + taxable_growth)

        market_value = equity_mv + taxable_mv

        equity_income = equity_mv * equity_yield
        taxable_income = taxable_mv * taxable_yield
        total_income = equity_income + taxable_income

        annual_income = total_income * 12

        reinvested_income = total_income * income_reinvestment_pct
        distributed_income = total_income * (1 - income_reinvestment_pct)

        cash += distributed_income

        cost_basis += reinvested_income + cash
        cash = 0.0

        monthly_cost_basis.append(cost_basis)
        monthly_market_value.append(market_value)
        monthly_cash.append(cash)
        monthly_annual_income.append(annual_income)

    cost_basis_results.append(monthly_cost_basis)
    market_value_results.append(monthly_market_value)
    cash_results.append(monthly_cash)
    annual_income_results.append(monthly_annual_income)

    if sim % max(1, simulations // 100) == 0:
        my_bar.progress(min(100, int(sim / simulations * 100)))

my_bar.empty()

# Convert to DataFrames
cost_basis_df = pd.DataFrame(cost_basis_results).T
market_value_df = pd.DataFrame(market_value_results).T
cash_df = pd.DataFrame(cash_results).T
income_df = pd.DataFrame(annual_income_results).T

def summarize(series):
    return {
        "5th Percentile": series.quantile(0.05),
        "Median": series.median(),
        "95th Percentile": series.quantile(0.95)
    }

final_cb = cost_basis_df.iloc[-1]
final_mv = market_value_df.iloc[-1]
final_cash = cash_df.iloc[-1]
final_income = income_df.iloc[-1]

cb_stats = summarize(final_cb)
mv_stats = summarize(final_mv)
cash_stats = summarize(final_cash)
income_stats = summarize(final_income)

working_capital = final_cb + final_cash
working_capital_stats = summarize(working_capital)

st.subheader(f"Simulation Results After {years} Years")

st.markdown(f"""
| Metric             | 5th Percentile    | Median           | 95th Percentile  |
|--------------------|-------------------|------------------|------------------|
| Cost Basis         | ${cb_stats['5th Percentile']:,.0f} | ${cb_stats['Median']:,.0f} | ${cb_stats['95th Percentile']:,.0f} |
| Market Value       | ${mv_stats['5th Percentile']:,.0f} | ${mv_stats['Median']:,.0f} | ${mv_stats['95th Percentile']:,.0f} |
| Cash               | ${cash_stats['5th Percentile']:,.0f} | ${cash_stats['Median']:,.0f} | ${cash_stats['95th Percentile']:,.0f} |
| **Working Capital (Cost Basis + Cash)** | ${working_capital_stats['5th Percentile']:,.0f} | ${working_capital_stats['Median']:,.0f} | ${working_capital_stats['95th Percentile']:,.0f} |
| Annual Income      | ${income_stats['5th Percentile']:,.0f} | ${income_stats['Median']:,.0f} | ${income_stats['95th Percentile']:,.0f} |
""")

st.subheader("Portfolio Value Over Time")

def plot_percentiles(df, label):
    percentiles = df.quantile([0.05, 0.5, 0.95], axis=1).T.rename(
        columns={0.05: "5th Percentile", 0.5: "Median", 0.95: "95th Percentile"}
    )
    st.line_chart(percentiles)
    st.caption(f"{label} with 5th, Median, and 95th percentiles")

plot_percentiles(cost_basis_df, "Cost Basis")
plot_percentiles(market_value_df, "Market Value")
plot_percentiles(cash_df, "Cash")
plot_percentiles(cost_basis_df + cash_df, "Working Capital (Cost Basis + Cash)")
plot_percentiles(income_df, "Annual Income")

