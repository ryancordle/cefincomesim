# Streamlit-based CEF Income Simulation UI (Full Monte Carlo Version)

import streamlit as st
import numpy as np
import pandas as pd

# === Configuration ===
st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app runs a full Monte Carlo simulation based on realistic CEF portfolio behavior — including reinvestment, cost basis tracking, yield sensitivity, and profit harvesting.
""")

# === Sidebar ===
st.sidebar.header("Simulation Controls")
run_sim = st.sidebar.button("▶️ Run Simulation")

initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=10000, max_value=1000000, value=180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 5)
simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 1000, step=100)
reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", 0, 100, 100) / 100
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

# === Simulation ===
if run_sim:
    np.random.seed(42)
    periods_per_year = 12
    months = years * periods_per_year
    universe_size = 250
    num_cefs_initial = 200

    equity_return = (0.04, 0.07, 0.10)
    equity_yield = (0.06, 0.08, 0.10)
    equity_vol = (0.15, 0.17, 0.20)

    taxable_return = (-0.02, 0.00, 0.02)
    taxable_yield = (0.08, 0.11, 0.15)
    taxable_vol = (0.08, 0.10, 0.12)

    cef_type = np.random.rand(simulations, universe_size) < taxable_income_ratio
    base_returns = np.zeros((simulations, universe_size))
    base_yields = np.zeros((simulations, universe_size))

    for i in range(simulations):
        eq = ~cef_type[i]
        tx = cef_type[i]
        base_returns[i, eq] = np.random.triangular(*equity_return, size=eq.sum())
        base_yields[i, eq] = np.random.triangular(*equity_yield, size=eq.sum())
        base_returns[i, tx] = np.random.triangular(*taxable_return, size=tx.sum())
        base_yields[i, tx] = np.random.triangular(*taxable_yield, size=tx.sum())

    base_returns /= periods_per_year
    base_yields /= periods_per_year

    prices = np.random.triangular(3, 21.5, 40, size=(simulations, universe_size))
    units = np.zeros_like(prices)
    target = initial_investment / num_cefs_initial
    units[:, :num_cefs_initial] = target / prices[:, :num_cefs_initial]

    cost_basis = units * prices
    cash = np.zeros(simulations)
    monthly_working_capital = np.zeros((simulations, months))
    monthly_income = np.zeros((simulations, months))

    for m in range(months):
        portfolio_value = (prices * units).sum(axis=1) + cash
        income = (prices * units * base_yields).sum(axis=1)
        reinvest_income = income * reinvestment_pct
        cash += reinvest_income

        for i in range(simulations):
            for j in range(universe_size):
                if prices[i, j] == 0 or units[i, j] == 0:
                    continue
                value = prices[i, j] * units[i, j]
                if value > portfolio_value[i] * 0.03:
                    avg_cost = cost_basis[i, j] / units[i, j]
                    if prices[i, j] > avg_cost:
                        excess = value - portfolio_value[i] * 0.03
                        sell_units = excess / prices[i, j]
                        sell_units = min(sell_units, units[i, j])
                        units[i, j] -= sell_units
                        cash[i] += sell_units * prices[i, j]
                        cost_basis[i, j] -= (sell_units / (units[i, j] + sell_units)) * cost_basis[i, j]
                        if units[i, j] < 1e-6:
                            units[i, j] = 0
                            cost_basis[i, j] = 0

        total_value = (prices * units).sum(axis=1) + cash
        reinvest_unit_value = total_value / universe_size

        for i in range(simulations):
            for j in range(universe_size):
                if prices[i, j] > 0:
                    target_units = reinvest_unit_value / prices[i, j]
                    diff = target_units - units[i, j]
                    if diff > 0:
                        buy = min(diff, cash[i] / prices[i, j])
                        cost_basis[i, j] += buy * prices[i, j]
                        units[i, j] += buy
                        cash[i] -= buy * prices[i, j]

        monthly_working_capital[:, m] = cash + cost_basis.sum(axis=1)
        monthly_income[:, m] = income * 12
        prices *= (1 + base_returns)

    # === Results ===
    final_value = (prices * units).sum(axis=1) + cash
    final_cap = cash + cost_basis.sum(axis=1)
    final_income = monthly_income[:, -1]

    summary = pd.DataFrame({
        "Metric": ["Median", "Mean", "5th Percentile", "95th Percentile"],
        "Final Portfolio Value": [
            f"${np.median(final_value):,.0f}",
            f"${np.mean(final_value):,.0f}",
            f"${np.percentile(final_value, 5):,.0f}",
            f"${np.percentile(final_value, 95):,.0f}"
        ],
        "Final Working Capital": [
            f"${np.median(final_cap):,.0f}",
            f"${np.mean(final_cap):,.0f}",
            f"${np.percentile(final_cap, 5):,.0f}",
            f"${np.percentile(final_cap, 95):,.0f}"
        ],
        "Final Annual Income": [
            f"${np.median(final_income):,.0f}",
            f"${np.mean(final_income):,.0f}",
            f"${np.percentile(final_income, 5):,.0f}",
            f"${np.percentile(final_income, 95):,.0f}"
        ]
    })

    st.subheader("📊 Simulation Summary Table")
    st.dataframe(summary.set_index("Metric"))
else:
    st.info("Adjust parameters and click ▶️ Run Simulation to begin.")
