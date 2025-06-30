# Streamlit-based CEF Income Simulation UI with Full Original Logic

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app simulates long-term growth and income of a closed-end fund (CEF) portfolio based on your detailed investment philosophy.

*Adjust the inputs below and explore the results of a Monte Carlo simulation based on your original logic, including price returns, yields, interest rates, rebalancing, and trading.*
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")
run_sim = st.sidebar.button("▶️ Run Simulation")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

st.sidebar.header("Portfolio Composition")
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

st.sidebar.header("CEF Universe Size")
universe_size = st.sidebar.slider("Number of Different CEFs Possible", 50, 500, 250, step=10)

st.sidebar.header("Initial Portfolio")
num_cefs_initial = st.sidebar.slider("Number of CEFs Initially Held", 10, 300, 200)

st.sidebar.header("Interest Rate Environment")
initial_market_interest_rate = st.sidebar.slider("Initial Market Interest Rate (%)", 0, 15, 4) / 100
interest_rate_volatility_monthly = st.sidebar.number_input("Monthly Interest Rate Volatility (decimal)", 0.0001, 0.01, 0.003, format="%.5f")
interest_rate_price_sensitivity = st.sidebar.number_input("Price Sensitivity to Interest Rate (Duration)", 1.0, 10.0, 4.0)
yield_sensitivity_to_rates = st.sidebar.number_input("Yield Sensitivity to Interest Rate", 0.0, 1.0, 0.5)

if run_sim:
    st.info("Running simulation... this may take a minute for many simulations/years.")

    # Constants & params
    periods_per_year = 12
    total_periods = years * periods_per_year
    max_possible_cefs = universe_size

    # Define CEF parameter distributions (from original code)
    equity_annual_base_return_dist = (0.04, 0.07, 0.10) # min, mode, max
    equity_annual_yield_dist = (0.06, 0.08, 0.10)
    equity_annual_volatility_dist = (0.15, 0.17, 0.20)

    taxable_income_annual_base_return_dist = (-0.02, 0.00, 0.02)
    taxable_income_annual_yield_dist = (0.08, 0.11, 0.15)
    taxable_income_annual_volatility_dist = (0.08, 0.10, 0.12)

    # Initialize arrays
    np.random.seed(42)
    is_taxable_income_cef = np.random.rand(simulations, max_possible_cefs) < taxable_income_ratio

    cef_base_annual_returns = np.zeros((simulations, max_possible_cefs))
    cef_annual_volatilities = np.zeros((simulations, max_possible_cefs))
    cef_base_annual_yields = np.zeros((simulations, max_possible_cefs))

    for i in range(simulations):
        equity_indices = ~is_taxable_income_cef[i]
        num_equity = np.sum(equity_indices)
        if num_equity > 0:
            cef_base_annual_returns[i, equity_indices] = np.random.triangular(*equity_annual_base_return_dist, size=num_equity)
            cef_annual_volatilities[i, equity_indices] = np.random.triangular(*equity_annual_volatility_dist, size=num_equity)
            cef_base_annual_yields[i, equity_indices] = np.random.triangular(*equity_annual_yield_dist, size=num_equity)
        taxable_indices = is_taxable_income_cef[i]
        num_taxable = np.sum(taxable_indices)
        if num_taxable > 0:
            cef_base_annual_returns[i, taxable_indices] = np.random.triangular(*taxable_income_annual_base_return_dist, size=num_taxable)
            cef_annual_volatilities[i, taxable_indices] = np.random.triangular(*taxable_income_annual_volatility_dist, size=num_taxable)
            cef_base_annual_yields[i, taxable_indices] = np.random.triangular(*taxable_income_annual_yield_dist, size=num_taxable)

    cef_monthly_base_returns = cef_base_annual_returns / periods_per_year
    cef_monthly_volatilities = cef_annual_volatilities / np.sqrt(periods_per_year)

    cef_prices = np.random.triangular(3, 21.5, 40, size=(simulations, max_possible_cefs))
    monthly_market_interest_rates = np.zeros((simulations, total_periods))
    monthly_market_interest_rates[:, 0] = initial_market_interest_rate

    initial_units_per_cef_target_value = (initial_investment / num_cefs_initial)
    cef_units = np.zeros((simulations, max_possible_cefs))
    cef_units[:, :num_cefs_initial] = initial_units_per_cef_target_value / cef_prices[:, :num_cefs_initial]

    cost_basis_dollars = cef_units * cef_prices
    cash = np.zeros(simulations)

    monthly_distributed_income = np.zeros((simulations, total_periods))
    monthly_reinvested_income = np.zeros((simulations, total_periods))
    monthly_working_capital = np.zeros((simulations, total_periods))
    trades_per_month = np.zeros((simulations, total_periods))

    num_total_cefs_invested_in = np.full(simulations, num_cefs_initial, dtype=int)

    # Run simulation loop
    for month in range(1, total_periods + 1):
        current_month_index = month - 1
        if current_month_index > 0:
            monthly_market_interest_rates[:, current_month_index] = np.clip(
                monthly_market_interest_rates[:, current_month_index - 1] + np.random.normal(0, interest_rate_volatility_monthly, simulations),
                0.005, 0.15
            )
        current_interest_rate_for_month = monthly_market_interest_rates[:, current_month_index]

        if current_month_index == 0:
            interest_rate_delta = np.zeros(simulations)
        else:
            interest_rate_delta = current_interest_rate_for_month - monthly_market_interest_rates[:, current_month_index - 1]

        price_impact_from_rates = -interest_rate_delta * interest_rate_price_sensitivity

        returns = np.random.normal(
            cef_monthly_base_returns,
            cef_monthly_volatilities,
            size=(simulations, max_possible_cefs)
        )
        returns += price_impact_from_rates[:, np.newaxis]
        cef_prices *= (1 + returns)

        yield_adjustment_from_rates = (current_interest_rate_for_month - initial_market_interest_rate) * yield_sensitivity_to_rates
        current_cef_monthly_yields = np.clip(
            cef_base_annual_yields / periods_per_year + yield_adjustment_from_rates[:, np.newaxis] / periods_per_year,
            0.001, 0.20 / periods_per_year
        )

        # SELL: Take profits >1%
        for i in range(simulations):
            held_cefs = np.where(cef_units[i, :num_total_cefs_invested_in[i]] > 0)[0]
            if held_cefs.size > 0:
                avg_costs = np.where(cef_units[i, held_cefs] > 0, cost_basis_dollars[i, held_cefs] / cef_units[i, held_cefs], 0)
                gains = (cef_prices[i, held_cefs] - avg_costs) / np.where(avg_costs > 0, avg_costs, 1)
                sell_mask = gains >= 0.01

                trades_per_month[i, current_month_index] += sell_mask.sum()
                units_to_sell = np.where(sell_mask, cef_units[i, held_cefs] * 0.5, 0)
                units_to_sell = np.minimum(units_to_sell, cef_units[i, held_cefs])
                cash[i] += (units_to_sell * cef_prices[i, held_cefs]).sum()

                for idx, j in enumerate(held_cefs):
                    if units_to_sell[idx] > 0:
                        proportion = units_to_sell[idx] / cef_units[i, j]
                        cef_units[i, j] -= units_to_sell[idx]
                        cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * proportion
                        if cef_units[i, j] < 1e-9:
                            cef_units[i, j] = 0
                            cost_basis_dollars[i, j] = 0

        # Cap any single fund >3% portfolio at market value (without selling at loss)
        for i in range(simulations):
            portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
            cap_val = portfolio_value * 0.03
            for j in range(num_total_cefs_invested_in[i]):
                cef_val = cef_prices[i, j] * cef_units[i, j]
                if cef_val > cap_val and cef_units[i, j] > 0:
                    avg_cost = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0
                    if cef_prices[i, j] < avg_cost:
                        continue
                    excess = cef_val - cap_val
                    units_to_sell = min(excess / cef_prices[i, j], cef_units[i, j])
                    cef_units[i, j] -= units_to_sell
                    cash[i] += units_to_sell * cef_prices[i, j]
                    proportion = units_to_sell / (units_to_sell + cef_units[i, j]) if (units_to_sell + cef_units[i, j]) > 1e-9 else 0
                    cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * proportion
                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

        # Distributions and reinvestment
        gross_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)
        distributed_income = gross_income * 0.7
        monthly_distributed_income[:, current_month_index] = distributed_income
        reinvestable_income = gross_income * 0.3
        cash += reinvestable_income
        monthly_reinvested_income[:, current_month_index] = reinvestable_income

        # Add new funds & rebalance equal weight
        for i in range(simulations):
            portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]

            while num_total_cefs_invested_in[i] < max_possible_cefs:
                target_value = portfolio_value / (num_total_cefs_invested_in[i] + 1)
                if cash[i] >= target_value * 0.95:
                    num_total_cefs_invested_in[i] += 1
                    portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
                else:
                    break

            if num_total_cefs_invested_in[i] > 0:
                target_value_per_fund = portfolio_value / num_total_cefs_invested_in[i]
                for j in range(num_total_cefs_invested_in[i]):
                    current_val = cef_prices[i, j] * cef_units[i, j]
                    diff = target_value_per_fund - current_val
                    if diff > 0:
                        units_to_buy = min(diff / cef_prices[i, j], cash[i] / cef_prices[i, j])
                        if units_to_buy > 0:
                            cef_units[i, j] += units_to_buy
                            cost_basis_dollars[i, j] += units_to_buy * cef_prices[i, j]
                            cash[i] -= units_to_buy * cef_prices[i, j]
                    elif diff < 0 and cef_units[i, j] > 0:
                        avg_cost = cost_basis_dollars[i, j] / cef_units[i, j]
                        if cef_prices[i, j] < avg_cost:
                            continue
                        units_to_sell = min(abs(diff) / cef_prices[i, j], cef_units[i, j])
                        if units_to_sell > 0:
                            cef_units[i, j] -= units_to_sell
                            cash[i] += units_to_sell * cef_prices[i, j]
                            proportion = units_to_sell / (units_to_sell + cef_units[i, j]) if (units_to_sell + cef_units[i, j]) > 1e-9 else 0
                            cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * proportion
                            if cef_units[i, j] < 1e-9:
                                cef_units[i, j] = 0
                                cost_basis_dollars[i, j] = 0

        monthly_working_capital[:, current_month_index] = cash + cost_basis_dollars.sum(axis=1)

    final_market_value = (cef_prices * cef_units).sum(axis=1) + cash
    final_working_capital = monthly_working_capital[:, -1]
    final_distributed_income = monthly_distributed_income[:, -1] * 12  # annualize last month income

    summary_df = pd.DataFrame({
        "Metric": ["Median", "Mean", "5th Percentile", "95th Percentile"],
        "Final Market Value": [
            f"${np.median(final_market_value):,.0f}",
            f"${np.mean(final_market_value):,.0f}",
            f"${np.percentile(final_market_value, 5):,.0f}",
            f"${np.percentile(final_market_value, 95):,.0f}",
        ],
        "Final Working Capital": [
            f"${np.median(final_working_capital):,.0f}",
            f"${np.mean(final_working_capital):,.0f}",
            f"${np.percentile(final_working_capital, 5):,.0f}",
            f"${np.percentile(final_working_capital, 95):,.0f}",
        ],
        "Final Annual Distributed Income": [
            f"${np.median(final_distributed_income):,.0f}",
            f"${np.mean(final_distributed_income):,.0f}",
            f"${np.percentile(final_distributed_income, 5):,.0f}",
            f"${np.percentile(final_distributed_income, 95):,.0f}",
        ],
    })

    st.subheader("📊 Simulation Summary Table")
    st.dataframe(summary_df.set_index("Metric"))

    st.caption("Note: This simulation uses your original detailed logic, including rebalancing, profit-taking, and yield adjustments.")

else:
    st.info("Adjust the parameters and click **Run Simulation** to begin.")
