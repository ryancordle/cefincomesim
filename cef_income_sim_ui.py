import streamlit as st
import numpy as np
import pandas as pd

# --- Sidebar Inputs ---

st.sidebar.header("Simulation Parameters")

initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=10000, max_value=1_000_000, value=180000, step=1000)
years = st.sidebar.slider("Simulation Length (years)", 1, 50, 5)
reinvest_pct = st.sidebar.slider("Income Reinvestment %", 0, 100, 30)
num_cefs_initial = st.sidebar.slider("Initial Number of CEFs", 10, 250, 200)

# Taxable Income Ratio
taxable_income_ratio = st.sidebar.slider("Taxable Income Ratio (%)", 0, 100, 50) / 100

# Equity CEF params (annual returns)
equity_min_return = st.sidebar.slider("Equity CEF Min Annual Return", -0.05, 0.10, 0.04)
equity_mode_return = st.sidebar.slider("Equity CEF Mode Annual Return", -0.05, 0.15, 0.07)
equity_max_return = st.sidebar.slider("Equity CEF Max Annual Return", 0.00, 0.20, 0.10)

# Equity yield
equity_min_yield = st.sidebar.slider("Equity CEF Min Annual Yield", 0.00, 0.10, 0.06)
equity_mode_yield = st.sidebar.slider("Equity CEF Mode Annual Yield", 0.00, 0.12, 0.08)
equity_max_yield = st.sidebar.slider("Equity CEF Max Annual Yield", 0.00, 0.15, 0.10)

# Equity volatility
equity_min_vol = st.sidebar.slider("Equity CEF Min Annual Volatility", 0.10, 0.30, 0.15)
equity_mode_vol = st.sidebar.slider("Equity CEF Mode Annual Volatility", 0.10, 0.30, 0.17)
equity_max_vol = st.sidebar.slider("Equity CEF Max Annual Volatility", 0.10, 0.40, 0.20)

# Taxable Income CEF params (annual returns)
taxable_min_return = st.sidebar.slider("Taxable Income CEF Min Annual Return", -0.05, 0.05, -0.02)
taxable_mode_return = st.sidebar.slider("Taxable Income CEF Mode Annual Return", -0.05, 0.05, 0.00)
taxable_max_return = st.sidebar.slider("Taxable Income CEF Max Annual Return", 0.00, 0.10, 0.02)

# Taxable Income yield
taxable_min_yield = st.sidebar.slider("Taxable Income CEF Min Annual Yield", 0.05, 0.20, 0.08)
taxable_mode_yield = st.sidebar.slider("Taxable Income CEF Mode Annual Yield", 0.05, 0.25, 0.11)
taxable_max_yield = st.sidebar.slider("Taxable Income CEF Max Annual Yield", 0.10, 0.30, 0.15)

# Taxable Income volatility
taxable_min_vol = st.sidebar.slider("Taxable Income CEF Min Annual Volatility", 0.05, 0.20, 0.08)
taxable_mode_vol = st.sidebar.slider("Taxable Income CEF Mode Annual Volatility", 0.05, 0.25, 0.10)
taxable_max_vol = st.sidebar.slider("Taxable Income CEF Max Annual Volatility", 0.05, 0.30, 0.12)

# Interest rate params
initial_market_interest_rate = st.sidebar.slider("Initial Market Interest Rate", 0.00, 0.10, 0.04)
interest_rate_volatility_monthly = st.sidebar.slider("Interest Rate Monthly Volatility", 0.000, 0.01, 0.003)
interest_rate_price_sensitivity = st.sidebar.slider("Interest Rate Price Sensitivity", 0.0, 10.0, 4.0)
yield_sensitivity_to_rates = st.sidebar.slider("Yield Sensitivity to Rates", 0.0, 1.0, 0.5)

# Other constants
periods_per_year = 12
total_periods = years * periods_per_year
num_simulations = 1000
max_possible_cefs = 250

# --- Initialize Simulation Variables ---

# Determine CEF types for each sim (True = taxable, False = equity)
is_taxable_income_cef = np.random.rand(num_simulations, max_possible_cefs) < taxable_income_ratio

# Initialize parameter arrays
cef_base_annual_returns = np.zeros((num_simulations, max_possible_cefs))
cef_annual_volatilities = np.zeros((num_simulations, max_possible_cefs))
cef_base_annual_yields = np.zeros((num_simulations, max_possible_cefs))

for i in range(num_simulations):
    # Equity CEFs
    equity_idx = ~is_taxable_income_cef[i]
    n_equity = np.sum(equity_idx)
    if n_equity > 0:
        cef_base_annual_returns[i, equity_idx] = np.random.triangular(equity_min_return, equity_mode_return, equity_max_return, n_equity)
        cef_annual_volatilities[i, equity_idx] = np.random.triangular(equity_min_vol, equity_mode_vol, equity_max_vol, n_equity)
        cef_base_annual_yields[i, equity_idx] = np.random.triangular(equity_min_yield, equity_mode_yield, equity_max_yield, n_equity)

    # Taxable Income CEFs
    taxable_idx = is_taxable_income_cef[i]
    n_taxable = np.sum(taxable_idx)
    if n_taxable > 0:
        cef_base_annual_returns[i, taxable_idx] = np.random.triangular(taxable_min_return, taxable_mode_return, taxable_max_return, n_taxable)
        cef_annual_volatilities[i, taxable_idx] = np.random.triangular(taxable_min_vol, taxable_mode_vol, taxable_max_vol, n_taxable)
        cef_base_annual_yields[i, taxable_idx] = np.random.triangular(taxable_min_yield, taxable_mode_yield, taxable_max_yield, n_taxable)

# Convert annual to monthly
cef_monthly_base_returns = cef_base_annual_returns / periods_per_year
cef_monthly_volatilities = cef_annual_volatilities / np.sqrt(periods_per_year)

# Initialize prices randomly from triangular (3, 21.5, 40)
cef_prices = np.random.triangular(3, 21.5, 40, size=(num_simulations, max_possible_cefs))

# Initialize portfolio units and cost basis
initial_units_per_cef_target_value = initial_investment / num_cefs_initial
cef_units = np.zeros((num_simulations, max_possible_cefs))
cef_units[:, :num_cefs_initial] = initial_units_per_cef_target_value / cef_prices[:, :num_cefs_initial]
cost_basis_dollars = cef_units * cef_prices

cash = np.zeros(num_simulations)

# Arrays to track outputs
monthly_distributed_income = np.zeros((num_simulations, total_periods))
monthly_reinvested_income = np.zeros((num_simulations, total_periods))
monthly_working_capital = np.zeros((num_simulations, total_periods))
trades_per_month = np.zeros((num_simulations, total_periods))
num_total_cefs_invested_in = np.full(num_simulations, num_cefs_initial, dtype=int)

# Track market interest rates
monthly_market_interest_rates = np.zeros((num_simulations, total_periods))
monthly_market_interest_rates[:, 0] = initial_market_interest_rate

# --- Simulation Loop ---
for month in range(1, total_periods + 1):
    idx = month - 1

    # Simulate market interest rate
    if idx > 0:
        monthly_market_interest_rates[:, idx] = np.clip(
            monthly_market_interest_rates[:, idx-1] + np.random.normal(0, interest_rate_volatility_monthly, num_simulations),
            0.005, 0.15
        )
    current_rate = monthly_market_interest_rates[:, idx]
    rate_delta = np.zeros(num_simulations) if idx == 0 else current_rate - monthly_market_interest_rates[:, idx-1]

    # Price returns adjusted for interest rate impact
    price_impact = -rate_delta * interest_rate_price_sensitivity

    returns = np.random.normal(cef_monthly_base_returns, cef_monthly_volatilities)
    returns += price_impact[:, np.newaxis]
    cef_prices *= (1 + returns)

    # Adjust yields
    yield_adj = (current_rate - initial_market_interest_rate) * yield_sensitivity_to_rates
    current_yields = np.clip(
        cef_base_annual_yields / periods_per_year + yield_adj[:, np.newaxis] / periods_per_year,
        0.001, 0.20 / periods_per_year
    )

    # SELL: Take profits > 1%
    for i in range(num_simulations):
        held_idx = np.where(cef_units[i, :num_total_cefs_invested_in[i]] > 0)[0]
        if held_idx.size > 0:
            avg_cost = np.where(cef_units[i, held_idx] > 0,
                                cost_basis_dollars[i, held_idx] / cef_units[i, held_idx], 0)
            gains = (cef_prices[i, held_idx] - avg_cost) / np.where(avg_cost > 0, avg_cost, 1)
            sell_signals = gains >= 0.01
            trades_per_month[i, idx] += sell_signals.sum()

            units_to_sell = np.where(sell_signals, cef_units[i, held_idx] * 0.5, 0)
            units_to_sell = np.minimum(units_to_sell, cef_units[i, held_idx])
            value_sold = (units_to_sell * cef_prices[i, held_idx]).sum()
            cash[i] += value_sold

            for k, cef_j in enumerate(held_idx):
                if units_to_sell[k] > 0:
                    prop = units_to_sell[k] / cef_units[i, cef_j]
                    cef_units[i, cef_j] -= units_to_sell[k]
                    cost_basis_dollars[i, cef_j] -= cost_basis_dollars[i, cef_j] * prop
                    if cef_units[i, cef_j] < 1e-9:
                        cef_units[i, cef_j] = 0
                        cost_basis_dollars[i, cef_j] = 0

    # No fund >3% of portfolio (no selling at loss)
    for i in range(num_simulations):
        port_val = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
        thresh = port_val * 0.03
        for j in range(num_total_cefs_invested_in[i]):
            val = cef_prices[i, j] * cef_units[i, j]
            if val > thresh and cef_units[i, j] > 0:
                avg_cost_j = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0
                if cef_prices[i, j] < avg_cost_j:
                    continue
                excess_val = val - thresh
                units_sell = min(excess_val / cef_prices[i, j], cef_units[i, j])
                cef_units[i, j] -= units_sell
                cash[i] += units_sell * cef_prices[i, j]
                prop = units_sell / (cef_units[i, j] + units_sell) if (cef_units[i, j] + units_sell) > 1e-9 else 0
                cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * prop
                if cef_units[i, j] < 1e-9:
                    cef_units[i, j] = 0
                    cost_basis_dollars[i, j] = 0

    # Distributions & reinvestment
    gross_income = (cef_prices * cef_units * current_yields).sum(axis=1)
    dist_income = gross_income * (1 - reinvest_pct / 100)
    reinvest_income = gross_income * (reinvest_pct / 100)
    monthly_distributed_income[:, idx] = dist_income
    cash += reinvest_income
    monthly_reinvested_income[:, idx] = reinvest_income

    # Add new funds & rebalance equal weight (no selling at loss)
    for i in range(num_simulations):
        port_val = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
        while num_total_cefs_invested_in[i] < max_possible_cefs:
            if port_val <= 0:
                break
            potential_num = num_total_cefs_invested_in[i] + 1
            target_val = port_val / potential_num
            if cash[i] >= target_val * 0.95:
                num_total_cefs_invested_in[i] += 1
                port_val = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
            else:
                break

        if num_total_cefs_invested_in[i] > 0:
            target_val = port_val / num_total_cefs_invested_in[i]
            for j in range(num_total_cefs_invested_in[i]):
                current_val = cef_prices[i, j] * cef_units[i, j]
                diff = target_val - current_val
                if diff > 0:
                    units_can_buy = cash[i] / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_needed = diff / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_buy = min(units_needed, units_can_buy)
                    if units_buy > 0:
                        cef_units[i, j] += units_buy
                        cost_basis_dollars[i, j] += units_buy * cef_prices[i, j]
                        cash[i] -= units_buy * cef_prices[i, j]
                elif diff < 0 and cef_units[i, j] > 0:
                    avg_cost_j = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0
                    if cef_prices[i, j] < avg_cost_j:
                        continue
                    units_sell = min(abs(diff) / cef_prices[i, j], cef_units[i, j])
                    if units_sell > 0:
                        cef_units[i, j] -= units_sell
                        cash[i] += units_sell * cef_prices[i, j]
                        prop = units_sell / (cef_units[i, j] + units_sell) if (cef_units[i, j] + units_sell) > 1e-9 else 0
                        cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * prop
                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

    # Track working capital = cash + cost basis sum
    monthly_working_capital[:, idx] = cash + cost_basis_dollars.sum(axis=1)

# --- Results Summary ---

final_portfolio_value = (cef_prices * cef_units).sum(axis=1) + cash

def summary_df(data, name):
    return pd.DataFrame({
        f"{name} Mean": [np.mean(data)],
        f"{name} Median": [np.median(data)],
        f"{name} 5th %ile": [np.percentile(data, 5)],
        f"{name} 95th %ile": [np.percentile(data, 95)],
    })

# Summary tables for final values
port_val_summary = summary_df(final_portfolio_value, "Portfolio Value ($)")
working_cap_summary = summary_df(monthly_working_capital[:, -1], "Working Capital ($)")
income_summary = summary_df(monthly_distributed_income[:, -1], "Monthly Distributed Income ($)")
reinvest_summary = summary_df(monthly_reinvested_income[:, -1], "Monthly Reinvested Income ($)")

st.header("Simulation Results (After {} Years)".format(years))

st.subheader("Portfolio Value Summary")
st.table(port_val_summary.T)

st.subheader("Working Capital Summary")
st.table(working_cap_summary.T)

st.subheader("Monthly Distributed Income Summary")
st.table(income_summary.T)

st.subheader("Monthly Reinvested Income Summary")
st.table(reinvest_summary.T)
