import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulation", layout="wide")

st.title("CEF Portfolio Income & Growth Monte Carlo Simulation")

# --- User Inputs ---

initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=10000, max_value=1_000_000, value=180000, step=5000)
years = st.sidebar.slider("Years to Simulate", min_value=1, max_value=30, value=5)
reinvest_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", min_value=0, max_value=100, value=30)
num_cefs_initial = st.sidebar.slider("Initial Number of CEFs", min_value=10, max_value=250, value=200)

periods_per_year = 12
total_periods = years * periods_per_year
num_simulations = 1000
universe_size = 250
max_possible_cefs = universe_size

# --- Parameters ---
taxable_income_ratio = 0.5
equity_annual_base_return_dist = (0.04, 0.07, 0.10)
equity_annual_yield_dist = (0.06, 0.08, 0.10)
equity_annual_volatility_dist = (0.15, 0.17, 0.20)
taxable_income_annual_base_return_dist = (-0.02, 0.00, 0.02)
taxable_income_annual_yield_dist = (0.08, 0.11, 0.15)
taxable_income_annual_volatility_dist = (0.08, 0.10, 0.12)
initial_market_interest_rate = 0.04
interest_rate_volatility_monthly = 0.003
interest_rate_price_sensitivity = 4.0
yield_sensitivity_to_rates = 0.5

# --- Initialize CEF Types ---
np.random.seed(42)
is_taxable_income_cef = np.random.rand(num_simulations, max_possible_cefs) < taxable_income_ratio

cef_base_annual_returns = np.zeros((num_simulations, max_possible_cefs))
cef_annual_volatilities = np.zeros((num_simulations, max_possible_cefs))
cef_base_annual_yields = np.zeros((num_simulations, max_possible_cefs))

for i in range(num_simulations):
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
cef_prices = np.random.triangular(3, 21.5, 40, size=(num_simulations, max_possible_cefs))

monthly_market_interest_rates = np.zeros((num_simulations, total_periods))
monthly_market_interest_rates[:, 0] = initial_market_interest_rate

initial_units_per_cef_target_value = (initial_investment / num_cefs_initial)
cef_units = np.zeros((num_simulations, max_possible_cefs))
cef_units[:, :num_cefs_initial] = initial_units_per_cef_target_value / cef_prices[:, :num_cefs_initial]

cost_basis_dollars = cef_units * cef_prices
cash = np.zeros(num_simulations)

monthly_distributed_income = np.zeros((num_simulations, total_periods))
monthly_reinvested_income = np.zeros((num_simulations, total_periods))
monthly_working_capital = np.zeros((num_simulations, total_periods))
trades_per_month = np.zeros((num_simulations, total_periods))
num_total_cefs_invested_in = np.full(num_simulations, num_cefs_initial, dtype=int)

# --- Simulation Loop ---
for month in range(1, total_periods + 1):
    current_month_index = month - 1

    if current_month_index > 0:
        monthly_market_interest_rates[:, current_month_index] = np.clip(
            monthly_market_interest_rates[:, current_month_index - 1] + np.random.normal(0, interest_rate_volatility_monthly, num_simulations),
            0.005, 0.15
        )
    current_interest_rate_for_month = monthly_market_interest_rates[:, current_month_index]

    if current_month_index == 0:
        interest_rate_delta = np.zeros(num_simulations)
    else:
        interest_rate_delta = current_interest_rate_for_month - monthly_market_interest_rates[:, current_month_index - 1]

    price_impact_from_rates = -interest_rate_delta * interest_rate_price_sensitivity
    returns = np.random.normal(
        cef_monthly_base_returns, cef_monthly_volatilities, size=(num_simulations, max_possible_cefs)
    )
    returns += price_impact_from_rates[:, np.newaxis]
    cef_prices *= (1 + returns)

    yield_adjustment_from_rates = (current_interest_rate_for_month - initial_market_interest_rate) * yield_sensitivity_to_rates
    current_cef_monthly_yields = np.clip(
        cef_base_annual_yields / periods_per_year + yield_adjustment_from_rates[:, np.newaxis] / periods_per_year,
        0.001, 0.20 / periods_per_year
    )

    # SELL PROFITS >1%
    for i in range(num_simulations):
        held_indices = np.where(cef_units[i, :num_total_cefs_invested_in[i]] > 0)[0]
        if held_indices.size > 0:
            avg_cost = np.where(cef_units[i, held_indices] > 0,
                                cost_basis_dollars[i, held_indices] / cef_units[i, held_indices], 0)
            price_gain = (cef_prices[i, held_indices] - avg_cost) / np.where(avg_cost > 0, avg_cost, 1)
            sell_signals = price_gain >= 0.01
            trades_per_month[i, current_month_index] += sell_signals.sum()
            units_to_sell = np.where(sell_signals, cef_units[i, held_indices] * 0.5, 0)
            units_to_sell = np.minimum(units_to_sell, cef_units[i, held_indices])
            value_sold = (units_to_sell * cef_prices[i, held_indices]).sum()
            cash[i] += value_sold
            for idx, cef_idx in enumerate(held_indices):
                if units_to_sell[idx] > 0:
                    proportion_sold = units_to_sell[idx] / cef_units[i, cef_idx]
                    cef_units[i, cef_idx] -= units_to_sell[idx]
                    cost_basis_dollars[i, cef_idx] -= cost_basis_dollars[i, cef_idx] * proportion_sold
                    if cef_units[i, cef_idx] < 1e-9:
                        cef_units[i, cef_idx] = 0
                        cost_basis_dollars[i, cef_idx] = 0

    # No fund >3%
    for i in range(num_simulations):
        portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
        three_pct = portfolio_value * 0.03
        for j in range(num_total_cefs_invested_in[i]):
            val = cef_prices[i, j] * cef_units[i, j]
            if val > three_pct and cef_units[i, j] > 0:
                avg_cost = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0
                if cef_prices[i, j] < avg_cost:
                    continue
                excess_val = val - three_pct
                units_to_sell = min(excess_val / cef_prices[i, j], cef_units[i, j])
                cef_units[i, j] -= units_to_sell
                cash[i] += units_to_sell * cef_prices[i, j]
                if (cef_units[i, j] + units_to_sell) > 1e-9:
                    proportion_sold = units_to_sell / (cef_units[i, j] + units_to_sell)
                    cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * proportion_sold
                else:
                    cost_basis_dollars[i, j] = 0
                if cef_units[i, j] < 1e-9:
                    cef_units[i, j] = 0
                    cost_basis_dollars[i, j] = 0

    # Income distributions
    gross_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)
    distributed = gross_income * (1 - reinvest_pct / 100)
    reinvested = gross_income * (reinvest_pct / 100)
    cash += distributed
    monthly_distributed_income[:, current_month_index] = distributed
    monthly_reinvested_income[:, current_month_index] = reinvested
    cash += reinvested  # add reinvested income to cash for rebalancing

    # Add new funds & rebalance to equal weight (no selling at loss)
    for i in range(num_simulations):
        portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
        while num_total_cefs_invested_in[i] < max_possible_cefs:
            next_num = num_total_cefs_invested_in[i] + 1
            target_val = portfolio_value / next_num
            if cash[i] >= target_val * 0.95:
                num_total_cefs_invested_in[i] = next_num
                portfolio_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
            else:
                break
        if num_total_cefs_invested_in[i] > 0:
            target_val = portfolio_value / num_total_cefs_invested_in[i]
            for j in range(num_total_cefs_invested_in[i]):
                current_val = cef_prices[i, j] * cef_units[i, j]
                diff = target_val - current_val
                if diff > 0:
                    buy_units = min(cash[i] / cef_prices[i, j], diff / cef_prices[i, j])
                    if buy_units > 0:
                        cef_units[i, j] += buy_units
                        cost_basis_dollars[i, j] += buy_units * cef_prices[i, j]
                        cash[i] -= buy_units * cef_prices[i, j]
                elif diff < 0 and cef_units[i, j] > 0:
                    avg_cost = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0
                    if cef_prices[i, j] < avg_cost:
                        continue
                    sell_units = min(abs(diff) / cef_prices[i, j], cef_units[i, j])
                    if sell_units > 0:
                        cef_units[i, j] -= sell_units
                        cash[i] += sell_units * cef_prices[i, j]
                        proportion_sold = sell_units / (cef_units[i, j] + sell_units)
                        cost_basis_dollars[i, j] -= cost_basis_dollars[i, j] * proportion_sold
                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

    monthly_working_capital[:, current_month_index] = cash + cost_basis_dollars.sum(axis=1)

# --- Results ---

final_market_value = (cef_prices * cef_units).sum(axis=1) + cash

def summarize_metric(arr, label):
    return pd.DataFrame({
        "Mean": np.mean(arr),
        "Median": np.median(arr),
        "5th Percentile": np.percentile(arr, 5),
        "95th Percentile": np.percentile(arr, 95)
    }, index=[label])

st.header("Portfolio Summary at End of Simulation")

df_market_val = summarize_metric(final_market_value, "Market Value")
df_working_cap = summarize_metric(monthly_working_capital[:, -1], "Working Capital (Cost Basis + Cash)")
df_dist_income = summarize_metric(monthly_distributed_income[:, -1], "Distributed Income (Monthly)")
df_reinvest_income = summarize_metric(monthly_reinvested_income[:, -1], "Reinvested Income (Monthly)")

summary_df = pd.concat([df_market_val, df_working_cap, df_dist_income, df_reinvest_income])
st.table(summary_df.style.format("${:,.2f}"))

st.header("CEF Investment Count")

st.write(f"Mean CEFs Invested In: {np.mean(num_total_cefs_invested_in):.1f}")
st.write(f"Median CEFs Invested In: {np.median(num_total_cefs_invested_in):.0f}")
st.write(f"Min CEFs Invested In: {np.min(num_total_cefs_invested_in)}")
st.write(f"Max CEFs Invested In: {np.max(num_total_cefs_invested_in)}")

st.markdown("---")

st.write("### Note: This simulation runs 1000 trials with stochastic returns and distributions based on your inputs.")

