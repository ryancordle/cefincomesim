import streamlit as st
import numpy as np
import pandas as pd

# --- Sidebar Inputs ---

st.sidebar.header("Simulation Parameters")

num_cefs_initial = st.sidebar.slider("Initial Number of CEFs", 10, 250, 200, step=10)
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=10000, max_value=1_000_000, value=180000, step=1000)
years = st.sidebar.slider("Years to Simulate", 1, 50, 5)
periods_per_year = st.sidebar.slider("Periods Per Year (Months)", 1, 12, 12)
num_simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

taxable_income_ratio = st.sidebar.slider(
    "Taxable Income CEF Allocation (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
) / 100

# CEF parameter sliders
st.sidebar.header("Equity CEF Parameters")
equity_annual_base_return_dist = (
    st.sidebar.slider("Equity Base Return Min", 0.00, 0.20, 0.04),
    st.sidebar.slider("Equity Base Return Mode", 0.00, 0.20, 0.07),
    st.sidebar.slider("Equity Base Return Max", 0.00, 0.30, 0.10)
)
equity_annual_yield_dist = (
    st.sidebar.slider("Equity Yield Min", 0.00, 0.20, 0.06),
    st.sidebar.slider("Equity Yield Mode", 0.00, 0.20, 0.08),
    st.sidebar.slider("Equity Yield Max", 0.00, 0.20, 0.10)
)
equity_annual_volatility_dist = (
    st.sidebar.slider("Equity Volatility Min", 0.00, 0.30, 0.15),
    st.sidebar.slider("Equity Volatility Mode", 0.00, 0.30, 0.17),
    st.sidebar.slider("Equity Volatility Max", 0.00, 0.40, 0.20)
)

st.sidebar.header("Taxable Income CEF Parameters")
taxable_income_annual_base_return_dist = (
    st.sidebar.slider("Taxable Income Base Return Min", -0.10, 0.10, -0.02),
    st.sidebar.slider("Taxable Income Base Return Mode", -0.10, 0.10, 0.00),
    st.sidebar.slider("Taxable Income Base Return Max", -0.10, 0.10, 0.02)
)
taxable_income_annual_yield_dist = (
    st.sidebar.slider("Taxable Income Yield Min", 0.00, 0.30, 0.08),
    st.sidebar.slider("Taxable Income Yield Mode", 0.00, 0.30, 0.11),
    st.sidebar.slider("Taxable Income Yield Max", 0.00, 0.30, 0.15)
)
taxable_income_annual_volatility_dist = (
    st.sidebar.slider("Taxable Income Volatility Min", 0.00, 0.30, 0.08),
    st.sidebar.slider("Taxable Income Volatility Mode", 0.00, 0.30, 0.10),
    st.sidebar.slider("Taxable Income Volatility Max", 0.00, 0.30, 0.12)
)

st.sidebar.header("Interest Rate Environment")
initial_market_interest_rate = st.sidebar.slider("Initial Market Interest Rate", 0.00, 0.20, 0.04)
interest_rate_volatility_monthly = st.sidebar.slider("Monthly Interest Rate Volatility", 0.000, 0.01, 0.003)
interest_rate_price_sensitivity = st.sidebar.slider("Price Sensitivity to Rates", 0.0, 10.0, 4.0)
yield_sensitivity_to_rates = st.sidebar.slider("Yield Sensitivity to Rates", 0.0, 1.0, 0.5)

# Income reinvestment slider
income_reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", 0, 100, 100) / 100

# --- Constants derived ---
total_periods = years * periods_per_year
universe_size = 250
max_possible_cefs = universe_size

# --- Initialize CEF types based on slider allocation ---
num_simulations = int(num_simulations)
is_taxable_income_cef = np.random.rand(num_simulations, max_possible_cefs) < taxable_income_ratio

# Assign parameters arrays
cef_base_annual_returns = np.zeros((num_simulations, max_possible_cefs))
cef_annual_volatilities = np.zeros((num_simulations, max_possible_cefs))
cef_base_annual_yields = np.zeros((num_simulations, max_possible_cefs))

for i in range(num_simulations):
    equity_indices = ~is_taxable_income_cef[i]
    num_equity = np.sum(equity_indices)
    if num_equity > 0:
        cef_base_annual_returns[i, equity_indices] = np.random.triangular(
            *equity_annual_base_return_dist, size=num_equity)
        cef_annual_volatilities[i, equity_indices] = np.random.triangular(
            *equity_annual_volatility_dist, size=num_equity)
        cef_base_annual_yields[i, equity_indices] = np.random.triangular(
            *equity_annual_yield_dist, size=num_equity)

    taxable_indices = is_taxable_income_cef[i]
    num_taxable = np.sum(taxable_indices)
    if num_taxable > 0:
        cef_base_annual_returns[i, taxable_indices] = np.random.triangular(
            *taxable_income_annual_base_return_dist, size=num_taxable)
        cef_annual_volatilities[i, taxable_indices] = np.random.triangular(
            *taxable_income_annual_volatility_dist, size=num_taxable)
        cef_base_annual_yields[i, taxable_indices] = np.random.triangular(
            *taxable_income_annual_yield_dist, size=num_taxable)

# Convert annual rates to monthly
cef_monthly_base_returns = cef_base_annual_returns / periods_per_year
cef_monthly_volatilities = cef_annual_volatilities / np.sqrt(periods_per_year)

# Initialize prices
cef_prices = np.random.triangular(3, 21.5, 40, size=(num_simulations, max_possible_cefs))

# Initialize interest rates
monthly_market_interest_rates = np.zeros((num_simulations, total_periods))
monthly_market_interest_rates[:, 0] = initial_market_interest_rate

# Initialize units, cost basis, cash, arrays as before
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

    # Simulate market interest rate
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

    # Monthly returns for each CEF
    returns = np.random.normal(
        cef_monthly_base_returns,
        cef_monthly_volatilities,
        size=(num_simulations, max_possible_cefs)
    )
    returns += price_impact_from_rates[:, np.newaxis]
    cef_prices *= (1 + returns)

    # Adjust yields by interest rates
    yield_adjustment_from_rates = (current_interest_rate_for_month - initial_market_interest_rate) * yield_sensitivity_to_rates
    current_cef_monthly_yields = np.clip(
        cef_base_annual_yields / periods_per_year + yield_adjustment_from_rates[:, np.newaxis] / periods_per_year,
        0.001, 0.20 / periods_per_year
    )

    # SELL: Take profits >1%
    for i in range(num_simulations):
        current_held_cef_indices = np.where(cef_units[i, :num_total_cefs_invested_in[i]] > 0)[0]

        if current_held_cef_indices.size > 0:
            current_avg_cost_per_unit = np.where(cef_units[i, current_held_cef_indices] > 0,
                                                 cost_basis_dollars[i, current_held_cef_indices] / cef_units[i, current_held_cef_indices], 0)
            price_gain_per_unit = (cef_prices[i, current_held_cef_indices] - current_avg_cost_per_unit) / \
                                  np.where(current_avg_cost_per_unit > 0, current_avg_cost_per_unit, 1)

            sell_signals = price_gain_per_unit >= 0.01

            trades_per_month[i, current_month_index] += sell_signals.sum()

            units_to_sell_profit = np.where(sell_signals, cef_units[i, current_held_cef_indices] * 0.5, 0)
            units_to_sell_profit = np.minimum(units_to_sell_profit, cef_units[i, current_held_cef_indices])

            value_sold_from_profit = (units_to_sell_profit * cef_prices[i, current_held_cef_indices]).sum()
            cash[i] += value_sold_from_profit

            for k_idx, j in enumerate(current_held_cef_indices):
                if units_to_sell_profit[k_idx] > 0:
                    proportion_sold = units_to_sell_profit[k_idx] / cef_units[i, j]
                    cef_units[i, j] -= units_to_sell_profit[k_idx]
                    cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)

                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

    # No fund exceeds 3% of portfolio (without selling at loss)
    for i in range(num_simulations):
        current_portfolio_market_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
        three_percent_threshold = current_portfolio_market_value * 0.03

        relevant_cef_indices = np.arange(num_total_cefs_invested_in[i])

        for j in relevant_cef_indices:
            current_cef_value = cef_prices[i, j] * cef_units[i, j]

            if current_cef_value > three_percent_threshold and cef_units[i, j] > 0:

                current_avg_cost_per_unit_j = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0

                if cef_prices[i, j] < current_avg_cost_per_unit_j:
                    continue

                excess_value = current_cef_value - three_percent_threshold
                units_to_sell = excess_value / cef_prices[i, j]

                units_to_sell = min(units_to_sell, cef_units[i, j])

                cef_units[i, j] -= units_to_sell
                cash[i] += (units_to_sell * cef_prices[i, j])

                if (current_cef_value / cef_prices[i, j]) > 1e-9:
                    proportion_sold = units_to_sell / (current_cef_value / cef_prices[i, j])
                    cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)
                else:
                    cost_basis_dollars[i, j] = 0

                if cef_units[i, j] < 1e-9:
                    cef_units[i, j] = 0
                    cost_basis_dollars[i, j] = 0

    # Distributions (market value * monthly yield)
    gross_potential_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)

    distributed_portion = gross_potential_income * 0.7
    monthly_distributed_income[:, current_month_index] = distributed_portion

    reinvestable_portion = gross_potential_income * 0.3 * income_reinvestment_pct
    cash += reinvestable_portion
    monthly_reinvested_income[:, current_month_index] = reinvestable_portion

    # Add new funds & rebalance equal weight (without selling at loss)
    for i in range(num_simulations):
        current_portfolio_market_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]

        while num_total_cefs_invested_in[i] < max_possible_cefs:
            if current_portfolio_market_value <= 0:
                break

            potential_next_num_cefs = num_total_cefs_invested_in[i] + 1
            target_value_for_new_fund = current_portfolio_market_value / potential_next_num_cefs

            new_fund_cash_threshold = target_value_for_new_fund

            if cash[i] >= new_fund_cash_threshold * 0.95:
                num_total_cefs_invested_in[i] += 1
                current_portfolio_market_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]
            else:
                break

        if num_total_cefs_invested_in[i] > 0:
            target_value_per_fund = current_portfolio_market_value / num_total_cefs_invested_in[i]

            for j in range(num_total_cefs_invested_in[i]):
                current_cef_value = cef_prices[i, j] * cef_units[i, j]
                diff_value = target_value_per_fund - current_cef_value

                if diff_value > 0:
                    buyable_units = cash[i] / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_needed = diff_value / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_to_buy = min(units_needed, buyable_units)

                    if units_to_buy > 0:
                        cef_units[i, j] += units_to_buy
                        cost_basis_dollars[i, j] += (units_to_buy * cef_prices[i, j])
                        cash[i] -= (units_to_buy * cef_prices[i, j])

                elif diff_value < 0 and cef_units[i, j] > 0:

                    current_avg_cost = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0

                    if cef_prices[i, j] < current_avg_cost:
                        continue

                    units_to_sell_rebalance = abs(diff_value) / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_to_sell_rebalance = min(units_to_sell_rebalance, cef_units[i, j])

                    if units_to_sell_rebalance > 0:
                        cef_units[i, j] -= units_to_sell_rebalance
                        cash[i] += (units_to_sell_rebalance * cef_prices[i, j])

                        if (cef_units[i, j] + units_to_sell_rebalance) > 1e-9:
                            proportion_sold = units_to_sell_rebalance / (cef_units[i, j] + units_to_sell_rebalance)
                            cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)
                        else:
                            cost_basis_dollars[i, j] = 0

                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

    # Track working capital (cost basis + cash)
    monthly_working_capital[:, current_month_index] = cash + cost_basis_dollars.sum(axis=1)

# --- Results Summary Tables ---

final_portfolio_value = (cef_prices * cef_units).sum(axis=1) + cash
final_working_capital = monthly_working_capital[:, -1]
final_distributed_income = monthly_distributed_income[:, -1]
final_reinvested_income = monthly_reinvested_income[:, -1]

def summary_df(data):
    return pd.DataFrame({
        "Mean": [np.mean(data)],
        "Median": [np.median(data)],
        "5th Percentile": [np.percentile(data, 5)],
        "95th Percentile": [np.percentile(data, 95)]
    }).round(2)

st.header("Portfolio Value Summary")
st.table(summary_df(final_portfolio_value))

st.header("Working Capital Summary (Cost Basis + Cash)")
st.table(summary_df(final_working_capital))

st.header("Monthly Distributed Income Summary (Last Month)")
st.table(summary_df(final_distributed_income))

st.header("Monthly Reinvested Income Summary (Last Month)")
st.table(summary_df(final_reinvested_income))
