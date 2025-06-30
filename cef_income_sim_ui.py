import streamlit as st
import numpy as np
import pandas as pd

st.title("CEF Portfolio Monte Carlo Simulation")

# --- Sliders for user input ---
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=10000, max_value=1000000, value=180000, step=5000)
num_cefs_initial = st.sidebar.slider("Initial Number of CEFs", min_value=50, max_value=250, value=200, step=10)
years = st.sidebar.slider("Years to Simulate", min_value=1, max_value=30, value=5, step=1)
num_simulations = st.sidebar.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)
income_reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", min_value=0, max_value=100, value=30, step=5) / 100

periods_per_year = 12
total_periods = years * periods_per_year
universe_size = 250
max_possible_cefs = universe_size

# CEF type ratios and parameters
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

# Initialize CEF types
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
        cef_monthly_base_returns, 
        cef_monthly_volatilities, 
        size=(num_simulations, max_possible_cefs)
    )
    returns += price_impact_from_rates[:, np.newaxis]
    cef_prices *= (1 + returns)

    yield_adjustment_from_rates = (current_interest_rate_for_month - initial_market_interest_rate) * yield_sensitivity_to_rates
    current_cef_monthly_yields = np.clip(
        cef_base_annual_yields / periods_per_year + yield_adjustment_from_rates[:, np.newaxis] / periods_per_year,
        0.001, 0.20 / periods_per_year
    )

    # SELL profits >1%
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

    # No fund >3% without selling at loss
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
                if (current_cef_value / cef_prices[i,j]) > 1e-9:
                    proportion_sold = units_to_sell / (current_cef_value / cef_prices[i,j])
                    cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)
                else:
                    cost_basis_dollars[i, j] = 0
                if cef_units[i, j] < 1e-9:
                    cef_units[i, j] = 0
                    cost_basis_dollars[i, j] = 0

    gross_potential_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)

    distributed_portion = gross_potential_income * (1 - income_reinvestment_pct)
    monthly_distributed_income[:, current_month_index] = distributed_portion

    reinvestable_portion = gross_potential_income * income_reinvestment_pct
    cash += reinvestable_portion
    monthly_reinvested_income[:, current_month_index] = reinvestable_portion

    # Add new funds & rebalance equal weight (no loss selling)
    for i in range(num_simulations):
        current_portfolio_market_value = (cef_prices[i, :] * cef_units[i, :]).sum() + cash[i]

        while num_total_cefs_invested_in[i] < max_possible_cefs:
            if current_portfolio_market_value <= 0:
                break
            potential_next_num_cefs = num_total_cefs_invested_in[i] + 1
            target_value_for_a_single_new_fund = current_portfolio_market_value / potential_next_num_cefs
            new_fund_cash_threshold = target_value_for_a_single_new_fund
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
                    buyable_units_with_cash = cash[i] / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_needed = diff_value / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_to_buy = min(units_needed, buyable_units_with_cash)
                    if units_to_buy > 0:
                        cef_units[i, j] += units_to_buy
                        cost_basis_dollars[i, j] += (units_to_buy * cef_prices[i, j])
                        cash[i] -= (units_to_buy * cef_prices[i, j])
                elif diff_value < 0 and cef_units[i, j] > 0:
                    current_avg_cost_per_unit_j = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i]()_
