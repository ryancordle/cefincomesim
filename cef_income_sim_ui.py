import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
num_cefs_initial = 200 # Initial number of CEFs
initial_investment = 180000
years = 5
periods_per_year = 12
total_periods = years * periods_per_year # This is 240 months
num_simulations = 1000 # Adjust as needed for performance

# --- NEW: Define the universe size ---
universe_size = 250
max_possible_cefs = universe_size # Max number of distinct CEFs we can ever invest in

# --- NEW: CEF Type Definitions and Parameters ---
taxable_income_ratio = 0.5 # Proportion of CEFs that are taxable income (CHANGED TO 50%)
# Equity CEF parameters
equity_annual_base_return_dist = (0.04, 0.07, 0.10) # min, mode, max for annual price growth
equity_annual_yield_dist = (0.06, 0.08, 0.10) # min, mode, max for annual yield (CHANGED TO 6-10%)
equity_annual_volatility_dist = (0.15, 0.17, 0.20) # min, mode, max for annual price volatility

# Taxable Income CEF parameters
taxable_income_annual_base_return_dist = (-0.02, 0.00, 0.02) # min, mode, max for annual price growth
taxable_income_annual_yield_dist = (0.08, 0.11, 0.15) # min, mode, max for annual yield
taxable_income_annual_volatility_dist = (0.08, 0.10, 0.12) # min, mode, max for annual price volatility

# Interest Rate Environment Parameters (global for simplicity, could be type-specific if needed)
initial_market_interest_rate = 0.04 # Starting market interest rate (e.g., 4%)
interest_rate_volatility_monthly = 0.003 # Volatility of monthly changes in market interest rate (e.g., 0.3%)
interest_rate_price_sensitivity = 4.0 # How much CEF price changes for a 1% change in interest rate (like duration)
yield_sensitivity_to_rates = 0.5 # How much CEF's yield changes for a 1% change in market interest rate (0.5 means a 1% market rate change translates to a 0.5% CEF yield change)

# --- Initialize CEF Types for each simulation and CEF ---
# True for taxable income, False for equity
is_taxable_income_cef = np.random.rand(num_simulations, max_possible_cefs) < taxable_income_ratio

# --- Initialize CEF-specific parameters based on their type ---
cef_base_annual_returns = np.zeros((num_simulations, max_possible_cefs))
cef_annual_volatilities = np.zeros((num_simulations, max_possible_cefs))
cef_base_annual_yields = np.zeros((num_simulations, max_possible_cefs))

for i in range(num_simulations):
    # Assign equity CEF parameters
    equity_indices = ~is_taxable_income_cef[i]
    num_equity = np.sum(equity_indices)
    if num_equity > 0:
        cef_base_annual_returns[i, equity_indices] = np.random.triangular(*equity_annual_base_return_dist, size=num_equity)
        cef_annual_volatilities[i, equity_indices] = np.random.triangular(*equity_annual_volatility_dist, size=num_equity)
        cef_base_annual_yields[i, equity_indices] = np.random.triangular(*equity_annual_yield_dist, size=num_equity)

    # Assign taxable income CEF parameters
    taxable_indices = is_taxable_income_cef[i]
    num_taxable = np.sum(taxable_indices)
    if num_taxable > 0:
        cef_base_annual_returns[i, taxable_indices] = np.random.triangular(*taxable_income_annual_base_return_dist, size=num_taxable)
        cef_annual_volatilities[i, taxable_indices] = np.random.triangular(*taxable_income_annual_volatility_dist, size=num_taxable)
        cef_base_annual_yields[i, taxable_indices] = np.random.triangular(*taxable_income_annual_yield_dist, size=num_taxable)

# Convert annual rates to monthly
cef_monthly_base_returns = cef_base_annual_returns / periods_per_year
cef_monthly_volatilities = cef_annual_volatilities / np.sqrt(periods_per_year)

# MODIFIED: Initialize CEF prices randomly from a triangular distribution ($3-$40, mode at $21.5)
cef_prices = np.random.triangular(3, 21.5, 40, size=(num_simulations, max_possible_cefs))

# Track the simulated market interest rates over time for each simulation
monthly_market_interest_rates = np.zeros((num_simulations, total_periods))
monthly_market_interest_rates[:, 0] = initial_market_interest_rate # Initialize first month's interest rate

# Initialize only the first num_cefs_initial funds
# Each initial CEF is assumed to be part of the initial investment
initial_units_per_cef_target_value = (initial_investment / num_cefs_initial)

cef_units = np.zeros((num_simulations, max_possible_cefs))
# Distribute initial investment by targeting a value per CEF
# Divide initial target value by random initial price to get units for each CEF
cef_units[:, :num_cefs_initial] = initial_units_per_cef_target_value / cef_prices[:, :num_cefs_initial]

# Cost basis tracking: Initial investment is the initial cost basis for units purchased
cost_basis_dollars = cef_units * cef_prices # Total cost basis in dollars for each CEF holding

cash = np.zeros(num_simulations)

monthly_distributed_income = np.zeros((num_simulations, total_periods))
monthly_reinvested_income = np.zeros((num_simulations, total_periods))
monthly_working_capital = np.zeros((num_simulations, total_periods))

trades_per_month = np.zeros((num_simulations, total_periods))

num_total_cefs_invested_in = np.full(num_simulations, num_cefs_initial, dtype=int)


# Simulation loop
for month in range(1, total_periods + 1):
    current_month_index = month - 1

    # --- Simulate Market Interest Rate for the Current Month ---
    if current_month_index > 0:
        monthly_market_interest_rates[:, current_month_index] = np.clip(
            monthly_market_interest_rates[:, current_month_index - 1] + np.random.normal(0, interest_rate_volatility_monthly, num_simulations),
            0.005, 0.15
        )
    current_interest_rate_for_month = monthly_market_interest_rates[:, current_month_index]

    # --- Calculate Interest Rate Change (delta) ---
    if current_month_index == 0:
        interest_rate_delta = np.zeros(num_simulations)
    else:
        interest_rate_delta = current_interest_rate_for_month - monthly_market_interest_rates[:, current_month_index - 1]

    # --- Adjust Price Returns based on Interest Rate Delta and CEF's base return/volatility ---
    # Negative impact: higher rates -> lower prices; lower rates -> higher prices
    price_impact_from_rates = -interest_rate_delta * interest_rate_price_sensitivity

    # Simulate monthly returns for each CEF based on its type's base return and volatility
    # This combines the base return, idiosyncratic volatility, and the interest rate impact
    returns = np.random.normal(
        cef_monthly_base_returns, # Base monthly price return for each CEF type
        cef_monthly_volatilities, # Monthly volatility for each CEF type
        size=(num_simulations, max_possible_cefs)
    )
    returns += price_impact_from_rates[:, np.newaxis] # Add interest rate impact
    cef_prices *= (1 + returns)

    # --- Adjust CEF Monthly Yields based on Current Market Interest Rate and CEF's base yield ---
    yield_adjustment_from_rates = (current_interest_rate_for_month - initial_market_interest_rate) * yield_sensitivity_to_rates
    current_cef_monthly_yields = np.clip(
        cef_base_annual_yields / periods_per_year + yield_adjustment_from_rates[:, np.newaxis] / periods_per_year,
        0.001, 0.20 / periods_per_year # Minimum 0.1% monthly, Max 20% annual
    )

    # SELL: Take profits >1%
    for i in range(num_simulations):
        current_held_cef_indices = np.where(cef_units[i, :num_total_cefs_invested_in[i]] > 0)[0]

        if current_held_cef_indices.size > 0:
            current_avg_cost_per_unit = np.where(
                cef_units[i, current_held_cef_indices] > 0,
                cost_basis_dollars[i, current_held_cef_indices] / cef_units[i, current_held_cef_indices],
                0
            )
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

    # No Fund Exceeds 3% of Portfolio (without selling at a loss)
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

    # Distributions (based on market value and individual CEF monthly yields)
    gross_potential_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)

    distributed_portion = gross_potential_income * 0.7
    monthly_distributed_income[:, current_month_index] = distributed_portion

    reinvestable_portion = gross_potential_income * .3
    cash += reinvestable_portion
    monthly_reinvested_income[:, current_month_index] = reinvestable_portion

    # Add New Funds & Rebalance to Equal Weight (without selling at a loss)
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

        # GLOBAL REBALANCE TO MAINTAIN EQUAL WEIGHTING
        if num_total_cefs_invested_in[i] > 0:
            target_value_per_fund = current_portfolio_market_value / num_total_cefs_invested_in[i]

            for j in range(num_total_cefs_invested_in[i]):
                current_cef_value = cef_prices[i, j] * cef_units[i, j]

                diff_value = target_value_per_fund - current_cef_value

                # If we need to buy (diff_value > 0)
                if diff_value > 0:
                    buyable_units_with_cash = cash[i] / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_needed = diff_value / cef_prices[i, j] if cef_prices[i, j] > 0 else 0
                    units_to_buy = min(units_needed, buyable_units_with_cash)

                    if units_to_buy > 0:
                        cef_units[i, j] += units_to_buy
                        cost_basis_dollars[i, j] += (units_to_buy * cef_prices[i, j])
                        cash[i] -= (units_to_buy * cef_prices[i, j])

                # If we need to sell (diff_value < 0), meaning fund is overweighted
                elif diff_value < 0 and cef_units[i, j] > 0:

                    current_avg_cost_per_unit_j = cost_basis_dollars[i, j] / cef_units[i, j] if cef_units[i, j] > 0 else 0

                    if cef_prices[i, j] < current_avg_cost_per_unit_j:
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

    # Track working capital (Cash + Total Cost Basis of currently held positions) every month
    monthly_working_capital[:, current_month_index] = cash + cost_basis_dollars.sum(axis=1)

# === Summary Output ===

# Final portfolio value (using market value)
final_value = (cef_prices * cef_units).sum(axis=1) + cash
print("\n=== Portfolio & Income Summary (End of Simulation) ===")
print(f"Mean Final Portfolio (Market Value): ${np.mean(final_value):,.2f}")
print(f"Median Final Portfolio (Market Value): ${np.median(final_value):,.2f}")
print(f"5th Percentile: ${np.percentile(final_value, 5):,.2f}")
print(f"95th Percentile: ${np.percentile(final_value, 95):,.2f}")
print(f"Mean Final Month Distributed Income: ${np.mean(monthly_distributed_income[:, -1]):,.2f}")

# === Number of Total CEFs Invested In at End of Simulation ===
print(f"\nMean Total Number of CEFs Invested In at End: {np.mean(num_total_cefs_invested_in):.2f}")
print(f"Min Total Number of CEFs Invested In at End: {np.min(num_total_cefs_invested_in)}")
print(f"Max Total Number of CEFs Invested In at End: {np.max(num_total_cefs_invested_in)}")
print(f"Median Total Number of CEFs Invested In at End: {np.median(num_total_cefs_invested_in)}")

# === MONTHLY WORKING CAPITAL REPORT ===
print("\n=== Monthly Working Capital (Cash + Cost Basis) Report ===")
monthly_working_capital_df = pd.DataFrame({
    f"Month {i+1}": monthly_working_capital[:, i] for i in range(total_periods)
}).T

monthly_working_capital_summary = pd.DataFrame({
    "Mean Working Capital": monthly_working_capital_df.mean(axis=1),
    "Median Working Capital": monthly_working_capital_df.median(axis=1),
    "5th Percentile": monthly_working_capital_df.quantile(0.05, axis=1),
    "95th Percentile": monthly_working_capital_df.quantile(0.95, axis=1)
}).round(2)
print(monthly_working_capital_summary)

# === MONTHLY DISTRIBUTED INCOME REPORT ===
print("\n=== Monthly Distributed
