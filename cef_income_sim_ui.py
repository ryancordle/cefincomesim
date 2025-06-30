import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Monte Carlo CEF Portfolio Simulation")

def run_monte_carlo_simulation(
    num_cefs_initial,
    initial_investment,
    years,
    periods_per_year,
    num_simulations,
    universe_size,
    taxable_income_ratio,
    equity_annual_base_return_dist,
    equity_annual_yield_dist,
    equity_annual_volatility_dist,
    taxable_income_annual_base_return_dist,
    taxable_income_annual_yield_dist,
    taxable_income_annual_volatility_dist,
    initial_market_interest_rate,
    interest_rate_volatility_monthly,
    interest_rate_price_sensitivity,
    yield_sensitivity_to_rates
):
    """
    Runs the Monte Carlo simulation for CEF portfolio.

    Args:
        All parameters as defined in the Streamlit UI.

    Returns:
        A tuple containing:
        - final_value (np.array): Array of final portfolio values for each simulation.
        - num_total_cefs_invested_in (np.array): Array of total CEFs invested in for each simulation.
        - monthly_working_capital_summary (pd.DataFrame): Summary of monthly working capital.
        - monthly_distributed_income_summary (pd.DataFrame): Summary of monthly distributed income.
        - monthly_reinvested_income_summary (pd.DataFrame): Summary of monthly reinvested income.
    """

    total_periods = years * periods_per_year
    max_possible_cefs = universe_size

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
                        cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)
                        cef_units[i, j] -= units_to_sell_profit[k_idx]

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

                    if (current_cef_value / cef_prices[i,j]) > 1e-9:
                        proportion_sold = units_to_sell / (current_cef_value / cef_prices[i,j])
                        cost_basis_dollars[i, j] -= (cost_basis_dollars[i, j] * proportion_sold)
                    else:
                        cost_basis_dollars[i, j] = 0

                    if cef_units[i, j] < 1e-9:
                        cef_units[i, j] = 0
                        cost_basis_dollars[i, j] = 0

        # Distributions (based on market value and individual CEF monthly yields)
        gross_potential_income = (cef_prices * cef_units * current_cef_monthly_yields).sum(axis=1)

        distributed_portion = gross_potential_income * 0.0
        monthly_distributed_income[:, current_month_index] = distributed_portion

        reinvestable_portion = gross_potential_income * 1.0
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

                if cash[i] >= new_fund_cash_threshold * 0.95: # Ensure enough cash to buy a new fund
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
                            continue # Do not sell at a loss for rebalancing

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
    final_value = (cef_prices * cef_units).sum(axis=1) + cash

    # === MONTHLY WORKING CAPITAL REPORT ===
    monthly_working_capital_df = pd.DataFrame({
        f"Month {i+1}": monthly_working_capital[:, i] for i in range(total_periods)
    }).T

    monthly_working_capital_summary = pd.DataFrame({
        "Mean Working Capital": monthly_working_capital_df.mean(axis=1),
        "Median Working Capital": monthly_working_capital_df.median(axis=1),
        "5th Percentile": monthly_working_capital_df.quantile(0.05, axis=1),
        "95th Percentile": monthly_working_capital_df.quantile(0.95, axis=1)
    }).round(2)

    # === MONTHLY DISTRIBUTED INCOME REPORT ===
    monthly_distributed_income_df = pd.DataFrame({
        f"Month {i+1}": monthly_distributed_income[:, i] for i in range(total_periods)
    }).T
    monthly_distributed_income_summary = pd.DataFrame({
        "Mean Distributed Income": monthly_distributed_income_df.mean(axis=1),
        "Median Distributed Income": monthly_distributed_income_df.median(axis=1),
        "5th Percentile": monthly_distributed_income_df.quantile(0.05, axis=1),
        "95th Percentile": monthly_distributed_income_df.quantile(0.95, axis=1)
    }).round(2)

    # === MONTHLY REINVESTED INCOME REPORT ===
    monthly_reinvested_income_df = pd.DataFrame({
        f"Month {i+1}": monthly_reinvested_income[:, i] for i in range(total_periods)
    }).T
    monthly_reinvested_income_summary = pd.DataFrame({
        "Mean Reinvested Income": monthly_reinvested_income_df.mean(axis=1),
        "Median Reinvested Income": monthly_reinvested_income_df.median(axis=1),
        "5th Percentile": monthly_reinvested_income_df.quantile(0.05, axis=1),
        "95th Percentile": monthly_reinvested_income_df.quantile(0.95, axis=1)
    }).round(2)

    return (
        final_value,
        num_total_cefs_invested_in,
        monthly_working_capital_summary,
        monthly_distributed_income_summary,
        monthly_reinvested_income_summary
    )


# --- Streamlit UI ---
st.title("📈 Monte Carlo CEF Portfolio Simulation")

st.markdown("""
This application simulates the performance of a Closed-End Fund (CEF) portfolio over time using Monte Carlo methods.
Adjust the parameters in the sidebar to see how different assumptions impact the portfolio's growth, income, and composition.
""")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("General Simulation Settings", expanded=True):
    num_cefs_initial = st.number_input("Initial Number of CEFs", min_value=1, value=200, step=10)
    initial_investment = st.number_input("Initial Investment ($)", min_value=1000, value=100000, step=1000)
    years = st.number_input("Simulation Years", min_value=1, value=5, step=1)
    periods_per_year = st.number_input("Periods Per Year (Months)", min_value=1, value=12, step=1)
    num_simulations = st.number_input("Number of Simulations", min_value=100, value=1000, step=100)
    universe_size = st.number_input("Max Possible CEFs in Universe", min_value=1, value=250, step=10)
    taxable_income_ratio = st.slider("Proportion of Taxable Income CEFs", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

with st.sidebar.expander("Equity CEF Parameters"):
    st.markdown("--- **Annual Base Return** ---")
    equity_annual_base_return_min = st.number_input("Equity Return Min", min_value=-0.2, max_value=0.2, value=0.04, step=0.01, format="%.2f")
    equity_annual_base_return_mode = st.number_input("Equity Return Mode", min_value=-0.2, max_value=0.2, value=0.07, step=0.01, format="%.2f")
    equity_annual_base_return_max = st.number_input("Equity Return Max", min_value=-0.2, max_value=0.2, value=0.10, step=0.01, format="%.2f")
    equity_annual_base_return_dist = (equity_annual_base_return_min, equity_annual_base_return_mode, equity_annual_base_return_max)

    st.markdown("--- **Annual Yield** ---")
    equity_annual_yield_min = st.number_input("Equity Yield Min", min_value=0.0, max_value=0.2, value=0.06, step=0.01, format="%.2f")
    equity_annual_yield_mode = st.number_input("Equity Yield Mode", min_value=0.0, max_value=0.2, value=0.08, step=0.01, format="%.2f")
    equity_annual_yield_max = st.number_input("Equity Yield Max", min_value=0.0, max_value=0.2, value=0.10, step=0.01, format="%.2f")
    equity_annual_yield_dist = (equity_annual_yield_min, equity_annual_yield_mode, equity_annual_yield_max)

    st.markdown("--- **Annual Volatility** ---")
    equity_annual_volatility_min = st.number_input("Equity Volatility Min", min_value=0.0, max_value=0.5, value=0.15, step=0.01, format="%.2f")
    equity_annual_volatility_mode = st.number_input("Equity Volatility Mode", min_value=0.0, max_value=0.5, value=0.17, step=0.01, format="%.2f")
    equity_annual_volatility_max = st.number_input("Equity Volatility Max", min_value=0.0, max_value=0.5, value=0.20, step=0.01, format="%.2f")
    equity_annual_volatility_dist = (equity_annual_volatility_min, equity_annual_volatility_mode, equity_annual_volatility_max)

with st.sidebar.expander("Taxable Income CEF Parameters"):
    st.markdown("--- **Annual Base Return** ---")
    taxable_income_annual_base_return_min = st.number_input("Taxable Return Min", min_value=-0.2, max_value=0.2, value=-0.02, step=0.01, format="%.2f")
    taxable_income_annual_base_return_mode = st.number_input("Taxable Return Mode", min_value=-0.2, max_value=0.2, value=0.00, step=0.01, format="%.2f")
    taxable_income_annual_base_return_max = st.number_input("Taxable Return Max", min_value=-0.2, max_value=0.2, value=0.02, step=0.01, format="%.2f")
    taxable_income_annual_base_return_dist = (taxable_income_annual_base_return_min, taxable_income_annual_base_return_mode, taxable_income_annual_base_return_max)

    st.markdown("--- **Annual Yield** ---")
    taxable_income_annual_yield_min = st.number_input("Taxable Yield Min", min_value=0.0, max_value=0.2, value=0.08, step=0.01, format="%.2f")
    taxable_income_annual_yield_mode = st.number_input("Taxable Yield Mode", min_value=0.0, max_value=0.2, value=0.11, step=0.01, format="%.2f")
    taxable_income_annual_yield_max = st.number_input("Taxable Yield Max", min_value=0.0, max_value=0.2, value=0.15, step=0.01, format="%.2f")
    taxable_income_annual_yield_dist = (taxable_income_annual_yield_min, taxable_income_annual_yield_mode, taxable_income_annual_yield_max)

    st.markdown("--- **Annual Volatility** ---")
    taxable_income_annual_volatility_min = st.number_input("Taxable Volatility Min", min_value=0.0, max_value=0.5, value=0.08, step=0.01, format="%.2f")
    taxable_income_annual_volatility_mode = st.number_input("Taxable Volatility Mode", min_value=0.0, max_value=0.5, value=0.10, step=0.01, format="%.2f")
    taxable_income_annual_volatility_max = st.number_input("Taxable Volatility Max", min_value=0.0, max_value=0.5, value=0.12, step=0.01, format="%.2f")
    taxable_income_annual_volatility_dist = (taxable_income_annual_volatility_min, taxable_income_annual_volatility_mode, taxable_income_annual_volatility_max)

with st.sidebar.expander("Interest Rate Environment Parameters"):
    initial_market_interest_rate = st.number_input("Initial Market Interest Rate", min_value=0.0, max_value=0.2, value=0.04, step=0.001, format="%.3f")
    interest_rate_volatility_monthly = st.number_input("Interest Rate Volatility (Monthly)", min_value=0.0, max_value=0.01, value=0.003, step=0.0001, format="%.4f")
    interest_rate_price_sensitivity = st.number_input("Interest Rate Price Sensitivity (Duration-like)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    yield_sensitivity_to_rates = st.number_input("Yield Sensitivity to Rates", min_value=0.0, max_value=1.0, value=0.5, step=0.05)


if st.button("Run Simulation"):
    with st.spinner("Running Monte Carlo simulation... This may take a moment depending on the number of simulations."):
        (
            final_value,
            num_total_cefs_invested_in,
            monthly_working_capital_summary,
            monthly_distributed_income_summary,
            monthly_reinvested_income_summary
        ) = run_monte_carlo_simulation(
            num_cefs_initial,
            initial_investment,
            years,
            periods_per_year,
            num_simulations,
            universe_size,
            taxable_income_ratio,
            equity_annual_base_return_dist,
            equity_annual_yield_dist,
            equity_annual_volatility_dist,
            taxable_income_annual_base_return_dist,
            taxable_income_annual_yield_dist,
            taxable_income_annual_volatility_dist,
            initial_market_interest_rate,
            interest_rate_volatility_monthly,
            interest_rate_price_sensitivity,
            yield_sensitivity_to_rates
        )

    st.header("Simulation Results")

    st.subheader("Portfolio & Income Summary (End of Simulation)")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean Final Portfolio Value", f"${np.mean(final_value):,.2f}")
    col2.metric("Median Final Portfolio Value", f"${np.median(final_value):,.2f}")
    col3.metric("5th Percentile Value", f"${np.percentile(final_value, 5):,.2f}")
    col4.metric("95th Percentile Value", f"${np.percentile(final_value, 95):,.2f}")
    col5.metric("Mean Final Month Distributed Income", f"${np.mean(monthly_distributed_income_summary.iloc[-1]['Mean Distributed Income']):,.2f}")

    st.subheader("Number of Total CEFs Invested In at End of Simulation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean CEFs Invested", f"{np.mean(num_total_cefs_invested_in):.2f}")
    col2.metric("Min CEFs Invested", f"{np.min(num_total_cefs_invested_in)}")
    col3.metric("Max CEFs Invested", f"{np.max(num_total_cefs_invested_in)}")
    col4.metric("Median CEFs Invested", f"{np.median(num_total_cefs_invested_in)}")

    st.subheader("Monthly Working Capital (Cash + Cost Basis)")
    st.dataframe(monthly_working_capital_summary.style.format("${:,.2f}"))
    st.line_chart(monthly_working_capital_summary[["Mean Working Capital", "Median Working Capital", "5th Percentile", "95th Percentile"]])

    st.subheader("Monthly Distributed Income")
    st.dataframe(monthly_distributed_income_summary.style.format("${:,.2f}"))
    st.line_chart(monthly_distributed_income_summary[["Mean Distributed Income", "Median Distributed Income", "5th Percentile", "95th Percentile"]])

    st.subheader("Monthly Reinvested Income")
    st.dataframe(monthly_reinvested_income_summary.style.format("${:,.2f}"))
    st.line_chart(monthly_reinvested_income_summary[["Mean Reinvested Income", "Median Reinvested Income", "5th Percentile", "95th Percentile"]])

