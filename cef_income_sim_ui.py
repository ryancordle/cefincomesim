import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app simulates long-term growth and income of a closed-end fund (CEF) portfolio based on your investment philosophy.

*Adjust inputs on the sidebar and click Run Simulation.*
""")

# Sidebar inputs
st.sidebar.header("Simulation Settings")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500, step=100)

st.sidebar.header("Portfolio Composition")
taxable_income_ratio = st.sidebar.slider("% of Portfolio in Taxable Income CEFs", 0, 100, 50) / 100

st.sidebar.header("Yield Assumptions (Annual %)")
equity_yield_min, equity_yield_mode, equity_yield_max = st.sidebar.slider("Equity CEF Yield Range (%)", 4, 12, (6, 8), step=1)
taxable_yield_min, taxable_yield_mode, taxable_yield_max = st.sidebar.slider("Taxable Income CEF Yield Range (%)", 6, 20, (8, 11), step=1)

st.sidebar.header("Price Growth Assumptions (Annual %)")
equity_growth_min, equity_growth_mode, equity_growth_max = st.sidebar.slider("Equity CEF Growth Range (%)", -5, 15, (4, 7), step=1)
taxable_growth_min, taxable_growth_mode, taxable_growth_max = st.sidebar.slider("Taxable Income CEF Growth Range (%)", -5, 5, (-2, 0), step=1)

reinvestment_pct = st.sidebar.slider("Income Reinvestment Percentage (%)", 0, 100, 30) / 100

run_sim = st.sidebar.button("▶️ Run Simulation")

if run_sim:
    # Convert % to decimals
    equity_yield_dist = (equity_yield_min / 100, equity_yield_mode / 100, equity_yield_max / 100)
    taxable_yield_dist = (taxable_yield_min / 100, taxable_yield_mode / 100, taxable_yield_max / 100)
    equity_growth_dist = (equity_growth_min / 100, equity_growth_mode / 100, equity_growth_max / 100)
    taxable_growth_dist = (taxable_growth_min / 100, taxable_growth_mode / 100, taxable_growth_max / 100)

    # Constants from your original sim:
    num_cefs_initial = 200
    universe_size = 250
    max_possible_cefs = universe_size
    periods_per_year = 12
    total_periods = years * periods_per_year

    # Parameters related to market and funds
    initial_market_interest_rate = 0.04
    interest_rate_volatility_monthly = 0.003
    interest_rate_price_sensitivity = 4.0
    yield_sensitivity_to_rates = 0.5

    # Seed RNG for reproducibility
    rng = np.random.default_rng(42)

    # Pre-allocate arrays for storing results
    final_portfolio_values = np.zeros(simulations)
    final_working_capitals = np.zeros(simulations)
    final_annual_incomes = np.zeros(simulations)

    for sim_idx in range(simulations):
        # Assign types to funds (True=Taxable Income, False=Equity)
        is_taxable_income_cef = rng.random(max_possible_cefs) < taxable_income_ratio

        # Generate base returns and yields via triangular distributions for each fund
        cef_base_annual_returns = np.zeros(max_possible_cefs)
        cef_annual_volatilities = np.zeros(max_possible_cefs)
        cef_base_annual_yields = np.zeros(max_possible_cefs)

        # Equity parameters
        equity_indices = ~is_taxable_income_cef
        num_equity = np.sum(equity_indices)
        if num_equity > 0:
            cef_base_annual_returns[equity_indices] = rng.triangular(equity_growth_min/100, equity_growth_mode/100, equity_growth_max/100, size=num_equity)
            cef_annual_volatilities[equity_indices] = rng.triangular(0.15, 0.17, 0.20, size=num_equity)
            cef_base_annual_yields[equity_indices] = rng.triangular(equity_yield_min/100, equity_yield_mode/100, equity_yield_max/100, size=num_equity)

        # Taxable income parameters
        taxable_indices = is_taxable_income_cef
        num_taxable = np.sum(taxable_indices)
        if num_taxable > 0:
            cef_base_annual_returns[taxable_indices] = rng.triangular(taxable_growth_min/100, taxable_growth_mode/100, taxable_growth_max/100, size=num_taxable)
            cef_annual_volatilities[taxable_indices] = rng.triangular(0.08, 0.10, 0.12, size=num_taxable)
            cef_base_annual_yields[taxable_indices] = rng.triangular(taxable_yield_min/100, taxable_yield_mode/100, taxable_yield_max/100, size=num_taxable)

        # Monthly versions of returns and volatilities
        cef_monthly_base_returns = cef_base_annual_returns / periods_per_year
        cef_monthly_volatilities = cef_annual_volatilities / np.sqrt(periods_per_year)

        # Initialize prices ($3-$40, mode at $21.5)
        cef_prices = rng.triangular(3, 21.5, 40, size=max_possible_cefs)

        # Initialize units by spreading initial investment evenly across initial CEFs
        initial_units_per_cef_target_value = initial_investment / num_cefs_initial
        cef_units = np.zeros(max_possible_cefs)
        cef_units[:num_cefs_initial] = initial_units_per_cef_target_value / cef_prices[:num_cefs_initial]

        # Cost basis = units * prices initially
        cost_basis_dollars = cef_units * cef_prices

        cash = 0.0

        # Track working capital and income monthly
        monthly_working_capital = []
        monthly_income = []

        # Initialize market interest rate
        current_interest_rate = initial_market_interest_rate

        num_total_cefs_invested_in = num_cefs_initial

        for month in range(total_periods):
            # Simulate market interest rate changes
            delta_ir = rng.normal(0, interest_rate_volatility_monthly)
            current_interest_rate = np.clip(current_interest_rate + delta_ir, 0.005, 0.15)
            interest_rate_delta = delta_ir

            # Price impact from interest rates
            price_impact_from_rates = -interest_rate_delta * interest_rate_price_sensitivity

            # Simulate returns for each fund
            returns = rng.normal(cef_monthly_base_returns, cef_monthly_volatilities)
            returns += price_impact_from_rates

            # Update prices
            cef_prices = cef_prices * (1 + returns)
            cef_prices = np.clip(cef_prices, 0.01, None)  # no zero or negative prices

            # Adjust yields based on interest rates
            yield_adjustment = (current_interest_rate - initial_market_interest_rate) * yield_sensitivity_to_rates
            current_cef_monthly_yields = np.clip(
                cef_base_annual_yields / periods_per_year + yield_adjustment / periods_per_year,
                0.001,
                0.20 / periods_per_year
            )

            # Calculate distributions (income)
            gross_potential_income = np.sum(cef_prices * cef_units * current_cef_monthly_yields)

            # Split income into distributed and reinvested portions
            distributed_income = gross_potential_income * (1 - reinvestment_pct)
            reinvested_income = gross_potential_income * reinvestment_pct

            # Add reinvested income to cash for buying more units
            cash += reinvested_income

            # Add distributed income to monthly income record (annualized)
            monthly_income.append(distributed_income * periods_per_year)

            # Sell any CEF holdings that have gains >1% to take profits (sell 50% of units)
            for j in range(num_total_cefs_invested_in):
                if cef_units[j] <= 0:
                    continue
                avg_cost = cost_basis_dollars[j] / cef_units[j] if cef_units[j] > 0 else 0
                gain_pct = (cef_prices[j] - avg_cost) / avg_cost if avg_cost > 0 else 0
                if gain_pct >= 0.01:
                    units_to_sell = cef_units[j] * 0.5
                    value_sold = units_to_sell * cef_prices[j]
                    cef_units[j] -= units_to_sell
                    cost_basis_dollars[j] -= cost_basis_dollars[j] * (units_to_sell / (units_to_sell + cef_units[j]))
                    cash += value_sold

            # Add new CEFs if cash sufficient, rebalance equally
            current_portfolio_value = np.sum(cef_prices * cef_units) + cash
            while num_total_cefs_invested_in < max_possible_cefs:
                target_value = current_portfolio_value / (num_total_cefs_invested_in + 1)
                if cash >= target_value * 0.95:
                    num_total_cefs_invested_in += 1
                else:
                    break

            if num_total_cefs_invested_in > 0:
                target_value_per_cef = current_portfolio_value / num_total_cefs_invested_in
                for j in range(num_total_cefs_invested_in):
                    current_value = cef_prices[j] * cef_units[j]
                    diff = target_value_per_cef - current_value

                    # Buy units if underweight and cash available
                    if diff > 0:
                        units_to_buy = min(diff / cef_prices[j], cash / cef_prices[j])
                        cef_units[j] += units_to_buy
                        cost_basis_dollars[j] += units_to_buy * cef_prices[j]
                        cash -= units_to_buy * cef_prices[j]

                    # Sell units if overweight and price above cost basis
                    elif diff < 0 and cef_units[j] > 0:
                        avg_cost = cost_basis_dollars[j] / cef_units[j] if cef_units[j] > 0 else 0
                        if cef_prices[j] > avg_cost:
                            units_to_sell = min(abs(diff) / cef_prices[j], cef_units[j])
                            cef_units[j] -= units_to_sell
                            cost_basis_dollars[j] -= cost_basis_dollars[j] * (units_to_sell / (units_to_sell + cef_units[j]))
                            cash += units_to_sell * cef_prices[j]

            # Track working capital = cost basis + cash
            working_capital = np.sum(cost_basis_dollars) + cash
            monthly_working_capital.append(working_capital)

        # Store final results for this simulation
        final_portfolio_values[sim_idx] = np.sum(cef_prices * cef_units) + cash
        final_working_capitals[sim_idx] = np.sum(cost_basis_dollars) + cash
        final_annual_incomes[sim_idx] = monthly_income[-1] if monthly_income else 0

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        "Metric": ["Median", "Mean", "5th Percentile", "95th Percentile"],
        "Final Portfolio Market Value": [
            f"${np.median(final_portfolio_values):,.0f}",
            f"${np.mean(final_portfolio_values):,.0f}",
            f"${np.percentile(final_portfolio_values, 5):,.0f}",
            f"${np.percentile(final_portfolio_values, 95):,.0f}"
        ],
        "Final Working Capital": [
            f"${np.median(final_working_capitals):,.0f}",
            f"${np.mean(final_working_capitals):,.0f}",
            f"${np.percentile(final_working_capitals, 5):,.0f}",
            f"${np.percentile(final_working_capitals, 95):,.0f}"
        ],
        "Final Annual Income": [
            f"${np.median(final_annual_incomes):,.0f}",
            f"${np.mean(final_annual_incomes):,.0f}",
            f"${np.percentile(final_annual_incomes, 5):,.0f}",
            f"${np.percentile(final_annual_incomes, 95):,.0f}"
        ],
    })

    st.subheader("📊 Simulation Summary Table")
    st.dataframe(summary_df.set_index("Metric"))

else:
    st.info("Adjust the parameters on the left and click **Run Simulation** to begin.")
