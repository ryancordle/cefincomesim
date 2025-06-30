# 🧮 CEF Income Portfolio Monte Carlo Simulator

This project simulates the growth of a closed-end fund (CEF) income portfolio using a Monte Carlo model with interest rate sensitivity, reinvestment strategy, and tax-aware trading logic. It’s designed for long-term income investors with a reinvestment philosophy and yield-focused approach.

---

## 🚀 What It Does

- Models 1000+ future scenarios over 5–30 years
- Differentiates between equity and taxable income CEFs
- Dynamically simulates monthly distributions, interest rate impacts, and CEF-specific volatility
- Reinvests income and partially sells positions after modest gains (>1%)
- Caps any one fund at 3% of total portfolio to maintain diversification
- Tracks working capital, distributed income, and final market value

---

## 📊 Example Outputs

- Median portfolio value after 19 years: `$1.7M`
- 5th percentile: `$1.49M`
- Yield at end of simulation: ~7% on market value

---

## 📦 Requirements

Install packages with:

```bash
pip install numpy pandas matplotlib
