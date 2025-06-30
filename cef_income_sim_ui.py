import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="CEF Income Simulator", layout="wide")
st.title("📈 CEF Income Portfolio Simulator")
st.markdown("""
This app simulates long-term growth and income of a closed-end fund (CEF) portfolio based on your investment philosophy.

*Adjust the inputs below and explore the results of a Monte Carlo simulation based on price growth, yield, and interest rates.*
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Settings")

initial_investment = st.sidebar.number_input("Initial Investment ($)", 10000, 1000000, 180000, step=10000)
years = st.sidebar.slider("Years to Simulate", 1, 40, 19)
simulations = st.sidebar.slider("Number of Simulations", 100, 5000, 1000, step=100)

st.sidebar.header("Portfolio Composition")
taxable_income_ratio = st.sidebar

