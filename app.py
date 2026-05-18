import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import pandas as pd
import matplotlib.dates as mdates

# Page title shown at the top
st.title("Stock Market Dashboard")

#sidebar input where user choose stock
# AAPL is defaul
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")

#sidebar drop down to choose period 
period = st.sidebar.selectbox(
    "Time Period",
    ["6mo", "1y", "2y", "5y"]
)
benchmark = st.sidebar.text_input("Benchmark Ticker", "^GSPC")

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=20.0,
    value=4.5,
    step=0.1
)
forecast_days = st.sidebar.slider(
    "Forecast Days",
    30,
    365,
    90
)

# sidebarslider to choose average window
short_ma = st.sidebar.slider("Short Moving Average", 5, 50, 20)

long_ma = st.sidebar.slider("Long Moving Average", 50, 200, 50)

# download stock price data
data = yf.download(ticker, period=period, progress=False)
if data.empty:
    st.error("No data found. Check the ticker symbol and try again.")
    st.stop()

benchmark_data = yf.download(benchmark, period=period, progress=False)

if benchmark_data.empty:
    st.error("No benchmark data found. Check the benchmark ticker.")
    st.stop()

# converts a one-column data frame to a series
close = data["Close"].squeeze()
benchmark_close = benchmark_data["Close"].squeeze()

data["Short MA"] = close.rolling(short_ma).mean()
data["Long MA"] = close.rolling(long_ma).mean()

data["Returns"] = close.pct_change()
benchmark_returns = benchmark_close.pct_change()
# Normalise both series so they start at 100
normalised_stock = close / close.iloc[0] * 100
normalised_benchmark = benchmark_close / benchmark_close.iloc[0] * 100

# Align stock and benchmark returns on common dates
combined_returns = data["Returns"].to_frame("Stock").join(
    benchmark_returns.to_frame("Benchmark"),
    how="inner"
).dropna()

# Covariance matrix
cov_matrix = combined_returns.cov()

# Beta = Cov(stock, benchmark) / Var(benchmark)
beta = (
    cov_matrix.loc["Stock", "Benchmark"]
    / cov_matrix.loc["Benchmark", "Benchmark"]
)

# Annualised benchmark return
benchmark_annual_return = (
    combined_returns["Benchmark"].mean() * 252 * 100
)


data["Rolling Volatility"] = data["Returns"].rolling(21).std() * np.sqrt(252) * 100

data["Running Max"] = close.cummax()
data["Drawdown"] = (close / data["Running Max"] - 1) * 100

# uses root 252 as theres around 252 trading days a year 
latest_price = close.iloc[-1]
total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
annual_volatility = data["Returns"].std() * np.sqrt(252) * 100
average_daily_return = data["Returns"].mean()
annual_return = average_daily_return * 252 * 100
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
max_drawdown = data["Drawdown"].min()
# 5th percentile = 95% one-day VaR
var_95 = np.percentile(data["Returns"].dropna() * 100, 5)

# Average of returns worse than the VaR (Expected Shortfall)
expected_shortfall = (
    data["Returns"].dropna()[data["Returns"].dropna() * 100 <= var_95]
    .mean() * 100
)

# Alpha = Stock return - Beta × Benchmark return
alpha = annual_return - beta * benchmark_annual_return

# show the data
st.subheader("Raw Stock Data")
st.write(data)
st.subheader("Key Statistics")

st.write(f"Latest Price: ${latest_price:.2f}")
st.write(f"{period} Return: {total_return:.2f}%")
st.write(f"Annualised Return: {annual_return:.2f}%")
st.write(f"Annual Volatility: {annual_volatility:.2f}%")
st.write(f"Risk-Free Rate: {risk_free_rate:.2f}%")
st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")
st.write(f"Beta vs {benchmark}: {beta:.2f}")
st.write(f"Alpha vs {benchmark}: {alpha:.2f}%")

st.write(f"Maximum Drawdown: {max_drawdown:.2f}%")
st.write(f"95% 1-Day VaR: {var_95:.2f}%")
st.write(f"Expected Shortfall: {expected_shortfall:.2f}%")
st.subheader("VaR Interpretation")

var_table = {
    "95% 1-Day VaR": [
        "> -1%",
        "-1% to -2%",
        "-2% to -4%",
        "-4% to -6%",
        "< -6%"
    ],
    "Interpretation": [
        "Very low daily risk",
        "Low daily risk",
        "Moderate daily risk",
        "High daily risk",
        "Very high daily risk"
    ]
}

st.table(var_table)

st.subheader("Beta Interpretation")
beta_table = {
    "Beta Range": [
        "< 0",
        "0 - 0.5",
        "0.5 - 1.0",
        "1.0",
        "1.0 - 1.5",
        "> 1.5"
    ],
    "Interpretation": [
        "Moves opposite to the market",
        "Very defensive",
        "Less volatile than the market",
        "Moves roughly with the market",
        "Moderately aggressive",
        "Highly aggressive"
    ]
}

st.table(beta_table)

st.subheader("Closing Price with Moving Averages")

fig, ax = plt.subplots()

ax.plot(data.index, close, label="Close")
ax.plot(data.index, data["Short MA"], label=f"{short_ma}-day MA")
ax.plot(data.index, data["Long MA"], label=f"{long_ma}-day MA")

ax.set_title(f"{ticker} Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid()

st.pyplot(fig)

st.subheader("Trend Forecast")

# Create x-values: 0, 1, 2, ..., n-1
x = np.arange(len(close))

# Fit a straight line to historical prices
coefficients = np.polyfit(x, close, 1)

slope = coefficients[0]
intercept = coefficients[1]

# Historical fitted trend
fitted_trend = np.polyval(coefficients, x)

# Future x-values
future_x = np.arange(len(close) + forecast_days)

# Historical + future forecast
forecast_prices = np.polyval(coefficients, future_x)

fig_forecast, ax_forecast = plt.subplots()

# Actual historical prices
ax_forecast.plot(
    close.index,
    close,
    label="Actual Price",
    linewidth=2
)

# Fitted trend over history
ax_forecast.plot(
    close.index,
    fitted_trend,
    linestyle="--",
    linewidth=2,
    label="Trend Line"
)

# Create future dates
future_dates = pd.date_range(
    start=close.index[0],
    periods=len(future_x),
    freq="B"  # business days
)

# Forecast line
ax_forecast.plot(
    future_dates,
    forecast_prices,
    linestyle=":",
    linewidth=3,
    label="Forecast"
)

ax_forecast.set_title(f"{ticker} Trend Forecast")
ax_forecast.set_xlabel("Date")
ax_forecast.set_ylabel("Price")
ax_forecast.legend()
ax_forecast.grid()
ax_forecast.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax_forecast.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

fig_forecast.autofmt_xdate(rotation=45)
ax_forecast.tick_params(axis="x", labelsize=8)

st.pyplot(fig_forecast)

forecast_final_price = forecast_prices[-1]

st.write(f"Forecast Price in {forecast_days} Trading Days: ${forecast_final_price:.2f}")
st.write(f"Average Daily Trend: ${slope:.4f}")

st.subheader("Performance vs Benchmark")

fig_bench, ax_bench = plt.subplots()

ax_bench.plot(data.index, normalised_stock, label=ticker)
ax_bench.plot(benchmark_data.index, normalised_benchmark, label=benchmark)

ax_bench.set_title(f"{ticker} vs {benchmark}")
ax_bench.set_xlabel("Date")
ax_bench.set_ylabel("Normalised Price (Start = 100)")
ax_bench.legend()
ax_bench.grid()

st.pyplot(fig_bench)
st.subheader("Security Characteristic Line")

fig_scl, ax_scl = plt.subplots()

# Convert returns to percentages for easier interpretation
x = combined_returns["Benchmark"] * 100
y = combined_returns["Stock"] * 100

# Scatter plot of stock returns vs benchmark returns
ax_scl.scatter(x, y, alpha=0.5, marker="x")

# Fit regression line: y = beta*x + intercept
slope, intercept = np.polyfit(x, y, 1)

regression_line = slope * x + intercept

ax_scl.plot(x, regression_line,color="red",linewidth=2, label=f"Beta = {slope:.2f}")

ax_scl.axhline(0, linestyle="--", linewidth=1, color="black")
ax_scl.axvline(0, linestyle="--", linewidth=1, color="black")

ax_scl.set_title(f"{ticker} Returns vs {benchmark} Returns")
ax_scl.set_xlabel(f"{benchmark} Daily Return (%)")
ax_scl.set_ylabel(f"{ticker} Daily Return (%)")
ax_scl.legend()
ax_scl.grid()

st.pyplot(fig_scl)


st.subheader("Daily Returns")

fig2, ax2 = plt.subplots()

ax2.plot(data.index, data["Returns"] * 100) # * 100 for %

ax2.set_title(f"{ticker} Daily Returns")
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Return (%)")
ax2.axhline(0, linestyle="--", color="green",linewidth=1)
ax2.grid()

st.pyplot(fig2)

st.subheader("Time Series Autocorrelation")
st.write("""
A time series is a sequence of observations recorded over time, such as daily stock returns.

A lag looks back a certain number of periods.

- Lag 1: Does today's return help predict tomorrow's return?
- Lag 5: Do returns from one week ago still influence returns today?
- Lag 20: Do returns from one month ago still influence returns today?

Autocorrelation measures whether returns are related to their past values.

Interpretation of Lag 1:

- Positive autocorrelation:
  If the stock rises today, it is more likely to rise again tomorrow.
  If the stock falls today, it is more likely to fall again tomorrow.
  This is known as momentum.

- Negative autocorrelation:
  If the stock rises today, it is more likely to fall tomorrow.
  If the stock falls today, it is more likely to rebound tomorrow.
  This is known as mean reversion.

- Near zero:
  Today's return provides little information about tomorrow's return.
  This is consistent with an efficient market.
""")


clean_returns = data["Returns"].dropna()

lag_1_autocorr = clean_returns.autocorr(lag=1)
lag_5_autocorr = clean_returns.autocorr(lag=5)
lag_20_autocorr = clean_returns.autocorr(lag=20)

st.write(f"Lag 1 Autocorrelation: {lag_1_autocorr:.3f}")
st.write(f"Lag 5 Autocorrelation: {lag_5_autocorr:.3f}")
st.write(f"Lag 20 Autocorrelation: {lag_20_autocorr:.3f}")

autocorr_table = {
    "Autocorrelation": [
        "Positive",
        "Near zero",
        "Negative"
    ],
    "Meaning": [
        "Returns tend to continue in the same direction",
        "Little evidence of linear time dependence",
        "Returns tend to reverse direction"
    ],
    "Finance Interpretation": [
        "Momentum behaviour",
        "Market may be fairly efficient over this horizon",
        "Mean-reversion behaviour"
    ]
}

st.table(autocorr_table)

st.subheader("Lag-1 Return Relationship")

# Create x (yesterday's return) and y (today's return)
x_lag = clean_returns.shift(1).dropna() * 100
y_current = clean_returns.iloc[1:] * 100

# Fit a straight line: y = m x + c
coefficients = np.polyfit(x_lag, y_current, 1)
fitted_line = np.polyval(coefficients, x_lag)

slope = coefficients[0]
intercept = coefficients[1]

fig_acf, ax_acf = plt.subplots()

# Scatter points as crosses
ax_acf.scatter(
    x_lag,
    y_current,
    alpha=0.5,
    marker="x",
    label="Daily Returns"
)

# Fitted regression line in red
ax_acf.plot(
    x_lag,
    fitted_line,
    color="red",
    linewidth=2,
    label=f"Fitted Line (Slope = {slope:.3f})"
)

# Reference lines at zero
ax_acf.axhline(0, linestyle="--", linewidth=1)
ax_acf.axvline(0, linestyle="--", linewidth=1)

ax_acf.set_title(f"{ticker}: Today's Return vs Yesterday's Return")
ax_acf.set_xlabel("Yesterday's Return (%)")
ax_acf.set_ylabel("Today's Return (%)")
ax_acf.legend()
ax_acf.grid()
st.write(f"Regression Slope: {slope:.4f}")

st.pyplot(fig_acf)

st.subheader("Autocorrelation by Lag")

lags = range(1, 21)

autocorr_values = [
    clean_returns.autocorr(lag=lag)
    for lag in lags
]

n = len(clean_returns)
significance_bound = 1.96 / np.sqrt(n)

fig_lags, ax_lags = plt.subplots()

# Spikes from zero to each autocorrelation value
ax_lags.vlines(
    lags,
    0,
    autocorr_values,
    linewidth=2,
    alpha=0.7
)

# Connecting line
ax_lags.plot(
    lags,
    autocorr_values,
    linewidth=1.5,
    alpha=0.8,
    label="Autocorrelation"
)

# Dots at the tips
ax_lags.scatter(
    lags,
    autocorr_values,
    s=50
)

# Zero line
ax_lags.axhline(
    0,
    linestyle="--",
    linewidth=1
)

# Significance bounds
ax_lags.axhline(
    significance_bound,
    linestyle="--",
    color="red",
    linewidth=1.5,
    label="95% Significance Bound"
)

ax_lags.axhline(
    -significance_bound,
    linestyle="--",
    color="red",
    linewidth=1.5
)

# Light shading
ax_lags.axhspan(
    -significance_bound,
    significance_bound,
    color="red",
    alpha=0.08
)

ax_lags.set_title(f"{ticker} Return Autocorrelation")
ax_lags.set_xlabel("Lag (Days)")
ax_lags.set_ylabel("Autocorrelation")
ax_lags.legend()
ax_lags.grid(axis="y")

st.pyplot(fig_lags)

st.subheader("Rolling Volatility")

fig3, ax3 = plt.subplots()

ax3.plot(data.index, data["Rolling Volatility"])
# Green zone: low volatility
ax3.axhspan(0, 20, alpha=0.2, color="green")

# Orange zone: moderate volatility
ax3.axhspan(20, 40, alpha=0.2, color="orange")

# Red zone: high volatility
ax3.axhspan(40, 100, alpha=0.2, color="red")

ax3.set_title(f"{ticker} 21-Day Rolling Volatility")
ax3.set_xlabel("Date")
ax3.set_ylabel("Annualised Volatility (%)")
ax3.axhline(20, linestyle="--", linewidth=1)
ax3.axhline(40, linestyle="--", linewidth=1)
ax3.grid()

st.pyplot(fig3)

st.subheader("Drawdown")

fig4, ax4 = plt.subplots()

ax4.plot(data.index, data["Drawdown"], label="Drawdown")

ax4.fill_between(
    data.index,
    data["Drawdown"],
    0,
    alpha=0.3
)

ax4.axhline(0, linestyle="--", linewidth=1)

ax4.set_title(f"{ticker} Drawdown")
ax4.set_xlabel("Date")
ax4.set_ylabel("Drawdown (%)")
ax4.axhspan(-10, 0, alpha=0.1)
ax4.axhspan(-20, -10, alpha=0.15)
ax4.axhspan(-100, -20, alpha=0.2)
ax4.grid()
ax4.legend()

st.pyplot(fig4)

st.subheader("Distribution of Daily Returns")

fig5, ax5 = plt.subplots()

# Remove missing values and convert to percentages
returns_pct = data["Returns"].dropna() * 100

# Plot histogram
ax5.hist(returns_pct, bins=50, alpha=0.7, density=True)

# Mean and standard deviation of the returns
mu = returns_pct.mean()
sigma = returns_pct.std()

# Create x-values spanning the histogram range
x = np.linspace(returns_pct.min(), returns_pct.max(), 500)

# Normal distribution values
y = norm.pdf(x, mu, sigma)

# Plot the fitted normal curve
ax5.plot(x, y, linewidth=2, label="Normal Distribution")

# Vertical line at the mean return
ax5.axvline(returns_pct.mean(), linestyle="--", linewidth=2)
# 95% VaR line
ax5.axvline(
    var_95,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"95% VaR = {var_95:.2f}%"
)

# Titles and labels
ax5.set_title(f"{ticker} Daily Return Distribution")
ax5.set_xlabel("Daily Return (%)")
ax5.set_ylabel("Density")
ax5.legend()

st.pyplot(fig5)

st.header("Monte Carlo Simulation")
simulation_days = st.sidebar.slider("Monte Carlo Forecast Days", 30, 365, 252)

num_simulations = st.sidebar.slider("Number of Simulations", 100, 2000, 500)

daily_mean_return = data["Returns"].mean()
daily_volatility = data["Returns"].std()

last_price = close.iloc[-1]

simulation_df = np.zeros((simulation_days, num_simulations))

for sim in range(num_simulations):
    prices = [last_price]

    for day in range(1, simulation_days):
        random_return = np.random.normal(daily_mean_return, daily_volatility)
        next_price = prices[-1] * (1 + random_return)
        prices.append(next_price)

    simulation_df[:, sim] = prices

final_prices = simulation_df[-1, :]

expected_final_price = final_prices.mean()
median_final_price = np.median(final_prices)

percentile_5 = np.percentile(final_prices, 5)
percentile_95 = np.percentile(final_prices, 95)

prob_profit = np.mean(final_prices > last_price) * 100

st.subheader("Monte Carlo Summary")

st.write(f"Expected Final Price: ${expected_final_price:.2f}")
st.write(f"Median Final Price: ${median_final_price:.2f}")
st.write(f"5th Percentile Price: ${percentile_5:.2f}")
st.write(f"95th Percentile Price: ${percentile_95:.2f}")
st.write(f"Probability of Profit: {prob_profit:.2f}%")

mc_table = {
    "Metric": [
        "Expected Final Price",
        "Median Final Price",
        "5th Percentile",
        "95th Percentile",
        "Probability of Profit"
    ],
    "Meaning": [
        "Average simulated ending price",
        "Middle simulated ending price",
        "Pessimistic outcome: only 5% of simulations ended below this",
        "Optimistic outcome: only 5% of simulations ended above this",
        "Percentage of simulations ending above today's price"
    ]
}

st.table(mc_table)

mean_path = simulation_df.mean(axis=1)
lower_band = np.percentile(simulation_df, 5, axis=1)
upper_band = np.percentile(simulation_df, 95, axis=1)

st.subheader("Simulated Future Price Paths")

fig6, ax6 = plt.subplots()

ax6.plot(simulation_df, alpha=0.1)
ax6.plot(mean_path, linewidth=2, label="Mean Path")

ax6.plot(lower_band, linestyle="--", linewidth=2, label="5th Percentile")
ax6.plot(upper_band, linestyle="--", linewidth=2, label="95th Percentile")

ax6.set_title(f"{ticker} Monte Carlo Simulation")
ax6.set_xlabel("Future Trading Days")
ax6.set_ylabel("Simulated Price")
ax6.grid()
ax6.legend()

st.pyplot(fig6)

st.header("Portfolio Optimisation")

portfolio_tickers = st.sidebar.text_input(
    "Portfolio Tickers",
    "AAPL,MSFT,NVDA,GOOGL"
)

num_portfolios = st.sidebar.slider(
    "Number of Random Portfolios",
    1000,
    10000,
    3000
)

portfolio_mc_days = st.sidebar.slider(
    "Portfolio MC Forecast Days",
    30,
    365,
    252
)

portfolio_mc_sims = st.sidebar.slider(
    "Portfolio MC Simulations",
    100,
    2000,
    500
)

tickers_list = [ticker.strip().upper()
                for ticker in portfolio_tickers.split(",")]

portfolio_data = yf.download(
    tickers_list,
    period=period,
    progress=False
)["Close"]

if portfolio_data.empty:
    st.error("No portfolio data found.")
    st.stop()

portfolio_returns = portfolio_data.pct_change().dropna()

annual_asset_volatility = portfolio_returns.std() * np.sqrt(252)

# Keep only assets with volatility below 100%
valid_assets = annual_asset_volatility[
    annual_asset_volatility < 1.0
].index

portfolio_returns = portfolio_returns[valid_assets]
tickers_list = list(valid_assets)

correlation_matrix = portfolio_returns.corr()
expected_returns = portfolio_returns.mean() * 252
cov_matrix_portfolio = portfolio_returns.cov() * 252
portfolio_results = []
portfolio_weights = []

def portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.sum(weights * expected_returns)

    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    return portfolio_return, portfolio_volatility


def negative_sharpe(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return, portfolio_volatility = portfolio_performance(
        weights,
        expected_returns,
        cov_matrix
    )

    sharpe = (
        (portfolio_return * 100 - risk_free_rate)
        / (portfolio_volatility * 100)
    )

    return -sharpe

num_assets = len(tickers_list)

initial_weights = np.array([1 / num_assets] * num_assets)

bounds = tuple((0, 0.30) for asset in range(num_assets))

constraints = (
    {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
)

optimised_result = minimize(
    negative_sharpe,
    initial_weights,
    args=(expected_returns, cov_matrix_portfolio, risk_free_rate),
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

optimised_weights = optimised_result.x

optimised_return, optimised_volatility = portfolio_performance(
    optimised_weights,
    expected_returns,
    cov_matrix_portfolio
)

optimised_sharpe = -optimised_result.fun

optimised_portfolio_returns = portfolio_returns.dot(optimised_weights)

optimised_portfolio_value = (
    (1 + optimised_portfolio_returns).cumprod() * 100
)

portfolio_daily_mean = optimised_portfolio_returns.mean()
portfolio_daily_volatility = optimised_portfolio_returns.std()

starting_portfolio_value = 100

portfolio_simulation = np.zeros((portfolio_mc_days, portfolio_mc_sims))

for sim in range(portfolio_mc_sims):
    values = [starting_portfolio_value]

    for day in range(1, portfolio_mc_days):
        random_return = np.random.normal(
            portfolio_daily_mean,
            portfolio_daily_volatility
        )

        next_value = values[-1] * (1 + random_return)
        values.append(next_value)

    portfolio_simulation[:, sim] = values

portfolio_final_values = portfolio_simulation[-1, :]

portfolio_expected_final = portfolio_final_values.mean()
portfolio_median_final = np.median(portfolio_final_values)
portfolio_5th = np.percentile(portfolio_final_values, 5)
portfolio_95th = np.percentile(portfolio_final_values, 95)
portfolio_prob_profit = np.mean(
    portfolio_final_values > starting_portfolio_value
) * 100

benchmark_portfolio = (
    benchmark_close / benchmark_close.iloc[0] * 100
)

np.random.seed(42)

for i in range(num_portfolios):
    weights = np.random.random(len(tickers_list))
    weights = weights / np.sum(weights)

    portfolio_return = np.sum(weights * expected_returns)

    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix_portfolio, weights))
    )

    portfolio_sharpe = (
        (portfolio_return * 100 - risk_free_rate)
        / (portfolio_volatility * 100)
    )

    portfolio_results.append([
        portfolio_return,
        portfolio_volatility,
        portfolio_sharpe
    ])

    portfolio_weights.append(weights)
portfolio_results = np.array(portfolio_results)
portfolio_weights = np.array(portfolio_weights)

max_sharpe_index = np.argmax(portfolio_results[:, 2])

max_sharpe_return = portfolio_results[max_sharpe_index, 0]
max_sharpe_volatility = portfolio_results[max_sharpe_index, 1]
max_sharpe_ratio = portfolio_results[max_sharpe_index, 2]
max_sharpe_weights = portfolio_weights[max_sharpe_index]
min_volatility_index = np.argmin(portfolio_results[:, 1])

min_volatility_return = portfolio_results[min_volatility_index, 0]
min_volatility_volatility = portfolio_results[min_volatility_index, 1]
min_volatility_sharpe = portfolio_results[min_volatility_index, 2]
min_volatility_weights = portfolio_weights[min_volatility_index]

st.subheader("Portfolio Returns")
st.write(portfolio_returns.tail())
st.subheader("Correlation Matrix")
st.write(correlation_matrix)

fig_corr, ax_corr = plt.subplots(figsize=(8, 6))

im = ax_corr.imshow(correlation_matrix, vmin=-1, vmax=1)

ax_corr.set_xticks(range(len(correlation_matrix.columns)))
ax_corr.set_yticks(range(len(correlation_matrix.columns)))

ax_corr.set_xticklabels(correlation_matrix.columns, rotation=45)
ax_corr.set_yticklabels(correlation_matrix.columns)

fig_corr.colorbar(im)

ax_corr.set_title("Portfolio Correlation Heatmap")

st.pyplot(fig_corr)

st.subheader("Random Portfolio Simulation")

fig_pf, ax_pf = plt.subplots()

scatter = ax_pf.scatter(
    portfolio_results[:, 1] * 100,
    portfolio_results[:, 0] * 100,
    c=portfolio_results[:, 2],
    cmap="viridis",
    alpha=0.6
)

ax_pf.set_title("Random Portfolios: Risk vs Return")
ax_pf.set_xlabel("Annualised Volatility (%)")
ax_pf.set_ylabel("Annualised Return (%)")

fig_pf.colorbar(scatter, label="Sharpe Ratio")

ax_pf.scatter(
    max_sharpe_volatility * 100,
    max_sharpe_return * 100,
    marker="*",
    s=300,
    label="Max Sharpe Portfolio"
)
ax_pf.scatter(
    min_volatility_volatility * 100,
    min_volatility_return * 100,
    marker="X",
    s=200,
    label="Minimum Volatility Portfolio"
)
ax_pf.scatter(
    optimised_volatility * 100,
    optimised_return * 100,
    marker="D",
    s=250,
    label="Optimised Max Sharpe"
)

ax_pf.legend()

ax_pf.grid()

st.pyplot(fig_pf)

st.subheader("Maximum Sharpe Portfolio Weights")


for ticker_name, weight in zip(tickers_list, max_sharpe_weights):
    st.write(f"{ticker_name}: {weight * 100:.2f}%")

st.write(f"Expected Return: {max_sharpe_return * 100:.2f}%")
st.write(f"Expected Volatility: {max_sharpe_volatility * 100:.2f}%")
st.write(f"Sharpe Ratio: {max_sharpe_ratio:.2f}")

st.subheader("Minimum Volatility Portfolio Weights")

for ticker_name, weight in zip(tickers_list, min_volatility_weights):
    st.write(f"{ticker_name}: {weight * 100:.2f}%")

st.write(f"Expected Return: {min_volatility_return * 100:.2f}%")
st.write(f"Expected Volatility: {min_volatility_volatility * 100:.2f}%")
st.write(f"Sharpe Ratio: {min_volatility_sharpe:.2f}")

st.subheader("SciPy Optimised Max Sharpe Portfolio")

for ticker_name, weight in zip(tickers_list, optimised_weights):
    st.write(f"{ticker_name}: {weight * 100:.2f}%")

st.write(f"Expected Return: {optimised_return * 100:.2f}%")
st.write(f"Expected Volatility: {optimised_volatility * 100:.2f}%")
st.write(f"Sharpe Ratio: {optimised_sharpe:.2f}")

st.subheader("Optimised Portfolio Allocation")

fig_alloc, ax_alloc = plt.subplots()

ax_alloc.bar(tickers_list, optimised_weights * 100)

ax_alloc.set_title("SciPy Optimised Portfolio Weights")
ax_alloc.set_xlabel("Asset")
ax_alloc.set_ylabel("Weight (%)")
ax_alloc.grid(axis="y")

st.pyplot(fig_alloc)

st.subheader("Optimised Portfolio Allocation (Pie Chart)")

fig_pie, ax_pie = plt.subplots()

ax_pie.pie(
    optimised_weights,
    labels=tickers_list,
    autopct="%1.1f%%",
    startangle=90
)

ax_pie.set_title("SciPy Optimised Portfolio Allocation")

st.pyplot(fig_pie)

st.subheader("Optimised Portfolio Backtest")

fig_backtest, ax_backtest = plt.subplots()

ax_backtest.plot(
    optimised_portfolio_value.index,
    optimised_portfolio_value,
    label="Optimised Portfolio"
)

ax_backtest.plot(
    benchmark_portfolio.index,
    benchmark_portfolio,
    label=benchmark
)

ax_backtest.set_title("Optimised Portfolio vs Benchmark")
ax_backtest.set_xlabel("Date")
ax_backtest.set_ylabel("Growth of 100")
ax_backtest.legend()
ax_backtest.grid()

st.pyplot(fig_backtest)

st.subheader("Optimised Portfolio Monte Carlo Simulation")

st.write(f"Expected Final Portfolio Value: {portfolio_expected_final:.2f}")
st.write(f"Median Final Portfolio Value: {portfolio_median_final:.2f}")
st.write(f"5th Percentile Portfolio Value: {portfolio_5th:.2f}")
st.write(f"95th Percentile Portfolio Value: {portfolio_95th:.2f}")
st.write(f"Probability of Profit: {portfolio_prob_profit:.2f}%")

fig_port_mc, ax_port_mc = plt.subplots()

ax_port_mc.plot(portfolio_simulation, alpha=0.1)

mean_portfolio_path = portfolio_simulation.mean(axis=1)
lower_portfolio_band = np.percentile(portfolio_simulation, 5, axis=1)
upper_portfolio_band = np.percentile(portfolio_simulation, 95, axis=1)

ax_port_mc.plot(mean_portfolio_path, linewidth=2, label="Mean Path")
ax_port_mc.plot(lower_portfolio_band, linestyle="--", linewidth=2, label="5th Percentile")
ax_port_mc.plot(upper_portfolio_band, linestyle="--", linewidth=2, label="95th Percentile")

ax_port_mc.set_title("Optimised Portfolio Monte Carlo Simulation")
ax_port_mc.set_xlabel("Future Trading Days")
ax_port_mc.set_ylabel("Portfolio Value")
ax_port_mc.legend()
ax_port_mc.grid()

st.pyplot(fig_port_mc)

st.header("News and Macro Context")

st.subheader(f"Latest News for {ticker}")

stock_news = yf.Ticker(ticker).news

if len(stock_news) == 0:
    st.write("No recent news found.")
else:
    for article in stock_news[:5]:
        title = article.get("title", "No title")
        publisher = article.get("publisher", "Unknown publisher")
        link = article.get("link", "")

        st.write(f"**{title}**")
        st.write(f"Source: {publisher}")

        if link:
            st.link_button("Read article", link)

st.subheader("Portfolio Stock News")

for stock in tickers_list:
    st.write(f"### {stock}")

    try:
        news_items = yf.Ticker(stock).news

        if len(news_items) == 0:
            st.write("No recent news found.")

        for article in news_items[:3]:
            title = article.get("title", "No title")
            publisher = article.get("publisher", "Unknown publisher")
            link = article.get("link", "")

            st.write(f"**{title}**")
            st.write(f"Source: {publisher}")

            if link:
                st.link_button("Read article", link)

    except Exception as e:
        st.write(f"Could not load news for {stock}")

st.subheader("Macro Market Indicators")

macro_tickers = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "VIX Fear Index": "^VIX",
    "US 10-Year Treasury Yield": "^TNX",
    "US Dollar Index": "DX-Y.NYB"
}

macro_data = {}

for name, macro_ticker in macro_tickers.items():
    try:
        macro_prices = yf.download(
            macro_ticker,
            period="1mo",
            progress=False
        )["Close"].squeeze()

        if not macro_prices.empty:
            latest_value = macro_prices.iloc[-1]
            one_month_change = (
                latest_value / macro_prices.iloc[0] - 1
            ) * 100

            macro_data[name] = {
                "Latest Value": latest_value,
                "1-Month Change (%)": one_month_change
            }

    except Exception:
        st.write(f"Could not load {name}")

macro_df = pd.DataFrame(macro_data).T
st.dataframe(macro_df)

st.subheader("Macro Interpretation Guide")

macro_table = {
    "Indicator": [
        "S&P 500",
        "Nasdaq",
        "VIX",
        "US 10-Year Treasury Yield",
        "US Dollar Index"
    ],
    "Why It Matters": [
        "Broad US equity market direction",
        "Growth and technology stock sentiment",
        "Measures market fear and expected volatility",
        "Higher yields can pressure stock valuations",
        "Stronger dollar can affect global companies and commodities"
    ]
}

st.table(macro_table)