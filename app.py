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

screener_tickers = st.sidebar.text_area(
    "Stock Screener Universe",
    "AAPL,MSFT,NVDA,GOOGL,AMZN,META,TSLA,JPM,JNJ,KO"
)

run_screener = st.sidebar.checkbox(
    "Run Stock Screener",
    value=False
)

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

data = data.sort_index()

benchmark_data = yf.download(benchmark, period=period, progress=False)

if benchmark_data.empty:
    st.error("No benchmark data found. Check the benchmark ticker.")
    st.stop()

# converts a one-column data frame to a series
close = data["Close"].squeeze()
benchmark_close = benchmark_data["Close"].squeeze()

data["Short MA"] = close.rolling(short_ma).mean()
data["Long MA"] = close.rolling(long_ma).mean()

# Bollinger Bands
data["BB Middle"] = close.rolling(20).mean()
rolling_std = close.rolling(20).std()

data["BB Upper"] = data["BB Middle"] + 2 * rolling_std
data["BB Lower"] = data["BB Middle"] - 2 * rolling_std

# MACD
ema_12 = close.ewm(span=12, adjust=False).mean()
ema_26 = close.ewm(span=26, adjust=False).mean()

data["MACD"] = ema_12 - ema_26
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
data["MACD Histogram"] = data["MACD"] - data["Signal"]

data["Returns"] = close.pct_change()



# RSI (14-day)
delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

rs = average_gain / average_loss

data["RSI"] = 100 - (100 / (1 + rs))

benchmark_returns = benchmark_close.pct_change()
# Normalise both series so they start at 100
normalised_stock = close / close.iloc[0] * 100
normalised_benchmark = benchmark_close / benchmark_close.iloc[0] * 100

# Align stock and benchmark returns on common dates
stock_returns = data["Returns"].squeeze()
benchmark_returns = benchmark_returns.squeeze()

combined_returns = pd.DataFrame({
    "Stock": stock_returns,
    "Benchmark": benchmark_returns
}).dropna()

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

stock_info = yf.Ticker(ticker).info

data["Rolling Volatility"] = data["Returns"].rolling(21).std() * np.sqrt(252) * 100

# Rolling Sharpe Ratio (63 trading days ≈ 3 months)
rolling_return = data["Returns"].rolling(63).mean() * 252
rolling_volatility = data["Returns"].rolling(63).std() * np.sqrt(252)

data["Rolling Sharpe"] = (
    (rolling_return * 100 - risk_free_rate)
    / (rolling_volatility * 100)
)

data["Running Max"] = close.cummax()
data["Drawdown"] = (close / data["Running Max"] - 1) * 100

# uses root 252 as theres around 252 trading days a year 
latest_price = close.iloc[-1]
total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
annual_volatility = data["Returns"].std() * np.sqrt(252) * 100
average_daily_return = data["Returns"].mean()
annual_return = average_daily_return * 252 * 100
alpha = annual_return - beta * benchmark_annual_return
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
downside_returns = data["Returns"][data["Returns"] < 0]
downside_volatility = downside_returns.std() * np.sqrt(252) * 100
sortino_ratio = (annual_return - risk_free_rate) / downside_volatility

max_drawdown = data["Drawdown"].min()
# 5th percentile = 95% one-day VaR
var_95 = np.percentile(data["Returns"].dropna() * 100, 5)
returns_clean = data["Returns"].dropna() * 100
expected_shortfall = returns_clean[returns_clean <= var_95].mean()
# -----------------------------
# Enhanced Investment Decision Support
# -----------------------------

decision_score = 0
decision_reasons = []
risk_warnings = []

# Return
if annual_return > 15:
    decision_score += 2
    decision_reasons.append("Annualised return is strong.")
elif annual_return > risk_free_rate:
    decision_score += 1
    decision_reasons.append("Annualised return is above the risk-free rate.")
else:
    decision_score -= 2
    risk_warnings.append("Annualised return is below the risk-free rate.")

# Sharpe Ratio
if sharpe_ratio > 1.5:
    decision_score += 2
    decision_reasons.append("Sharpe ratio is strong.")
elif sharpe_ratio > 0.5:
    decision_score += 1
    decision_reasons.append("Sharpe ratio is acceptable.")
else:
    decision_score -= 1
    risk_warnings.append("Sharpe ratio is weak.")

# Sortino Ratio
if sortino_ratio > 2:
    decision_score += 2
    decision_reasons.append("Sortino ratio suggests strong downside-adjusted performance.")
elif sortino_ratio > 1:
    decision_score += 1
    decision_reasons.append("Sortino ratio is reasonable.")
else:
    decision_score -= 1
    risk_warnings.append("Sortino ratio is weak.")

# Drawdown
if max_drawdown > -15:
    decision_score += 2
    decision_reasons.append("Maximum drawdown is relatively controlled.")
elif max_drawdown > -30:
    decision_score += 0
    risk_warnings.append("Maximum drawdown is moderate.")
else:
    decision_score -= 2
    risk_warnings.append("Maximum drawdown is severe.")

# VaR
if var_95 > -2:
    decision_score += 2
    decision_reasons.append("Daily VaR is relatively low.")
elif var_95 > -4:
    decision_score += 0
    risk_warnings.append("Daily VaR is moderate.")
else:
    decision_score -= 2
    risk_warnings.append("Daily VaR is high.")

# Beta
if beta < 0.8:
    decision_score += 1
    decision_reasons.append("Beta is defensive versus the benchmark.")
elif beta <= 1.3:
    decision_score += 1
    decision_reasons.append("Beta is reasonably close to market sensitivity.")
else:
    decision_score -= 1
    risk_warnings.append("Beta is high, meaning the stock is more market-sensitive.")

# Alpha
if alpha > 5:
    decision_score += 2
    decision_reasons.append("Alpha is positive, suggesting outperformance versus benchmark exposure.")
elif alpha > 0:
    decision_score += 1
    decision_reasons.append("Alpha is slightly positive.")
else:
    decision_score -= 1
    risk_warnings.append("Alpha is negative versus benchmark exposure.")

# RSI
latest_rsi = data["RSI"].dropna().iloc[-1]

if latest_rsi > 75:
    decision_score -= 1
    risk_warnings.append("RSI is very high, suggesting the stock may be overbought.")
elif latest_rsi < 30:
    decision_score += 1
    decision_reasons.append("RSI is low, suggesting the stock may be oversold.")
else:
    decision_score += 1
    decision_reasons.append("RSI is not showing extreme overbought conditions.")

if decision_score >= 9:
    decision_signal = "Potential Buy"
elif decision_score >= 5:
    decision_signal = "Watch / Hold"
elif decision_score >= 1:
    decision_signal = "Wait"
else:
    decision_signal = "High Risk / Avoid"

st.subheader("Company Information")

company_name = stock_info.get("longName", ticker)
sector = stock_info.get("sector", "N/A")
industry = stock_info.get("industry", "N/A")
market_cap = stock_info.get("marketCap", "N/A")
pe_ratio = stock_info.get("trailingPE", "N/A")
dividend_yield = stock_info.get("dividendYield", "N/A")

st.write(f"**Company:** {company_name}")
st.write(f"**Sector:** {sector}")
st.write(f"**Industry:** {industry}")

if market_cap != "N/A":
    st.write(f"**Market Cap:** ${market_cap:,.0f}")
else:
    st.write("**Market Cap:** N/A")

st.write(f"**P/E Ratio:** {pe_ratio}")

if dividend_yield != "N/A":
    st.write(f"**Dividend Yield:** {dividend_yield * 100:.2f}%")
else:
    st.write("**Dividend Yield:** N/A")

st.subheader("Earnings and Dividend Calendar")

try:
    stock_obj = yf.Ticker(ticker)

    st.write("### Upcoming / Recent Earnings")

    try:
        earnings_dates = stock_obj.get_earnings_dates(limit=4)

        if earnings_dates is not None and not earnings_dates.empty:
            st.write(earnings_dates)
        else:
            st.write("No earnings dates found for this ticker.")
    except Exception:
        st.write("Earnings data is unavailable for this ticker.")

    st.write("### Recent Dividends")

    try:
        dividends = stock_obj.dividends

        if dividends is not None and not dividends.empty:
            st.write(dividends.tail(5))
        else:
            st.write("No recent dividend data found. This company may not currently pay dividends.")
    except Exception:
        st.write("Dividend data is unavailable for this ticker.")

except Exception:
    st.write("Could not load earnings or dividend data.")

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
st.write(f"Sortino Ratio: {sortino_ratio:.2f}")
st.write("""
Sortino Ratio is similar to Sharpe Ratio, but it only penalises downside risk.
A higher Sortino Ratio suggests the stock has delivered better returns relative to harmful volatility.
""")
st.write(f"Beta vs {benchmark}: {beta:.2f}")
st.write(f"Alpha vs {benchmark}: {alpha:.2f}%")

st.write(f"Maximum Drawdown: {max_drawdown:.2f}%")
st.write(f"95% 1-Day VaR: {var_95:.2f}%")
st.write(f"Expected Shortfall: {expected_shortfall:.2f}%")

st.subheader("Investment Decision Support")

st.write(f"### Dashboard Signal: {decision_signal}")
st.write(f"Decision Score: {decision_score}")

st.write("**Positive Factors:**")
if len(decision_reasons) > 0:
    for reason in decision_reasons:
        st.write(f"- {reason}")
else:
    st.write("- No major positive factors detected.")

st.write("**Risk Warnings:**")
if len(risk_warnings) > 0:
    for warning in risk_warnings:
        st.write(f"- {warning}")
else:
    st.write("- No major risk warnings detected.")

st.warning(
    "This is not financial advice. This signal is based only on the dashboard's quantitative metrics and should be combined with further research."
)

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

st.subheader("Relative Strength Index (RSI)")

st.write("""
RSI measures recent momentum on a scale from 0 to 100.

- Above 70 may suggest the stock is overbought.
- Below 30 may suggest the stock is oversold.
- Around 50 suggests balanced momentum.
""")

fig_rsi, ax_rsi = plt.subplots()

ax_rsi.plot(data.index, data["RSI"], label="RSI")

ax_rsi.axhline(70, linestyle="--", color="red", label="Overbought (70)")
ax_rsi.axhline(30, linestyle="--", color="green", label="Oversold (30)")
ax_rsi.axhline(50, linestyle="--", linewidth=1)

ax_rsi.set_title(f"{ticker} RSI (14-Day)")
ax_rsi.set_xlabel("Date")
ax_rsi.set_ylabel("RSI")
ax_rsi.set_ylim(0, 100)
ax_rsi.legend()
ax_rsi.grid()

st.pyplot(fig_rsi)

st.subheader("Bollinger Bands")

st.write("""
Bollinger Bands show where the stock price sits relative to its recent average range.

- Price near the upper band suggests strong recent momentum.
- Price near the lower band suggests weakness or possible oversold conditions.
- Wider bands mean higher volatility.
- Narrower bands mean lower volatility.
""")

fig_bb, ax_bb = plt.subplots()

ax_bb.plot(data.index, close, label="Close")
ax_bb.plot(data.index, data["BB Middle"], label="20-Day MA")
ax_bb.plot(data.index, data["BB Upper"], linestyle="--", label="Upper Band")
ax_bb.plot(data.index, data["BB Lower"], linestyle="--", label="Lower Band")

ax_bb.fill_between(
    data.index,
    data["BB Lower"],
    data["BB Upper"],
    alpha=0.1
)

ax_bb.set_title(f"{ticker} Bollinger Bands")
ax_bb.set_xlabel("Date")
ax_bb.set_ylabel("Price")
ax_bb.legend()
ax_bb.grid()

st.pyplot(fig_bb)

st.subheader("MACD")

st.write("""
MACD compares short-term and long-term momentum.

- MACD above the signal line suggests bullish momentum.
- MACD below the signal line suggests bearish momentum.
- A rising histogram suggests momentum is strengthening.
- A falling histogram suggests momentum is weakening.
""")

fig_macd, ax_macd = plt.subplots()

ax_macd.plot(data.index, data["MACD"], label="MACD")
ax_macd.plot(data.index, data["Signal"], label="Signal Line")

hist_colors = [
    "green" if x >= 0 else "red"
    for x in data["MACD Histogram"]
]

ax_macd.bar(
    data.index,
    data["MACD Histogram"],
    color=hist_colors,
    alpha=0.5
)

ax_macd.axhline(0, linestyle="--", linewidth=1)

ax_macd.set_title(f"{ticker} MACD")
ax_macd.set_xlabel("Date")
ax_macd.set_ylabel("MACD")
ax_macd.legend()
ax_macd.grid()

st.pyplot(fig_macd)

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

daily_returns_pct = data["Returns"] * 100

colors=[
    "green" if r>=0 else "red"
    for r in daily_returns_pct
]

ax2.bar(
    data.index,
    daily_returns_pct,
    width=1.0,
    color=colors
)
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

st.subheader("Rolling Sharpe Ratio")

st.write("""
The Rolling Sharpe Ratio measures how attractive the stock's returns have been
relative to risk over the last 63 trading days (about 3 months).

- Above 2: Excellent risk-adjusted performance
- Above 1: Good
- Around 0: Limited reward for risk
- Negative: Poor risk-adjusted performance
""")

fig_sharpe, ax_sharpe = plt.subplots()

ax_sharpe.plot(
    data.index,
    data["Rolling Sharpe"],
    linewidth=2
)

# Reference lines
ax_sharpe.axhline(2, linestyle="--", color="green", label="Excellent (2)")
ax_sharpe.axhline(1, linestyle="--", color="blue", label="Good (1)")
ax_sharpe.axhline(0, linestyle="--", color="black", label="Neutral (0)")

# Background zones
ax_sharpe.axhspan(2, 10, color="green", alpha=0.1)
ax_sharpe.axhspan(1, 2, color="blue", alpha=0.08)
ax_sharpe.axhspan(0, 1, color="yellow", alpha=0.08)
ax_sharpe.axhspan(-10, 0, color="red", alpha=0.08)

ax_sharpe.set_title(f"{ticker} Rolling Sharpe Ratio")
ax_sharpe.set_xlabel("Date")
ax_sharpe.set_ylabel("Sharpe Ratio")
ax_sharpe.legend()
ax_sharpe.grid()

st.pyplot(fig_sharpe)

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

initial_investment = st.sidebar.number_input(
    "Initial Investment (£)",
    min_value=100.0,
    value=1000.0,
    step=100.0
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
expected_return_pct = portfolio_expected_final - 100
median_return_pct = portfolio_median_final - 100
pessimistic_return_pct = portfolio_5th - 100
optimistic_return_pct = portfolio_95th - 100

expected_final_money = initial_investment * portfolio_expected_final / 100
median_final_money = initial_investment * portfolio_median_final / 100
pessimistic_final_money = initial_investment * portfolio_5th / 100
optimistic_final_money = initial_investment * portfolio_95th / 100
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

st.write(f"Expected Final Value: £{expected_final_money:,.2f} ({expected_return_pct:.2f}%)")
st.write(f"Median Final Value: £{median_final_money:,.2f} ({median_return_pct:.2f}%)")
st.write(f"5th Percentile Value: £{pessimistic_final_money:,.2f} ({pessimistic_return_pct:.2f}%)")
st.write(f"95th Percentile Value: £{optimistic_final_money:,.2f} ({optimistic_return_pct:.2f}%)")
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

try:
    stock_news = yf.Ticker(ticker).news

    if len(stock_news) == 0:
        st.write("No recent news found.")
    else:
        for article in stock_news[:5]:

            content = article.get("content", {})

            title = content.get("title", "No title")
            publisher = content.get("provider", {}).get("displayName", "Unknown publisher")
            link = content.get("canonicalUrl", {}).get("url", "")

            st.write(f"**{title}**")
            st.write(f"Source: {publisher}")

            if link:
                st.link_button("Read article", link)

except Exception as e:
    st.write(f"Could not load news for {ticker}.")

st.subheader("Portfolio Stock News")

for stock in tickers_list:
    st.write(f"### {stock}")

    try:
        news_items = yf.Ticker(stock).news

        if len(news_items) == 0:
            st.write("No recent news found.")

        for article in news_items[:3]:

            content = article.get("content", {})

            title = content.get("title", "No title")
            publisher = content.get("provider", {}).get("displayName", "Unknown publisher")
            link = content.get("canonicalUrl", {}).get("url", "")

            st.write(f"**{title}**")
            st.write(f"Source: {publisher}")

            if link:
                st.link_button("Read article", link)

    except Exception:
        st.write(f"Could not load news for {stock}.")

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

if run_screener:
    st.header("Stock Screener")

    screener_list = [
        stock.strip().upper()
        for stock in screener_tickers.split(",")
        if stock.strip() != ""
    ]

    screener_results = []

    for stock in screener_list:
        try:
            temp_data = yf.download(
                stock,
                period=period,
                progress=False
            )

            if temp_data.empty:
                continue

            temp_close = temp_data["Close"].squeeze()
            temp_returns = temp_close.pct_change().dropna()

            temp_annual_return = temp_returns.mean() * 252 * 100
            temp_annual_volatility = (
                temp_returns.std() * np.sqrt(252) * 100
            )

            if temp_annual_volatility == 0:
                continue

            temp_sharpe = (
                temp_annual_return - risk_free_rate
            ) / temp_annual_volatility

            temp_running_max = temp_close.cummax()
            temp_drawdown = (
                temp_close / temp_running_max - 1
            ) * 100

            temp_max_drawdown = temp_drawdown.min()

            temp_var_95 = np.percentile(
                temp_returns * 100,
                5
            )

            # Scoring
            temp_score = 0

            if temp_annual_return > 15:
                temp_score += 2
            elif temp_annual_return > risk_free_rate:
                temp_score += 1
            else:
                temp_score -= 1

            if temp_sharpe > 1:
                temp_score += 2
            elif temp_sharpe > 0.5:
                temp_score += 1
            else:
                temp_score -= 1

            if temp_max_drawdown > -20:
                temp_score += 1
            else:
                temp_score -= 1

            if temp_var_95 > -3:
                temp_score += 1
            else:
                temp_score -= 1

            screener_results.append({
                "Ticker": stock,
                "Annual Return (%)": temp_annual_return,
                "Annual Volatility (%)": temp_annual_volatility,
                "Sharpe Ratio": temp_sharpe,
                "Max Drawdown (%)": temp_max_drawdown,
                "95% VaR (%)": temp_var_95,
                "Score": temp_score
            })

        except Exception:
            pass

    if len(screener_results) == 0:
        st.write("No valid screener results found.")
    else:
        screener_df = pd.DataFrame(screener_results)

        screener_df = screener_df.sort_values(
            by="Score",
            ascending=False
        )

        st.subheader("Ranked Stocks")
        st.dataframe(screener_df)

        st.subheader("Top Ranked Stock")
        st.write(screener_df.iloc[0])