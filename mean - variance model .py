'''
    install these Libraries
    pyPortfolioOpt
    yfinance
    pandas
    numpy 
    matplotlib
    datetime

'''
# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns

# Override pandas_datareader's backend with yfinance
yf.pdr_override()


# Load portfolio price data
portfolio = pd.read_csv("portfolio.csv", parse_dates=True, index_col="Date")

# Visualize stock prices
portfolio[portfolio.index >= "2021-04-01"].plot(figsize=(15, 10))
plt.title("Stock Prices Over Time")
plt.show()

# Daily returns
returns = portfolio.pct_change().dropna()

# Sample covariance matrix (annualized)
sample_cov = returns.cov() * 252

# Ledoit-Wolf shrinkage (manual estimate)
def ledoit_wolf_shrinkage(cov_matrix, shrink_target=None, shrinkage=0.1):
    """
    Basic Ledoit-Wolf shrinkage estimator
    """
    if shrink_target is None:
        shrink_target = np.identity(len(cov_matrix)) * np.mean(np.diag(cov_matrix))
    return shrinkage * shrink_target + (1 - shrinkage) * cov_matrix

S = ledoit_wolf_shrinkage(sample_cov, shrinkage=0.1)

# Plot covariance heatmap
sns.heatmap(S, annot=False, cmap="coolwarm", xticklabels=portfolio.columns, yticklabels=portfolio.columns)
plt.title("Ledoit-Wolf Covariance Matrix (Approx)")
plt.show()

# Estimating expected returns (CAPM-style using historical mean return)
rf = 0.015  # Approx risk-free rate (e.g., 1.5%)
mu = returns.mean() * 252  # Annualized return estimate

# Bar plot of expected returns
mu.plot.barh(figsize=(10, 6))
plt.title("Expected Annual Returns (Historical Estimate)")
plt.xlabel("Expected Return")
plt.show()

# Monte Carlo Portfolio Simulation
n_assets = len(portfolio.columns)
n_samples = 10000

# Random weights using Dirichlet distribution
w = np.random.dirichlet(np.ones(n_assets), n_samples)

# Portfolio returns and volatilities
rets = w @ mu.values
stds = np.sqrt(np.einsum('ij,jk,ik->i', w, S, w))

sharpes = (rets - rf) / stds

# Max Sharpe portfolio
max_sharpe_idx = np.argmax(sharpes)
max_sharpe_weights = w[max_sharpe_idx]
ret_tangent = rets[max_sharpe_idx]
std_tangent = stds[max_sharpe_idx]

# Display weights of max Sharpe portfolio
print("\nMax Sharpe Portfolio Weights:")
for asset, weight in zip(portfolio.columns, max_sharpe_weights):
    if weight > 0.01:
        print(f"{asset}: {weight:.2%}")

print(f"\nExpected Return: {ret_tangent:.2%}")
print(f"Volatility: {std_tangent:.2%}")
print(f"Sharpe Ratio: {sharpes[max_sharpe_idx]:.2f}")

# Plot of Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 10))
sc = ax.scatter(stds, rets, c=sharpes, cmap="viridis_r", marker=".")
ax.scatter(std_tangent, ret_tangent, c='red', marker='X', s=200, label='Max Sharpe')

# Formatting
plt.colorbar(sc, label="Sharpe Ratio")
ax.set_xlabel("Volatility")
ax.set_ylabel("Expected Return")
ax.set_title("Efficient Frontier with Monte Carlo Portfolios")
ax.legend()
plt.tight_layout()
plt.show()
