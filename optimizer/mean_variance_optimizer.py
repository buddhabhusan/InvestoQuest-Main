# mean_variance_optimizer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def ledoit_wolf_shrinkage(cov_matrix, shrink_target=None, shrinkage=0.1):
    if shrink_target is None:
        shrink_target = np.identity(len(cov_matrix)) * np.mean(np.diag(cov_matrix))
    return shrinkage * shrink_target + (1 - shrinkage) * cov_matrix

def analyze_portfolio(df, risk_free_rate=0.015, shrinkage=0.0, max_volatility=1.0):
    # Step 1: Calculate log returns (more robust)
    returns = np.log(df / df.shift(1)).dropna()

    # Step 2: Estimate expected returns and covariance
    mu = returns.mean() * 252
    mu = mu.clip(lower=-1, upper=1)  # Avoid unrealistic returns

    cov_matrix = returns.cov() * 252
    if shrinkage > 0:
        cov_matrix = ledoit_wolf_shrinkage(cov_matrix, shrinkage=shrinkage)

    # Step 3: Monte Carlo portfolio simulation
    n_assets = len(df.columns)
    n_samples = 50000
    weights = np.random.dirichlet(np.ones(n_assets), size=n_samples)

    # Expected portfolio returns
    portfolio_returns = weights @ mu.values

    # Portfolio variances (vectorized)
    portfolio_variances = np.einsum('ij,jk,ik->i', weights, cov_matrix.values, weights)
    portfolio_stds = np.sqrt(portfolio_variances)

    # Step 4: Filter out extreme volatility
    valid = portfolio_stds < max_volatility
    if not np.any(valid):
        return "<p>No valid portfolios found under volatility cap.</p>"

    weights = weights[valid]
    portfolio_returns = portfolio_returns[valid]
    portfolio_stds = portfolio_stds[valid]
    sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_stds

    # Step 5: Max Sharpe portfolio
    max_idx = np.argmax(sharpe_ratios)
    max_weights = weights[max_idx]
    max_return = portfolio_returns[max_idx]
    max_vol = portfolio_stds[max_idx]
    max_sharpe = sharpe_ratios[max_idx]

    # Step 6: Display top weights
    weights_table = "<ul>"
    for asset, weight in zip(df.columns, max_weights):
        if weight > 0.01:
            weights_table += f"<li>{asset}: {weight:.2%}</li>"
    weights_table += "</ul>"

    # Step 7: Plot efficient frontier
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(portfolio_stds, portfolio_returns, c=sharpe_ratios, cmap='viridis', s=3)
    ax.scatter(max_vol, max_return, c='red', marker='X', s=100, label='Max Sharpe Portfolio')
    plt.colorbar(scatter, label="Sharpe Ratio")
    ax.set_title("Efficient Frontier (Monte Carlo Simulation)")
    ax.set_xlabel("Volatility (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    # Step 8: Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    img_data = buffer.getvalue()
    buffer.close()
    img_base64 = base64.b64encode(img_data).decode('utf-8')
    image_html = f"<img src='data:image/png;base64,{img_base64}' />"

    # Step 9: Assemble HTML output
    html_output = f"""
        <h3>Max Sharpe Portfolio</h3>
        {weights_table}
        <p><strong>Expected Return:</strong> {max_return:.2%}</p>
        <p><strong>Volatility:</strong> {max_vol:.2%}</p>
        <p><strong>Sharpe Ratio:</strong> {max_sharpe:.2f}</p>
        {image_html}
    """

    return html_output
