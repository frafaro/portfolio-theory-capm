# Portfolio Theory & CAPM — Efficient Frontier via Monte Carlo Simulation

Commented Python notebook (**`CAPM-commented.ipynb`**) that builds the efficient frontier for a basket of stocks by simulating a large number of random portfolios, identifies the **Maximum Sharpe Ratio portfolio** and the **Minimum Volatility portfolio**, and draws the **Capital Market Line (CML)**.

The sections below follow the notebook chronologically.

---

## 1. Data download and dataset construction

Daily prices for five Austrian stocks (Vienna Stock Exchange) are downloaded with `yfinance` for the year 2021:

```python
tickers = ['EBS.VI', 'OMV.VI', 'UQA.VI', 'VER.VI', 'VOE.VI']
prices = yf.download(tickers, start="2021-01-01", end="2022-01-01",
                     interval="1d", group_by='ticker',
                     auto_adjust=True, threads=True)
```

- `auto_adjust=True` replaces the raw close with the **adjusted close** (dividends and splits accounted for).
- `threads=True` downloads all tickers in parallel.

A clean DataFrame `df` is then built where **each column is one stock** and each row is one trading day's closing price.

---

## 2. Function definitions

### 2.1 `calc_portfolio_perf(weights, mean_returns, cov, rf)`

Given a vector of portfolio weights, this function computes the three key performance metrics of the portfolio. Everything is **annualized assuming 252 trading days per year**.

**Mean (annualized) portfolio return** — the weighted average of the assets' mean daily returns, scaled to one year:

$$R_p = \left( \sum_{i=1}^{N} w_i \, \bar{r}_i \right) \times 252$$

where $w_i$ is the weight of asset $i$ and $\bar{r}_i$ its mean daily return. In code: `np.sum(mean_returns * weights) * 252`.

**Portfolio volatility (annualized standard deviation)** — obtained from the covariance matrix $\Sigma$ of daily returns through the quadratic form:

$$\sigma_p = \sqrt{\mathbf{w}^{T} \, \Sigma \, \mathbf{w}} \times \sqrt{252}$$

In code: `np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)`, where `np.dot` performs the matrix multiplication. This is the step that captures **diversification**: the off-diagonal covariances make portfolio risk lower than the weighted average of individual risks.

**Sharpe ratio** — excess return per unit of risk, with $r_f$ the risk-free rate:

$$SR = \frac{R_p - r_f}{\sigma_p}$$

The function returns the tuple `(portfolio_return, portfolio_std, sharpe_ratio)`.

### 2.2 `simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)`

This function generates *n* portfolios with **random weights** to explore all possible combinations of the assets — a Monte Carlo approximation of the feasible set (and therefore of the efficient frontier, its upper-left edge).

How the randomness works, for each simulated portfolio:

1. `weights = np.random.random(len(mean_returns))` draws one uniform random number in $[0, 1)$ **for each asset** (here 5 numbers).
2. `weights /= np.sum(weights)` divides each weight by the sum of all weights, so that the weights are **normalized to sum to 1** (fully invested portfolio, no short selling since all weights are non-negative).
3. The metrics of that random portfolio are computed by calling `calc_portfolio_perf`.

Results are stored in a pre-allocated matrix `results_matrix = np.zeros((len(mean_returns)+3, num_portfolios))`: one column per portfolio; the first 3 rows hold *Return*, *Standard Deviation* and *Sharpe*, and the remaining rows (`+3` offset so the metrics are not overwritten) hold the weight of each asset.

**Output:** the matrix is transposed and returned as a pandas DataFrame (`results_df`) where **each row is one simulated portfolio** with columns:

```
Return | Standard_Deviation | Sharpe | EBS.VI | OMV.VI | UQA.VI | VER.VI | VOE.VI
```

so for every portfolio you see both its metrics and how much to invest in each asset.

---

## 3. Empirical computation

> **Important:** the dataset contains *prices*, but all portfolio theory works on *returns*. `pct_change()` converts prices into daily percentage returns.

```python
mean_returns = df.pct_change().mean()   # mean daily return of each asset
cov          = df.pct_change().cov()    # covariance matrix of daily returns
num_portfolios = 100000                 # number of simulated portfolios
rf = 0.035                              # assumed risk-free rate (e.g. proxy: Euribor)
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)
```

The two notable portfolios are then located inside `results_frame`:

```python
max_sharpe_port = results_frame.iloc[results_frame['Sharpe'].idxmax()]              # tangency portfolio
min_vol_port    = results_frame.iloc[results_frame['Standard_Deviation'].idxmin()]  # minimum variance portfolio
```

`idxmax()` / `idxmin()` return the row index of the best value and `iloc` extracts the whole row (a pandas **Series** containing metrics + weights).

A first scatter plot shows all 100,000 portfolios in the risk–return plane (x = standard deviation, y = return), colored by Sharpe ratio (`cmap='RdYlBu'`), with a **red star** on the Maximum Sharpe portfolio and a **green star** on the Minimum Volatility portfolio. The upper-left boundary of the cloud is the **efficient frontier**.

---

## 4. Capital Market Line (CML)

The CML is the straight line that starts from the risk-free rate on the y-axis and is **tangent to the efficient frontier at the Maximum Sharpe Ratio portfolio** (the *tangency portfolio*). It represents all the portfolios obtainable by combining the risk-free asset with the tangency portfolio.

Mathematical form of the line:

$$R = r_f + \underbrace{\frac{R_{tan} - r_f}{\sigma_{tan}}}_{\text{slope}} \cdot \sigma$$

where $(\sigma_{tan}, R_{tan})$ are the volatility and return of the tangency portfolio. Note that the slope of the CML **is exactly the Sharpe ratio of the tangency portfolio**.

How it is computed in the notebook — simple linear algebra / analytic geometry. The slope of a line through two known points $P_1(x_1, y_1)$ and $P_2(x_2, y_2)$ is $\frac{y_2 - y_1}{x_2 - x_1}$. Here the two points are:

- $P_1 = (0,\ r_f)$ — the risk-free asset: zero volatility, certain return $r_f = 0.035$;
- $P_2 = (\sigma_{tan},\ R_{tan})$ — the Maximum Sharpe portfolio found in the simulation.

```python
slope = (sharpe_port_return - rf) / sharpe_port_risk
```

To draw the line, 100 evenly spaced x-values (volatilities) are generated with `np.linspace(0, 0.32, 100)` and the line equation is evaluated at each point:

```python
x_cml = np.linspace(0, 0.32, 100)
y_cml = rf + slope * x_cml
```

The final plot overlays the CML (blue line) on the simulated-portfolio cloud, again highlighting the Maximum Sharpe portfolio (red star) and the Minimum Volatility portfolio (green star).

---

## 5. Output: the two optimal portfolios as DataFrames

Since `max_sharpe_port` and `min_vol_port` are pandas **Series**, the last cells convert each of them into **two new one-row DataFrames** for a clean, readable display:

```python
max_sharpe_port.to_frame().T   # Maximum Sharpe Ratio portfolio
min_vol_port.to_frame().T      # Minimum Volatility portfolio
```

Each resulting DataFrame has one row and the columns `Return`, `Standard_Deviation`, `Sharpe` plus one column per ticker with the **optimal weights** — i.e. the exact composition of the two outlined portfolios. The full simulation table can be inspected with `print(results_frame)`.

---

## Requirements

```
pandas, numpy, matplotlib, yfinance
```

Recommended environment: conda `arima` (see the `interactive-brokers-api` repo, file `arima.yml`). To run the notebook just **select the `arima` kernel** in your editor — no `conda activate` needed in the terminal.

> Note: the notebook downloads data from Yahoo Finance, so an Internet connection is required (but **no** TWS/Interactive Brokers). The yfinance API changes often: if the download fails, update the library with `pip install -U yfinance`.
