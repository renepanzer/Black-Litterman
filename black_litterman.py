import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches


# Ledoit Wolf Shrinkage stabilizes covariance matrix
def ledoit_wolf_covariance(returns):
    lw = LedoitWolf().fit(returns)
    return lw.covariance_


def black_litterman_expected_returns(Sigma, market_weights, risk_aversion, P, Q, Omega, tau):
    """
    This function calculates the expected returns based on the Black-Litterman formula:
        E[R]=[(tau*Sigma)^-1+P'Omega^-1P]^-1[(tauSigma)^-1Pi+P'Omega^-1Q]
    E[R]: Nx1 posterior combined return vector
    Sigma: NxN covariance matrix of dividend-adjusted returns
    market_weights: Nx1
    P: KxN vector that identifies the assets involved in the view
    Q: Kx1 vector with views
    Omega: KxK diagonal covariance matrix with view uncertainty
    risk_aversion (lambda): expected risk-return tradeoff. Rate at which an investor will forgo expected returns for less variance
        possible to calculate as risk premium / variance of the market excess returns ~ 3.07 (Idzorek)
        (E(r)-r_f)/sigma^2
    tau: scaling factor
    """

    # Implied equilibrium excess returns 
    pi = risk_aversion * (Sigma @ market_weights)
    
    # BL posterior mean
    tauSigma_inv = np.linalg.inv(tau * Sigma)
    Omega_inv = np.linalg.inv(Omega)
    # Posterior covariance matrix
    M = np.linalg.inv(tauSigma_inv + P.T @ Omega_inv @ P)
    mu_bl = M @ (tauSigma_inv @ pi + P.T @ Omega_inv @ Q)
    Sigma_post = Sigma + M

    return mu_bl, Sigma_post


def bl_optimiser(mu, Sigma, tickers, sectors, risk_aversion, sector_limits):
    N = len(mu)
    
    # Variable to be optimised, i.e. weights
    w = cp.Variable(N)

    # calculate expected return and variance and ensure covariance matrix is PSD 
    expected_return = mu @ w
    Sigma = 0.5 * (Sigma + Sigma.T)
    variance = cp.quad_form(w, Sigma)

    # Objective: maximise mu'w - lambda*w'Sigmaw
    objective = cp.Maximize(expected_return - (0.5 * risk_aversion) * variance)


    """
    Define constraints:
    1. Long only: w>0
    2. 90% invested: Sum(w_i) = 0.9 (10% are always allocated to Cash)
    3. Max. 5% invested in a single stock
    4. Max. 20% invested in a single sector     
    """
    constraints = [w >= 0,
                   cp.sum(w) == 0.9,
                   w <= 0.05]

    # Sector constraints
    for sector, limit in sector_limits.items():
        mask = (sectors == sector).astype(float)
        constraints.append(mask @ w <= limit)

    # use OSQP solver
    # could also use different solver or scipy.optimise 
    problem = cp.Problem(objective, constraints)
    
    problem.solve(solver=cp.OSQP)
    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
     raise RuntimeError(f"Optimiser failed: {problem.status}")

    weights = pd.Series(np.asarray(w.value).reshape(-1), index=tickers)
    weights = weights.clip(lower=0)
    weights["CASH"] = 0.1
    
    return weights


# dictionary of stocks including industry that should form part of the active portfolio
stocks = {
    "DTG.DE": "Ind", "FGR.PA": "Ind", "DHL.DE": "Ind", "WMS": "Ind", "CAT": "Ind",
    "RPM": "Mats", "CRH": "Mats", "HEI.DE": "Mats", "NUE": "Mats", "PRU.AX": "Mats",
    "NXT": "Tech", "IDCC": "Tech", "6690.HK": "Tech",
    "MKS.L": "Stap", "9989.T": "Stap", "NOMD": "Stap", "SHP.JO": "Stap", 
    "RCL": "Disc", "BMW.DE": "Disc",
    "RMV.L": "Telc", 
    "AEP": "Ener", "IBE.MC": "Ener", "RWE.DE": "Ener", "TTE": "Ener", "NGG": "Ener"
    }

# start and end dates to find returns and compute covariance matrix
# duration should roughly equal investment horizon
start_date = "2022-03-01"
end_date   = "2026-03-01"


tickers = list(stocks.keys())
sectors = np.array([stocks[t] for t in tickers])  


# get closing prics from yfinance
prices = yf.download(tickers,
                     start=start_date,
                     end=end_date,
                     auto_adjust=True,
                     progress=False)["Close"]

missing = [t for t in tickers if t not in prices.columns]

if missing:
    raise ValueError(f"No price data returned for: {missing}")
# reorder assets and drop nans 
# dropna() really isn't ideal here because one of the stocks was only listed halfway through 
# the time period, so it removes a lot of otherwise good data entries. ffill() would be an 
# alternative but results in zero return days for missing data from public holidays etc.
prices = prices.reindex(columns=tickers).dropna()
#prices = prices.reindex(columns=tickers).ffill()


# compute daily returns and covariance matrix (annualise values)
ret_daily = prices.pct_change().dropna() 
Sigma_ann = ledoit_wolf_covariance(ret_daily.values) * 252


N = len(tickers)
market_weights = np.ones(N) / N # really should get the market-cap weights; just used equal weighted for simplicity but theoretically it's incorrect
risk_aversion = 3.0 # idzorek ~market
tau = 0.025 # tau largely cancels, so the actual value is not as important


#construct view vector of assets
view_assets = ["DTG.DE", "FGR.PA", "DHL.DE", "WMS", "CAT",
               "RPM", "CRH", "HEI.DE", "NUE", "PRU.AX",
               "NXT", "IDCC", "6690.HK",
               "MKS.L", "9989.T", "NOMD", "SHP.JO",
               "RCL", "BMW.DE",
               "RMV.L",
               "AEP", "IBE.MC", "RWE.DE", "TTE", "NGG"]

# absolute returns based on median analyst forecast. Should ideally be determined from DCF models
view_returns = np.array([-0.01, 0.15, -0.07, 0.14, -0.05,
                         0.17, 0.12, 0.09, 0.02, 0.25,
                         0.03, 0.39, 0.37,
                         0.11, 0.29, 0.30, 0.20,
                         0.13, 0.06,
                         0.41,
                         0.08, -0.03, -0.01, 0.05, -0.03])  

# here views are absolute, i.e. AAPL will rise 3% in the next 12months; diagnoal in view matrix
# but views can also be relative; i.e. APPL will fall 5% compared to MSFT; off-diagonals in view matrix
# relative views are difficult to estimate and without proper methodology, it's better to focus on absolut returns only
K = len(view_assets)
P = np.zeros((K, N))
for k, t in enumerate(view_assets):
    P[k, tickers.index(t)] = 1.0

rf_annual = 0.04   # ideally the actual horizon-matched risk-free rate from FRED for example
Q = view_returns - rf_annual 

# uncertainty of mean, scaled to the prior
Omega = np.diag(np.diag(P @ (tau * Sigma_ann) @ P.T))


# find expected returns from Black-Litterman
mu_bl, Sigma_post = black_litterman_expected_returns(Sigma=Sigma_ann,
                                                     market_weights=market_weights,
                                                     risk_aversion=risk_aversion,
                                                     P=P,
                                                     Q=Q,
                                                     Omega=Omega,
                                                     tau=tau)

# max. 20% exposure to a single sector 
sector_limits = {s: 0.20 for s in np.unique(sectors)} 

# Mean-Variance optimisation
weights = bl_optimiser(mu=mu_bl,
                       Sigma=Sigma_post,
                       tickers=tickers,
                       sectors=sectors,
                       risk_aversion=risk_aversion,
                       sector_limits=sector_limits)

# print weights
print(weights.sort_values(ascending=False).round(4))


#====================================================================================
# Outputs: weights spreadsheet and allocation charts

def weights_table(weights, stocks):
    """
    Build a table of the optimal weights, one row per asset, sorted from
    largest to smallest, plus sector weights.
    """
    table = pd.DataFrame({"Ticker": weights.index,
                          "Sector": [stocks.get(t, "Cash") for t in weights.index],
                          "Weight": weights.values})

    table = table.sort_values("Weight", ascending=False).reset_index(drop=True)
    table["Weight (%)"] = (table["Weight"] * 100).round(2)

    by_sector = (table.groupby("Sector", as_index=False)["Weight"].sum()
                      .sort_values("Weight", ascending=False)
                      .reset_index(drop=True))

    by_sector["Weight (%)"] = (by_sector["Weight"] * 100).round(2)

    return table, by_sector


def save_to_excel(table, by_sector, path="results/optimal_weights.xlsx"):
    """
    Spreadsheet with asset and sector weights
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Asset Weights", index=False)
        by_sector.to_excel(writer, sheet_name="Sector Weights", index=False)

    print(f"Weights written to {path}")


def plot_sector_allocation(by_sector, path="results/figures/Sector-Allocation.png"):
    """
    Pie chart of the portfolio weights by sector.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.pie(by_sector["Weight"],
            labels=by_sector["Sector"],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 1})
    plt.title("Sector Allocation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_asset_allocation(table, path="results/figures/Asset-Allocation.png"):
    """
    Horizontal bar chart of the individual asset weights, coloured by sector.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    active_weights = table[table["Weight"] > 1e-6].copy()

    sectors_present = sorted(active_weights["Sector"].unique())
    cmap = plt.get_cmap("tab10")
    sector_colours = {s: cmap(i % 10) for i, s in enumerate(sectors_present)}
    colours = [sector_colours[s] for s in active_weights["Sector"]]

    plt.figure(figsize=(10, 8))
    plt.barh(active_weights["Ticker"], active_weights["Weight (%)"], color=colours)
    plt.gca().invert_yaxis()
    plt.xlabel("Weight (%)", fontsize=12)
    plt.title("Asset Allocation", fontsize=14, fontweight="bold")

    handles = [mpatches.Patch(color=sector_colours[s], label=s) for s in sectors_present]
    plt.legend(handles=handles, title="Sector", loc="lower right")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


table, by_sector = weights_table(weights, stocks)

print(by_sector.to_string(index=False))

save_to_excel(table, by_sector)
plot_sector_allocation(by_sector)
plot_asset_allocation(table)