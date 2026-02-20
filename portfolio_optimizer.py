"""
LWMA Emerging Markets Portfolio Optimizer
Mean-Variance Optimization (MVO) - Modern Portfolio Theory

Steps:
  1. Download 10 years of monthly returns  (direct Yahoo Finance API, no cache)
  2. Compute mean returns vector + covariance matrix
  3. Run Mean-Variance (max Sharpe) and Minimum Variance optimizations
  4. Report portfolio volatility, expected return, CVaR

Dependencies: numpy, pandas, requests, matplotlib   (no scipy / no yfinance)
"""

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PORTFOLIO = {
    "VWO":   0.40,    # Vanguard FTSE Emerging Markets ETF — broad EM
    "EEMV":  0.40,    # iShares MSCI EM Min Vol ETF — volatility dampener
    "HDB":   0.05,    # HDFC Bank (ADR) — defensive EM financials
    "TSM":   0.05,    # Taiwan Semiconductor (ADR) — tech infrastructure
    "TCEHY": 0.04,    # Tencent Holdings (OTC) — controlled China exposure
    "ITUB":  0.03,    # Itau Unibanco (ADR) — LatAm banking
    "VALE":  0.03,    # Vale S.A. (ADR) — commodity hedge
}

RISK_FREE_RATE = 0.045       # annualized (current T-bill ~ 4.5%)
CONFIDENCE_LEVEL = 0.95      # for VaR / CVaR
LOOKBACK_YEARS = 10
MAX_WEIGHT = 0.40            # per-asset ceiling
MIN_WEIGHT = 0.00            # no short selling


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download via Yahoo Finance v8 chart API (no yfinance, no cache)
# ═══════════════════════════════════════════════════════════════════════════════

_HEADERS = {"User-Agent": "Mozilla/5.0"}
_BASE = "https://query2.finance.yahoo.com/v8/finance/chart"


def _fetch_monthly_close(ticker: str, years: int) -> pd.Series | None:
    """Fetch monthly adjusted close from Yahoo Finance chart API."""
    end_ts = int(datetime.today().timestamp())
    start_ts = int((datetime.today() - timedelta(days=years * 365)).timestamp())

    url = f"{_BASE}/{ticker}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": "1mo",
        "events": "history",
    }

    try:
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    {ticker:<8} -- error: {exc}")
        return None

    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error", {}).get("description", "unknown")
        print(f"    {ticker:<8} -- API error: {err}")
        return None

    r = result[0]
    timestamps = r.get("timestamp", [])
    adj = r.get("indicators", {}).get("adjclose")
    if adj and adj[0].get("adjclose"):
        closes = adj[0]["adjclose"]
    else:
        closes = r.get("indicators", {}).get("quote", [{}])[0].get("close", [])

    if not timestamps or not closes:
        print(f"    {ticker:<8} -- no price data returned")
        return None

    dates = pd.to_datetime(timestamps, unit="s").normalize()
    s = pd.Series(closes, index=dates, name=ticker, dtype=float)
    s.dropna(inplace=True)
    return s


def download_returns(tickers: list[str], years: int = 10) -> pd.DataFrame:
    print(f"  Downloading {years}Y monthly data for {len(tickers)} assets...\n")

    all_series = {}
    for t in tickers:
        s = _fetch_monthly_close(t, years)
        if s is not None and len(s) > 12:
            all_series[t] = s
            print(f"    {t:<8}  {len(s):>3} months")
        else:
            print(f"    {t:<8} -- skipped (insufficient data)")
        time.sleep(0.3)

    if not all_series:
        raise RuntimeError("Could not download data for any ticker.")

    prices = pd.DataFrame(all_series)
    prices.sort_index(inplace=True)
    prices.dropna(how="all", inplace=True)

    returns = np.log(prices / prices.shift(1)).dropna()

    if returns.isnull().any().any():
        for col in returns.columns[returns.isnull().any()]:
            n = int(returns[col].isnull().sum())
            print(f"    {col}: {n} missing months (forward-filled)")
        returns.ffill(inplace=True)
        returns.fillna(0, inplace=True)

    print(f"\n  Period : {returns.index[0].strftime('%Y-%m')} to "
          f"{returns.index[-1].strftime('%Y-%m')}")
    print(f"  Months : {len(returns)}")
    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute Mean Returns + Covariance Matrix
# ═══════════════════════════════════════════════════════════════════════════════

def compute_statistics(returns: pd.DataFrame):
    mean_returns = returns.mean() * 12     # annualized
    cov_matrix   = returns.cov()  * 12     # annualized
    return mean_returns, cov_matrix


# ═══════════════════════════════════════════════════════════════════════════════
# QP SOLVER — analytical active-set method (numpy only, no scipy)
#
# Solves:  min  0.5 w'H w + g'w
#          s.t. sum(w) = 1,  lb <= w_i <= ub
#
# For n=9 assets this gives exact solutions in a handful of 9x9 inversions.
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_qp(H, g, lb=0.0, ub=0.4):
    n = H.shape[0]
    at_lower = np.zeros(n, dtype=bool)
    at_upper = np.zeros(n, dtype=bool)
    lam = 0.0

    for _ in range(4 * n):
        free = ~(at_lower | at_upper)
        free_idx = np.where(free)[0]
        n_free = len(free_idx)

        if n_free == 0:
            w = np.where(at_upper, ub, lb)
            s = w.sum()
            if s > 0:
                w /= s
            return w

        budget = 1.0 - lb * at_lower.sum() - ub * at_upper.sum()

        w_fixed = np.zeros(n)
        w_fixed[at_lower] = lb
        w_fixed[at_upper] = ub

        H_ff = H[np.ix_(free_idx, free_idx)]
        g_f = g[free_idx].copy()

        fixed_idx = np.where(~free)[0]
        if len(fixed_idx) > 0:
            g_f += H[np.ix_(free_idx, fixed_idx)] @ w_fixed[fixed_idx]

        H_ff_inv = np.linalg.inv(H_ff)
        e_f = np.ones(n_free)

        inv_g = H_ff_inv @ g_f
        inv_e = H_ff_inv @ e_f
        denom = e_f @ inv_e
        if abs(denom) < 1e-15:
            w_f = np.full(n_free, budget / max(n_free, 1))
        else:
            lam = -(budget + e_f @ inv_g) / denom
            w_f = H_ff_inv @ (-g_f - lam * e_f)

        w = w_fixed.copy()
        w[free_idx] = w_f

        # Check if any free variable violates bounds
        changed = False
        for i in free_idx:
            if w[i] < lb - 1e-10:
                at_lower[i] = True
                changed = True
            elif w[i] > ub + 1e-10:
                at_upper[i] = True
                changed = True

        if not changed:
            # Check KKT dual feasibility for bound-active variables
            dual = H @ w + g + lam * np.ones(n)
            release = False
            for i in range(n):
                if at_lower[i] and dual[i] < -1e-8:
                    at_lower[i] = False
                    release = True
                    break
                if at_upper[i] and dual[i] > 1e-8:
                    at_upper[i] = False
                    release = True
                    break
            if not release:
                break

    w = np.clip(w, lb, ub)
    if abs(w.sum() - 1.0) > 1e-8:
        w /= w.sum()
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Optimization
# ═══════════════════════════════════════════════════════════════════════════════

def portfolio_performance(weights, mean_returns, cov_matrix):
    mu  = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-12 else 0.0
    return ret, vol, sharpe


def optimize_min_variance(mean_returns, cov_matrix):
    cov = np.asarray(cov_matrix)
    return _solve_qp(2.0 * cov, np.zeros(cov.shape[0]), MIN_WEIGHT, MAX_WEIGHT)


def _optimize_target_return(mean_returns, cov_matrix, target, penalty=5000.0):
    mu  = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    H = 2.0 * cov + 2.0 * penalty * np.outer(mu, mu)
    g = -2.0 * penalty * target * mu
    return _solve_qp(H, g, MIN_WEIGHT, MAX_WEIGHT)


def optimize_max_sharpe(mean_returns, cov_matrix):
    mu  = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    w_min  = optimize_min_variance(mean_returns, cov_matrix)
    ret_lo = w_min @ mu
    ret_hi = mu.max()
    best_sharpe = -np.inf
    best_w = w_min.copy()
    for target in np.linspace(ret_lo, ret_hi, 150):
        w   = _optimize_target_return(mean_returns, cov_matrix, target)
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        s   = (ret - RISK_FREE_RATE) / vol if vol > 1e-12 else -np.inf
        if s > best_sharpe:
            best_sharpe = s
            best_w = w.copy()
    return best_w


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_var(weights, returns: pd.DataFrame, confidence=0.95):
    port_ret = returns.values @ weights
    return float(np.percentile(port_ret, (1 - confidence) * 100))


def compute_cvar(weights, returns: pd.DataFrame, confidence=0.95):
    port_ret = returns.values @ weights
    cutoff = np.percentile(port_ret, (1 - confidence) * 100)
    tail = port_ret[port_ret <= cutoff]
    return float(tail.mean()) if len(tail) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def compute_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    mu  = np.asarray(mean_returns)
    cov = np.asarray(cov_matrix)
    w_min  = optimize_min_variance(mean_returns, cov_matrix)
    ret_lo = w_min @ mu
    ret_hi = mu.max() * 0.98
    f_ret, f_vol = [], []
    for target in np.linspace(ret_lo, ret_hi, n_points):
        w = _optimize_target_return(mean_returns, cov_matrix, target)
        f_ret.append(w @ mu)
        f_vol.append(np.sqrt(w @ cov @ w))
    return np.array(f_ret), np.array(f_vol)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(mean_returns, cov_matrix, returns,
                 current_w, max_sharpe_w, min_var_w,
                 frontier_ret, frontier_vol, tickers):

    cov = np.asarray(cov_matrix)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("LWMA Emerging Markets Portfolio \u2014 Mean-Variance Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    # ── Left: Efficient Frontier ──
    ax = axes[0]
    ax.plot(frontier_vol * 100, frontier_ret * 100, "b-", lw=2.5,
            label="Efficient Frontier")

    markers = [
        ("Current Portfolio", current_w,    "#e74c3c", "s",  130),
        ("Max Sharpe",        max_sharpe_w, "#f39c12", "*",  220),
        ("Min Variance",      min_var_w,    "#27ae60", "D",  110),
    ]
    for name, w, color, marker, sz in markers:
        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        ax.scatter(vol * 100, ret * 100, c=color, marker=marker, s=sz,
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=f"{name}  (Sharpe {sharpe:.2f})")

    for i, t in enumerate(tickers):
        v = np.sqrt(cov[i, i]) * 100
        r = float(mean_returns.iloc[i]) * 100
        ax.scatter(v, r, c="grey", s=30, alpha=0.5, zorder=3)
        ax.annotate(t, (v, r), fontsize=7.5, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Efficient Frontier")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Right: Weight Comparison ──
    ax2 = axes[1]
    x = np.arange(len(tickers))
    bw = 0.25
    ax2.barh(x - bw, current_w * 100,   bw, label="Current",      color="#e74c3c", alpha=0.8)
    ax2.barh(x,      max_sharpe_w * 100, bw, label="Max Sharpe",   color="#f39c12", alpha=0.8)
    ax2.barh(x + bw, min_var_w * 100,    bw, label="Min Variance", color="#27ae60", alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(tickers)
    ax2.set_xlabel("Weight (%)")
    ax2.set_title("Weight Comparison")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("efficient_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Chart saved -> efficient_frontier.png")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_portfolio(label, weights, tickers, mean_returns, cov_matrix, returns):
    ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
    cvar = compute_cvar(weights, returns, CONFIDENCE_LEVEL)
    var_ = compute_var(weights, returns, CONFIDENCE_LEVEL)

    print(f"\n  {'=' * 52}")
    print(f"   {label}")
    print(f"  {'=' * 52}")
    print(f"   Expected Return   {ret * 100:>10.2f} %  (annualized)")
    print(f"   Volatility        {vol * 100:>10.2f} %  (annualized)")
    print(f"   Sharpe Ratio      {sharpe:>10.2f}")
    print(f"   VaR  (95%, mo)    {var_ * 100:>10.2f} %")
    print(f"   CVaR (95%, mo)    {cvar * 100:>10.2f} %")
    print(f"  {'-' * 52}")
    for t, wt in zip(tickers, weights):
        bar = "#" * int(wt * 40)
        print(f"   {t:<8} {wt * 100:>6.2f}%  {bar}")
    print()


def print_comparison_table(current_w, max_sharpe_w, min_var_w,
                           mean_returns, cov_matrix, returns):
    print(f"\n  {'=' * 62}")
    print("   SIDE-BY-SIDE COMPARISON")
    print(f"  {'=' * 62}")
    print(f"  {'Metric':<24} {'Current':>12} {'Max Sharpe':>12} {'Min Var':>12}")
    print(f"  {'-' * 62}")

    rows = []
    for w in [current_w, max_sharpe_w, min_var_w]:
        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix)
        cvar = compute_cvar(w, returns, CONFIDENCE_LEVEL)
        var_ = compute_var(w, returns, CONFIDENCE_LEVEL)
        rows.append((ret, vol, sharpe, var_, cvar))

    labels = [
        ("Exp. Return  (%)",   True),
        ("Volatility   (%)",   True),
        ("Sharpe Ratio",       False),
        ("VaR  95%  (mo, %)",  True),
        ("CVaR 95%  (mo, %)",  True),
    ]
    for i, (label, is_pct) in enumerate(labels):
        v = [rows[j][i] for j in range(3)]
        if is_pct:
            print(f"  {label:<24} {v[0]*100:>11.2f}% "
                  f"{v[1]*100:>11.2f}% {v[2]*100:>11.2f}%")
        else:
            print(f"  {label:<24} {v[0]:>12.2f} "
                  f"{v[1]:>12.2f} {v[2]:>12.2f}")
    print(f"  {'=' * 62}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("=" * 62)
    print("  LWMA Emerging Markets Portfolio Optimizer")
    print("  Mean-Variance Optimization  |  Modern Portfolio Theory")
    print("=" * 62)

    tickers = list(PORTFOLIO.keys())
    current_weights = np.array(list(PORTFOLIO.values()))

    # ── Step 1 ────────────────────────────────────────────────────────────
    print("\n STEP 1  Download historical monthly returns\n")
    returns = download_returns(tickers, years=LOOKBACK_YEARS)

    available = list(returns.columns)
    if set(available) != set(tickers):
        dropped = set(tickers) - set(available)
        print(f"\n  Re-weighting after dropping: {dropped}")
        mask = [t in available for t in tickers]
        tickers = [t for t, m in zip(tickers, mask) if m]
        current_weights = np.array([PORTFOLIO[t] for t in tickers])
        current_weights /= current_weights.sum()

    # ── Step 2 ────────────────────────────────────────────────────────────
    print("\n STEP 2  Compute statistics\n")
    mean_ret, cov_mat = compute_statistics(returns)

    print("  Annualized Mean Returns:")
    for t in tickers:
        print(f"    {t:<8} {mean_ret[t] * 100:>8.2f} %")

    print("\n  Correlation Matrix:\n")
    print(returns.corr().round(2).to_string())

    # ── Step 3 ────────────────────────────────────────────────────────────
    print(f"\n STEP 3  Optimization\n")
    print(f"  Constraints: long-only, max {MAX_WEIGHT*100:.0f}% per asset, fully invested\n")

    print("  Solving minimum-variance portfolio...")
    min_var_w = optimize_min_variance(mean_ret, cov_mat)

    print("  Solving max-Sharpe portfolio...")
    max_sharpe_w = optimize_max_sharpe(mean_ret, cov_mat)

    print_portfolio("CURRENT PORTFOLIO",
                    current_weights, tickers, mean_ret, cov_mat, returns)
    print_portfolio("MAX SHARPE (OPTIMIZED)",
                    max_sharpe_w, tickers, mean_ret, cov_mat, returns)
    print_portfolio("MINIMUM VARIANCE",
                    min_var_w, tickers, mean_ret, cov_mat, returns)

    # ── Step 4 ────────────────────────────────────────────────────────────
    print(" STEP 4  Risk Metrics\n")
    print_comparison_table(current_weights, max_sharpe_w, min_var_w,
                           mean_ret, cov_mat, returns)

    # ── Efficient Frontier ────────────────────────────────────────────────
    print("\n  Computing efficient frontier...")
    f_ret, f_vol = compute_efficient_frontier(mean_ret, cov_mat)

    plot_results(mean_ret, cov_mat, returns,
                 current_weights, max_sharpe_w, min_var_w,
                 f_ret, f_vol, tickers)

    # ── Export ────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "Ticker":         tickers,
        "Current_Wt":     np.round(current_weights, 4),
        "MaxSharpe_Wt":   np.round(max_sharpe_w, 4),
        "MinVariance_Wt": np.round(min_var_w, 4),
        "Ann_Return":     np.round(mean_ret.values, 4),
        "Ann_Volatility": np.round(
            [np.sqrt(cov_mat.iloc[i, i]) for i in range(len(tickers))], 4),
    })
    df.to_csv("optimization_results.csv", index=False)
    print("\n  Results exported -> optimization_results.csv")

    print("\n" + "=" * 62)
    print("  Done.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
