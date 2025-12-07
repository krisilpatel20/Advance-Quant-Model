import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta

# Try importing arch, handle if missing
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Quant Thesis: Advanced Models (Filtered)", layout="wide")
plt.style.use('ggplot')

st.title("Results of Advanced Quantitative Thesis (Filtered Probabilities)")
st.markdown("""
**Robust Financial Modeling Dashboard** incorporating:
GARCH/EGARCH | Regime Switching (Filtered) | Jump Diffusion | Heston Stochastic Vol | Kalman Filter Pairs | Macro Factors
""")

if not ARCH_AVAILABLE:
    st.error("⚠️ The 'arch' library is not installed. GARCH/EGARCH modules will be limited. Run `pip install arch`.")

# ==========================================
# 2. HELPER CLASSES & FUNCTIONS
# ==========================================

def format_plot_dates(ax, dates):
    """
    Helper to format x-axis dates for better readability.
    Handles Weekly (1/1, 1/8) and Monthly (Sep, Oct) gaps.
    """
    if len(dates) == 0:
        return
        
    # Convert to datetime if not already
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
        
    span_days = (dates[-1] - dates[0]).days
    
    # Locator and Formatter logic
    if span_days < 90: # Less than 3 months -> Weekly
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else: 
        # User requested granular monthly ticks even for long periods
        # We use MonthLocator. For very long periods, matplotlib might auto-hide some,
        # but we set the locator explicitly.
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center', fontsize=8)

class Calibrator:
    @staticmethod
    def calibrate_heston(returns):
        """
        Estimates Heston parameters from historical returns using a Method of Moments approach.
        
        NOTE: This is a 'Historical Proxy Calibration'. It fits parameters to the historical 
        volatility dynamics (via GARCH proxy), not to option prices. It captures how volatility 
        *has behaved*, not necessarily how the market *prices* future volatility (Implied Vol).
        """
        dt = 1/252
        
        # 1. Estimate Volatility Dynamics
        # Use GARCH(1,1) to get a proxy for spot volatility series
        am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1, dist='Normal')
        res = am.fit(disp='off')
        conditional_vol = res.conditional_volatility / 100 # Convert back to decimal
        variance = conditional_vol**2
        
        # 2. Estimate Heston Parameters from the variance series
        # dv = kappa(theta - v)dt + xi*sqrt(v)dW
        # Discretized: v_{t+1} - v_t = kappa*theta*dt - kappa*v_t*dt + noise
        # Regression: Y = alpha + beta*X + epsilon
        # Y = (v_{t+1} - v_t)/dt, X = v_t
        # beta = -kappa, alpha = kappa*theta -> theta = -alpha/beta
        
        variance = variance.values if hasattr(variance, 'values') else variance
        v_curr = variance[:-1]
        v_next = variance[1:]
        Y = (v_next - v_curr) / dt
        X = v_curr
        
        # Simple Linear Regression
        A = np.vstack([X, np.ones(len(X))]).T
        beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        
        kappa = -beta
        theta = alpha / kappa if kappa != 0 else np.mean(variance)
        
        # Ensure positive parameters
        kappa = max(kappa, 0.1)
        theta = max(theta, 0.01)
        
        # Estimate Vol of Vol (xi)
        # Residuals of the regression approximate xi * sqrt(v) * dW
        residuals = Y - (alpha + beta * X)
        # Var(residuals) approx xi^2 * v / dt
        # This is rough, let's just take std of residuals * sqrt(dt) / mean(sqrt(v))
        xi = np.std(residuals) * np.sqrt(dt) / np.mean(np.sqrt(v_curr))
        xi = max(xi, 0.1)
        
        # Estimate Correlation (rho)
        # Corr(returns, variance_changes)
        # Heston assumes corr between price process and vol process Brownian motions
        rho = np.corrcoef(returns[1:], np.diff(variance))[0, 1]
        
        # Estimate Drift (mu)
        mu = np.mean(returns) / dt + 0.5 * np.mean(variance) # Annualized geometric drift adjustment
        
        return {
            'mu': mu,
            'kappa': kappa,
            'theta': theta,
            'xi': xi,
            'rho': rho,
            'v0': variance[-1],
            'S0': 100.0 
        }

class KalmanFilterReg:
    """
    A simple Kalman Filter implementation to estimate the dynamic beta (slope)
    and alpha (intercept) between two asset price series.
    """
    def __init__(self, delta=1e-4, R=1e-3):
        # delta: process noise covariance (flexibility of beta)
        # R: measurement noise covariance
        self.delta = delta
        self.R = R
        self.trans_cov = delta / (1 - delta) * np.eye(2) # Process noise matrix
        self.obs_mat = np.expand_dims(np.vstack([[1], [1]]), axis=1) # Initial observation matrix placeholder

    def run_filter(self, y, x):
        """
        y: Dependent variable (Target Ticker)
        x: Independent variable (Reference Ticker)
        """
        n = len(y)
        state_mean = np.zeros((n, 2)) # [Alpha, Beta]
        state_cov = np.zeros((n, 2, 2))
        
        # Initial guesses
        state_mean[0] = [0, 1]
        state_cov[0] = np.eye(2)
        
        for t in range(1, n):
            # 1. Prediction Step
            # State stays same (Random Walk hypothesis), Covariance increases by process noise
            pred_state = state_mean[t-1]
            pred_cov = state_cov[t-1] + self.trans_cov
            
            # 2. Observation Step
            obs_mat = np.array([[1.0, x[t]]]) # Observation matrix H = [1, x_t]
            
            # Innovation (Prediction Error)
            y_pred = np.dot(obs_mat, pred_state)
            error = y[t] - y_pred
            
            # Innovation Covariance
            S = np.dot(np.dot(obs_mat, pred_cov), obs_mat.T) + self.R
            
            # Kalman Gain
            K = np.dot(pred_cov, obs_mat.T) / S
            
            # 3. Update Step
            state_mean[t] = pred_state + (K.flatten() * error)
            state_cov[t] = pred_cov - np.dot(np.dot(K, obs_mat), pred_cov)
            
        return state_mean, state_cov

class KalmanFilterTrend:
    """
    Local level model for trend extraction: y_t = mu_t + noise
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.Q = process_noise
        self.R = measurement_noise
    
    def filter(self, data):
        """Forward pass (causal estimates)"""
        n = len(data)
        estimates = np.zeros(n)
        covariances = np.zeros(n)
        
        # Initialize with mean of first 10 points
        init_window = min(10, n // 10)
        x = np.mean(data[:init_window])
        P = np.var(data[:init_window]) if init_window > 1 else 1.0
        
        for t in range(n):
            # Predict
            x_pred = x
            P_pred = P + self.Q
            
            # Update
            K = P_pred / (P_pred + self.R)
            x = x_pred + K * (data[t] - x_pred)
            P = (1 - K) * P_pred
            
            estimates[t] = x
            covariances[t] = P
        
        return estimates, covariances
    
    def smooth(self, data):
        """Forward + backward pass (uses all data)"""
        n = len(data)
        
        # Forward pass
        filtered_means, filtered_covs = self.filter(data)
        
        # Backward pass
        smoothed_means = np.zeros(n)
        smoothed_covs = np.zeros(n)
        
        smoothed_means[-1] = filtered_means[-1]
        smoothed_covs[-1] = filtered_covs[-1]
        
        for t in range(n - 2, -1, -1):
            P_pred = filtered_covs[t] + self.Q
            J = filtered_covs[t] / P_pred
            
            smoothed_means[t] = filtered_means[t] + J * (smoothed_means[t+1] - filtered_means[t])
            smoothed_covs[t] = filtered_covs[t] + J**2 * (smoothed_covs[t+1] - P_pred)
        
        return smoothed_means, smoothed_covs

def simulate_heston(S0, T, r, kappa, theta, sigma, rho, v0, steps, paths):
    """
    Simulate Monte Carlo paths for Heston Stochastic Volatility Model.
    dS = mu*S*dt + sqrt(v)*S*dW1
    dv = kappa*(theta - v)*dt + sigma*sqrt(v)*dW2
    """
    dt = T/steps
    prices = np.zeros((steps + 1, paths))
    vols = np.zeros((steps + 1, paths))
    prices[0] = S0
    vols[0] = v0
    
    for t in range(1, steps + 1):
        # Generate correlated Brownian motions
        Z1 = np.random.normal(size=paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=paths)
        
        # Volatility Process (ensure non-negative with max(...,0) or abs)
        v_prev = vols[t-1]
        # Discretization using Euler-Maruyama (absorbing barrier at 0 for vol)
        dv = kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.abs(v_prev)) * np.sqrt(dt) * Z2
        v_curr = np.abs(v_prev + dv)
        vols[t] = v_curr
        
        # Price Process
        dS = r * prices[t-1] * dt + np.sqrt(v_curr) * prices[t-1] * np.sqrt(dt) * Z1
        prices[t] = prices[t-1] + dS
        
    return prices, vols

def merton_jump_diffusion(S0, T, r, sigma, lam, mu_j, sigma_j, steps, paths):
    """
    Simulate Merton Jump Diffusion Paths.
    lam: intensity of jumps (jumps per year)
    mu_j: mean of jump size (log)
    sigma_j: std dev of jump size
    """
    dt = T/steps
    prices = np.zeros((steps + 1, paths))
    prices[0] = S0
    
    # Drift correction for jumps so it remains risk-neutral
    # Drift correction for jumps so it remains risk-neutral
    # drift = r - 0.5 * sigma**2 - lam * (exp(mu_j + 0.5*sigma_j²) - 1)
    drift = r - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    
    for t in range(1, steps + 1):
        z = np.random.normal(size=paths)
        # Poisson Jump Component
        # N is number of jumps in this step (usually 0 or 1 for small dt)
        N = np.random.poisson(lam * dt, size=paths)
        # Jump size J
        J = np.random.normal(mu_j, sigma_j, size=paths) * N
        
        # Geometric Brownian Motion + Jump
        # S_t = S_{t-1} * exp( (drift)*dt + sigma*dW + Sum(J) )
        prices[t] = prices[t-1] * np.exp(drift * dt + sigma * np.sqrt(dt) * z + J)
        
    return prices

class BacktestEngine:
    """
    Handles simple vectorised backtesting for regime-based strategies.
    """
    @staticmethod
    def run_strategy(prices, signals, initial_capital=10000.0, trailing_stop_pct=0.0):
        """
        prices: Series of asset prices
        signals: Series of 1 (Long) or 0 (Cash/Neutral). Index must match prices.
        trailing_stop_pct: Float (e.g., 0.05 for 5%). If > 0, applies trailing stop.
        """
        # Align
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]
        
        # Calculate Returns
        returns = prices.pct_change().fillna(0)
        
        # Note: Vectorized approach is hard with path-dependent trailing stop.
        # We will use the loop for everything to be consistent and accurate with stops.
        
        equity_curve = [initial_capital]
        trades = []
        position = 0 # 0: Cash, 1: Long
        entry_price = 0
        entry_date = None
        max_price_since_entry = 0
        
        cash = initial_capital
        holdings = 0
        
        for date, price, signal in zip(prices.index, prices, signals):
            # Mark to Market
            if position == 1:
                current_val = cash + holdings * price
                
                # Check Trailing Stop
                if trailing_stop_pct > 0:
                    max_price_since_entry = max(max_price_since_entry, price)
                    stop_price = max_price_since_entry * (1 - trailing_stop_pct)
                    
                    if price < stop_price:
                        # Trigger Stop Loss
                        position = 0
                        exit_price = price
                        cash = holdings * exit_price
                        holdings = 0
                        
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'Side': 'Long',
                            'Entry Date': entry_date,
                            'Exit Date': date,
                            'Buy Price': entry_price,
                            'Sell Price': exit_price,
                            'PnL (%)': pnl * 100,
                            'Status': 'Trailing Stop'
                        })
                        equity_curve.append(cash)
                        continue # Skip normal signal processing for this bar
            else:
                current_val = cash
            
            # Signal Processing
            if position == 0 and signal == 1:
                # Buy
                position = 1
                entry_price = price
                entry_date = date
                max_price_since_entry = price
                
                holdings = cash / price
                cash = 0
            elif position == 1 and signal == 0:
                # Sell
                position = 0
                exit_price = price
                cash = holdings * exit_price
                holdings = 0
                
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'Side': 'Long',
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Buy Price': entry_price,
                    'Sell Price': exit_price,
                    'PnL (%)': pnl * 100,
                    'Status': 'Closed'
                })
            
            equity_curve.append(current_val)
            
        # Capture Open Position
        if position == 1:
            current_price = prices.iloc[-1]
            current_val = holdings * current_price
            pnl = (current_price - entry_price) / entry_price
            trades.append({
                'Side': 'Long',
                'Entry Date': entry_date,
                'Exit Date': None, # Open
                'Buy Price': entry_price,
                'Sell Price': current_price, # Mark-to-Market
                'PnL (%)': pnl * 100,
                'Status': 'Open'
            })
            equity_curve[-1] = current_val # Update last point
            
        # Convert to Series
        equity_curve_series = pd.Series(equity_curve[1:], index=prices.index)
        benchmark_curve = initial_capital * (1 + returns).cumprod()
        
        # Derived Strategy Returns
        strat_returns = equity_curve_series.pct_change().fillna(0)
                
        return {
            'equity_curve': equity_curve_series,
            'benchmark_curve': benchmark_curve,
            'trades': pd.DataFrame(trades),
            'returns': strat_returns
        }

    @staticmethod
    def calculate_metrics(returns, risk_free_rate=0.0):
        """
        Calculates Sharpe, Sortino, Max Drawdown
        """
        if len(returns) < 2: return {}
        
        # Annualization factor
        ann_factor = 252
        
        # Excess Returns
        excess_ret = returns - (risk_free_rate / 252)
        
        # Sharpe
        sharpe = np.sqrt(ann_factor) * excess_ret.mean() / (returns.std() + 1e-9)
        
        # Sortino (Downside Deviation)
        downside = returns[returns < 0]
        sortino = np.sqrt(ann_factor) * excess_ret.mean() / (downside.std() + 1e-9)
        
        # Max Drawdown
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        
        # CAGR
        total_ret = (1 + returns).prod()
        n_years = len(returns) / 252
        cagr = (total_ret ** (1/n_years)) - 1 if n_years > 0 else 0
        
        return {
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'CAGR': cagr
        }

@st.cache_resource
def fit_regime_model(model_data, n_regimes, switch_vol, switch_trend):
    """
    Cached helper to fit Markov Regression.
    Returns the fitted result object.
    """
    try:
        mod_markov = MarkovRegression(
            model_data,
            k_regimes=n_regimes,
            trend='c',
            switching_variance=switch_vol,
            switching_trend=switch_trend
        )
        res_markov = mod_markov.fit(search_reps=50, disp=False)
        return res_markov
    except Exception as e:
        return None

@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Handle MultiIndex if present (common in new yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1, drop_level=True) if ticker in df.columns.get_level_values(1) else df
            # If structure is different (Ticker as top level)
            if ticker in df.columns:
                 df = df[ticker]
            # Fallback for simple single ticker download structure
            elif 'Close' in df.columns and len(df.columns) > 1 and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

        # Standard cleaning
        # Check if 'Close' exists, if not try 'Adj Close'
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        df['Returns'] = df['Close'].pct_change().dropna()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

def get_analyst_target(ticker):
    """Fetches 1-year price target from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        target = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('previousClose')
        
        if target and current:
            implied_return = np.log(target / current) # Log return for consistency
            return target, implied_return
        return None, None
    except:
        return None, None

def calculate_beta(ticker_returns, benchmark_ticker='SPY', lookback_years=2):
    """Calculates Beta against a benchmark."""
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_years*365)
        bench = yf.download(benchmark_ticker, start=start, end=end, progress=False)
        
        # Handle MultiIndex for Benchmark
        if isinstance(bench.columns, pd.MultiIndex):
             if benchmark_ticker in bench.columns.get_level_values(1):
                 bench = bench.xs(benchmark_ticker, axis=1, level=1, drop_level=True)
             elif 'Close' in bench.columns:
                 bench.columns = bench.columns.droplevel(1)

        if 'Close' not in bench.columns and 'Adj Close' in bench.columns:
            bench['Close'] = bench['Adj Close']
            
        bench_ret = bench['Close'].pct_change().dropna()
        
        # Align data
        common_idx = ticker_returns.index.intersection(bench_ret.index)
        if len(common_idx) < 30: return 1.0 # Fallback
        
        y = ticker_returns.loc[common_idx]
        x = bench_ret.loc[common_idx]
        
        cov = np.cov(y, x)[0, 1]
        var = np.var(x)
        return cov / var
    except:
        return 1.0 # Fallback to market beta

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Thesis Parameters")
    
    # Market Region Selector
    market_region = st.selectbox("Market Region", ["US Market (USD)", "Indian Market (INR)"])
    
    if market_region == "Indian Market (INR)":
        CURRENCY = "₹"
        BENCHMARK = "^NSEI"
        DEFAULT_RF = 7.0
        SUFFIX = ".NS"
    else:
        CURRENCY = "$"
        BENCHMARK = "SPY"
        DEFAULT_RF = 4.0
        SUFFIX = ""

    # Ticker Inputs
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        raw_ticker = st.text_input("Main Ticker", "RELIANCE" if market_region == "Indian Market (INR)" else "AAPL").upper()
    with col_t2:
        raw_pair = st.text_input("Pair Ticker", "").upper()
    
    # Auto-append suffix if needed
    TICKER = raw_ticker + SUFFIX if (SUFFIX and not raw_ticker.endswith(SUFFIX)) else raw_ticker
    PAIR_TICKER = raw_pair + SUFFIX if (SUFFIX and raw_pair and not raw_pair.endswith(SUFFIX)) else raw_pair
    
    st.caption(f"Active Ticker: {TICKER}")
    
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    
    st.subheader("Model Settings")
    rf_rate = st.number_input("Risk Free Rate (%)", 0.0, 20.0, DEFAULT_RF) / 100
    
    st.info(f"Benchmark: {BENCHMARK} | Currency: {CURRENCY}")

# ==========================================
# 4. DATA LOADING
# ==========================================
df_main = load_data(TICKER, start_date, end_date)

if df_main is not None:
    st.subheader(f"Data Analysis: {TICKER}")
    
    # Layout Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Volatility (GARCH)", 
        "Regime Switching", 
        "Stochastic (Heston/Jump)", 
        "Kalman Filter", 
        "Macro Factors",
        "Structural",
        "Backtest"
    ])

    # ==========================================
    # TAB 1: GARCH / EGARCH
    # ==========================================
    with tab1:
        st.write("### GARCH & EGARCH Volatility Modeling")
        if ARCH_AVAILABLE:
            returns_pct = df_main['Returns'] * 100 # Rescale for better optimization
            
            col1, col2 = st.columns(2)
            
            # GARCH(1,1)
            with col1:
                st.write("**GARCH(1,1) Model**")
                garch_model = arch_model(returns_pct, vol='Garch', p=1, q=1, dist='Normal')
                res_garch = garch_model.fit(disp='off')
                st.write(f"Alpha (Shock): {res_garch.params['alpha[1]']:.4f}")
                st.write(f"Beta (Persistence): {res_garch.params['beta[1]']:.4f}")
                
                fig_g, ax_g = plt.subplots(figsize=(10,4))
                ax_g.plot(res_garch.conditional_volatility, color='blue', label='Conditional Volatility')
                ax_g.set_title("GARCH(1,1) Conditional Volatility")
                format_plot_dates(ax_g, returns_pct.index) # Apply Date Formatting
                st.pyplot(fig_g)

            # GJR-GARCH(1,1) - Robust Leverage Effect
            with col2:
                st.write("**GJR-GARCH(1,1) Model**")
                # o=1 enables the asymmetric term (gamma)
                gjr_model = arch_model(returns_pct, vol='Garch', p=1, o=1, q=1, dist='Normal')
                res_gjr = gjr_model.fit(disp='off')
                
                st.write(f"Alpha (Shock): {res_gjr.params['alpha[1]']:.4f}")
                
                # Gamma is the leverage term in GJR-GARCH
                if 'gamma[1]' in res_gjr.params:
                    gamma_val = res_gjr.params['gamma[1]']
                    st.write(f"Gamma (Leverage): {gamma_val:.4f}")
                    if gamma_val > 0:
                        st.caption("Positive Gamma implies 'Leverage Effect': Volatility rises more when prices fall.")
                    else:
                        st.caption("Zero/Neg Gamma: No standard leverage effect.")
                else:
                    st.warning("Gamma not found.")

                fig_e, ax_e = plt.subplots(figsize=(10,4))
                ax_e.plot(res_gjr.conditional_volatility, color='green', label='Conditional Volatility')
                ax_e.set_title("GJR-GARCH(1,1) Conditional Volatility")
                format_plot_dates(ax_e, returns_pct.index) # Apply Date Formatting
                st.pyplot(fig_e)
        else:
            st.warning("Please install 'arch' library to view GARCH models.")

    # ==========================================
    # TAB 2: REGIME SWITCHING
    # ==========================================
    with tab2:
        st.write("### Markov Regime Switching Model")
        st.markdown("""
        Identifies hidden market states (e.g., Bull vs Bear) from return dynamics.  
        Each regime has distinct mean return and volatility characteristics.
        """)
        st.markdown("[Reference: Regime Switching Models (James D. Hamilton)](https://econweb.ucsd.edu/~jhamilton/palgrave.pdf)")
        
        # ===== CONFIGURATION =====
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            regime_freq = st.selectbox("Data Frequency", ["Daily", "Weekly"], index=0)
        with col_config2:
            lookback_years = st.slider("Lookback Period (Years)", 1, 10, 5)
        with col_config3:
            n_regimes = st.slider("Number of Regimes", 2, 4, 2)
        
        # New: Signal Stability Control
        stability = st.slider("Signal Stability (Pre-Smoothing)", 0, 10, 3, 
                              help="0 = Raw Data (Fastest), 10 = Very Smooth (Lagged). Higher values filter out noise.")
        
        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            switch_trend = st.checkbox("Switching Mean (Trend)", value=True,
                                        help="Uncheck if convergence fails")
        with col_sw2:
            switch_vol = st.checkbox("Switching Volatility", value=True,
                                      help="Uncheck to focus ONLY on Trend (ignore volatility changes)")
        
        # ===== PRE-FLIGHT CHECKS =====
        warnings = []
        if lookback_years <= 1:
            warnings.append("⚠️ Very short history - consider 3+ years for stable regimes")
            if regime_freq == "Weekly":
                warnings.append("❌ Cannot use Weekly with <1 year. Switch to Daily.")
                regime_freq = "Daily"
        
        if regime_freq == "Daily" and switch_trend and lookback_years < 3:
            warnings.append("⚠️ Daily + Switching Trend needs 3+ years. Disabling...")
            switch_trend = False
        
        if warnings:
            for w in warnings:
                st.warning(w)
        
        # ===== DATA PREPARATION =====
        start_dt_regime = datetime.now() - timedelta(days=lookback_years*365)
        df_regime = load_data(TICKER, start_dt_regime, end_date)
        
        if df_regime is None:
            st.error("Could not load data")
            st.stop()
        
        # Prepare data
        if regime_freq == "Weekly":
            returns = df_regime['Returns'].resample('W').sum()
        else:
            returns = df_regime['Returns']
        
        # Apply Pre-Smoothing (EWMA) if requested
        if stability > 0:
            returns = returns.ewm(span=stability, adjust=False).mean()
            st.caption(f"ℹ️ Applied EWMA Smoothing (Span={stability}) to reduce noise.")
        
        model_data = returns.dropna() * 100  # Scale to percentages
        
        st.caption(f"Modeling {len(model_data)} {regime_freq.lower()} returns from {start_dt_regime.date()}")
        
        # ===== MODEL FITTING =====
        try:
            mod_markov = MarkovRegression(
                model_data,
                k_regimes=n_regimes,
                trend='c',
                switching_variance=switch_vol,
                switching_trend=switch_trend
            )
            
            with st.spinner(f"Fitting {n_regimes}-regime model..."):
                res_markov = mod_markov.fit(search_reps=50, disp=False)
            
            # ===== CONVERGENCE CHECKS =====
            if not res_markov.mle_retvals['converged']:
                st.error("⛔ Model did not converge. Try longer history or simpler model.")
                st.stop()
            
            trans_matrix = res_markov.regime_transition
            
            # Check for degenerate regimes
            if np.any(trans_matrix > 0.99):
                st.warning("⚠️ Near-permanent regimes detected - consider fewer regimes")
            
            # ===== REGIME CHARACTERIZATION =====
            regime_stats = []
            for i in range(n_regimes):
                # Handle case where switching_trend=False (single 'const')
                if f'const[{i}]' in res_markov.params:
                    mean_val = res_markov.params[f'const[{i}]']
                else:
                    mean_val = res_markov.params.get('const', 0.0)
                
                # Handle case where switching_variance=False (single 'sigma2')
                if f'sigma2[{i}]' in res_markov.params:
                    vol_val = np.sqrt(res_markov.params[f'sigma2[{i}]'])
                else:
                    vol_val = np.sqrt(res_markov.params.get('sigma2', 1.0))
                    
                regime_stats.append({
                    'regime': i,
                    'mean': float(mean_val),
                    'vol': float(vol_val),
                    'persistence': float(trans_matrix[i, i])
                })
            
            # Sort by mean (high to low)
            regime_stats = sorted(regime_stats, key=lambda x: x['mean'], reverse=True)
            
            # ===== DISPLAY REGIMES =====
            st.write("### 📊 Identified Regimes")
            
            cols = st.columns(n_regimes)
            labels = ['🟢 Bull', '🟡 Normal', '🔴 Bear', '⚫ Crisis']
            
            for idx, (col, regime) in enumerate(zip(cols, regime_stats)):
                with col:
                    st.markdown(f"**{labels[idx]} (Regime {regime['regime']})**")
                    st.metric("Mean Return", f"{regime['mean']:.2f}%")
                    st.metric("Volatility", f"{regime['vol']:.2f}%")
                    st.metric("Persistence", f"{regime['persistence']:.1%}")
                    
                    avg_duration = 1 / (1 - regime['persistence'] + 1e-10)
                    st.caption(f"Avg duration: {avg_duration:.1f} {regime_freq.lower()} periods")
            
            # ===== CURRENT STATE =====
            # Use .iloc[-1] to get the probabilities at the LAST time step
            last_probs = res_markov.filtered_marginal_probabilities.iloc[-1]
            current_regime = np.argmax(last_probs)
            current_prob = last_probs.iloc[current_regime]
            
            regime_label = labels[[r['regime'] for r in regime_stats].index(current_regime)]
            
            st.info(f"""
            **📍 Current Market State:** {regime_label} (Regime {current_regime})  
            **Confidence:** {current_prob:.1%}  
            **As of:** {model_data.index[-1].date()}
            """)
            
            # ===== VISUALIZATION =====
            st.write("### 📈 Regime Analysis (Real-time / Filtered)")
            
            fig_m, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # Plot 1: Returns with regime shading
            axes[0].plot(model_data.index, model_data, color='black', alpha=0.6, linewidth=1)
            
            for i, regime in enumerate(regime_stats):
                # Use .iloc for robust column access
                # CHANGED: Use filtered probabilities to avoid look-ahead bias
                probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
                # Invert color map: i=0 (Bull) -> 1.0 (Green), i=N (Bear) -> 0.0 (Red)
                color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
                color = plt.cm.RdYlGn(color_idx)
                
                axes[0].fill_between(model_data.index, model_data.min(), model_data.max(),
                                      where=(probs > 0.6),
                                      alpha=0.15, color=color, label=labels[i])
            
            axes[0].set_title(f"{TICKER} Returns with Regime Periods")
            axes[0].legend(loc='upper left')
            axes[0].set_ylabel("Return (%)")
            
            # Plot 2: Probabilities
            # Option to smooth the probability line itself for readability
            smooth_probs = st.checkbox("Smooth Probabilities (4-period Rolling)", value=True, key="smooth_probs_check")
            
            for i, regime in enumerate(regime_stats):
                # Invert color map: i=0 (Bull) -> 1.0 (Green), i=N (Bear) -> 0.0 (Red)
                color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
                color = plt.cm.RdYlGn(color_idx)
                
                # Get raw probabilities
                # CHANGED: Use filtered probabilities
                raw_probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
                
                # Apply smoothing if requested
                if smooth_probs:
                    plot_probs = raw_probs.rolling(window=4, min_periods=1).mean()
                else:
                    plot_probs = raw_probs
                
                # Use fill_between (Area Chart) for better readability than just lines
                axes[1].fill_between(model_data.index, 0, plot_probs, 
                                     color=color, alpha=0.3, label=labels[i])
                axes[1].plot(model_data.index, plot_probs, color=color, linewidth=1.5)

            axes[1].axhline(1/n_regimes, color='gray', linestyle='--', alpha=0.4, 
                            label='Equal probability')
            axes[1].set_title("Regime Probabilities (Filtered/Real-time)")
            axes[1].set_ylabel("Probability")
            axes[1].set_ylim([0, 1])
            axes[1].legend()
            
            # Plot 3: Expected Return
            # Plot 3: Expected Return
            # Helper to safely get const
            def get_const(i):
                # Check for regime specific const first, then global const
                if f'const[{i}]' in res_markov.params:
                    return float(res_markov.params[f'const[{i}]'])
                return float(res_markov.params.get('const', 0.0))

            # Initialize expected_ret as a Series with the correct index
            expected_ret = pd.Series(0.0, index=model_data.index)
            
            for i in range(n_regimes):
                # CHANGED: Use filtered probabilities
                prob = res_markov.filtered_marginal_probabilities.iloc[:, i]
                const_val = get_const(i)
                expected_ret += prob * const_val
            
            axes[2].plot(model_data.index, expected_ret, color='darkblue', linewidth=2)
            axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
            
            # Fill between requires numpy arrays for 'where' sometimes, or robust Series
            axes[2].fill_between(model_data.index, 0, expected_ret,
                                  where=(expected_ret > 0), color='green', alpha=0.3)
            axes[2].fill_between(model_data.index, 0, expected_ret,
                                  where=(expected_ret < 0), color='red', alpha=0.3)
            axes[2].set_title("Regime-Weighted Expected Return")
            axes[2].set_ylabel("Expected Return (%)")
            
            # Format dates for ALL axes and ensure labels are visible
            # Format dates: Only for the last axis to avoid overlap
            format_plot_dates(axes[-1], model_data.index)
            axes[-1].tick_params(labelbottom=True)
            
            # Ensure other axes don't show labels (redundant with sharex but safe)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)
                
            st.pyplot(fig_m)
            
            # ===== PARAMETERS TABLE =====
            with st.expander("📋 Technical Parameters"):
                summary_data = {
                    "Parameter": res_markov.params.index,
                    "Value": res_markov.params.values.astype(float),
                    "Std Error": res_markov.bse.values.astype(float),
                    "P-Value": res_markov.pvalues.values.astype(float)
                }
                df_summary = pd.DataFrame(summary_data)
                # Format only numeric columns to avoid error with "Parameter" string column
                st.dataframe(df_summary.style.format({
                    "Value": "{:.4f}",
                    "Std Error": "{:.4f}",
                    "P-Value": "{:.4f}"
                }))
                
                st.caption("AIC: {:.2f} | BIC: {:.2f}".format(res_markov.aic, res_markov.bic))
            
        except Exception as e:
            st.error(f"❌ Model fitting failed: {str(e)}")
            
            st.write("**💡 Troubleshooting:**")
            st.info("""
            - Try **longer history** (5+ years recommended)
            - **Disable 'Switching Mean'** (simpler model)
            - Use **Weekly data** instead of Daily
            - Reduce **number of regimes** to 2
            - Try a **different ticker** (less volatile assets work better)
            """)

    # ==========================================
    # TAB 3: STOCHASTIC MODELS (Heston/Jump)
    # ==========================================
    with tab3:
        st.write("### Advanced Stochastic Simulations")
        
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            sim_type = st.radio("Select Model", ["Merton Jump Diffusion", "Heston Stochastic Volatility"])
        with col_conf2:
            drift_type = st.radio("Drift Strategy", [
                "Risk-Neutral (Risk-Free Rate)", 
                "Historical Mean (Real World)",
                "CAPM (Expected Return)",
                "Analyst Consensus (1Y Target)",
                "Custom View"
            ])
            
        # Determine Drift
        if drift_type == "Risk-Neutral (Risk-Free Rate)":
            mu_drift = rf_rate
            st.caption(f"Using Risk-Free Rate: {rf_rate*100:.2f}% (Standard for Pricing)")
            
        elif drift_type == "Historical Mean (Real World)":
            hist_mu = df_main['Log_Returns'].mean() * 252
            mu_drift = hist_mu
            st.caption(f"Using Historical Mean: {hist_mu*100:.2f}% (Past Performance)")
            
        elif drift_type == "CAPM (Expected Return)":
            beta = calculate_beta(df_main['Returns'], benchmark_ticker=BENCHMARK)
            mkt_return = 0.08 # Assumed 8% market return
            capm_ret = rf_rate + beta * (mkt_return - rf_rate)
            # Convert simple return to log return approx
            mu_drift = np.log(1 + capm_ret)
            st.metric("CAPM Beta", f"{beta:.2f}")
            st.caption(f"CAPM Expected Return: {capm_ret*100:.2f}% (Beta: {beta:.2f} vs {BENCHMARK})")
            
        elif drift_type == "Analyst Consensus (1Y Target)":
            target, implied_ret = get_analyst_target(TICKER)
            if target:
                mu_drift = implied_ret
                st.metric("Analyst Target", f"{CURRENCY}{target:.2f}")
                st.caption(f"Implied Drift: {implied_ret*100:.2f}% (from Consensus)")
            else:
                st.warning("No Analyst Target found. Reverting to Historical.")
                mu_drift = df_main['Log_Returns'].mean() * 252
                
        elif drift_type == "Custom View":
            custom_ret = st.number_input("Expected Annual Return (%)", -50.0, 100.0, 10.0) / 100
            mu_drift = np.log(1 + custom_ret)
            st.caption(f"Using Custom Drift: {custom_ret*100:.2f}%")

        # Helper to generate future dates
        last_date = df_main.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(253)] # 0 to 252
        
        import plotly.graph_objects as go
        
        # Random Seed for Reproducibility
        seed = st.number_input("Random Seed (Fixes the simulation)", 1, 10000, 42)
        np.random.seed(seed)
        
        if sim_type == "Merton Jump Diffusion":
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Parameters**")
                mj_lam = st.slider("Jump Intensity (Lambda)", 0.1, 10.0, 1.0, help="Avg jumps per year")
                mj_mu = st.slider("Jump Mean Size", -0.5, 0.5, -0.1)
                mj_sigma = st.slider("Jump Std Dev", 0.01, 0.5, 0.1)
                mj_vol = st.slider("Diffusive Volatility", 0.05, 1.0, 0.2)
            
            with col2:
                current_price = df_main['Close'].iloc[-1]
                # Pass mu_drift instead of rf_rate
                paths = merton_jump_diffusion(current_price, 1.0, mu_drift, mj_vol, mj_lam, mj_mu, mj_sigma, 252, 50)
                
                # Calculate Statistics
                mean_path = paths.mean(axis=1)
                median_path = np.median(paths, axis=1)
                p05_path = np.percentile(paths, 5, axis=1)
                p95_path = np.percentile(paths, 95, axis=1)
                
                final_mean = mean_path[-1]
                final_median = median_path[-1]
                
                m1, m2 = st.columns(2)
                m1.metric("Projected Mean (Avg)", f"{CURRENCY}{final_mean:,.2f}")
                m2.metric("Projected Median (50th %)", f"{CURRENCY}{final_median:,.2f}")
                
                # Plotly Chart
                fig = go.Figure()
                
                # Add Cone of Uncertainty (5th-95th)
                fig.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=np.concatenate([p95_path, p05_path[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='90% Confidence Interval',
                    showlegend=True
                ))
                
                # Add individual paths (lightly) - Reduced count for clarity
                for i in range(min(20, paths.shape[1])):
                    fig.add_trace(go.Scatter(
                        x=future_dates, 
                        y=paths[:, i], 
                        mode='lines', 
                        line=dict(color='rgba(100, 100, 255, 0.05)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                # Add Mean Path
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=mean_path, 
                    mode='lines', 
                    name='Mean Path',
                    line=dict(color='orange', width=3),
                    hovertemplate=f'Mean: {CURRENCY}%{{y:.2f}}'
                ))
                
                # Add Median Path
                fig.add_trace(go.Scatter(
                    x=future_dates, 
                    y=median_path, 
                    mode='lines', 
                    name='Median Path',
                    line=dict(color='white', width=3, dash='dash'),
                    hovertemplate=f'Median: {CURRENCY}%{{y:.2f}}'
                ))
                
                fig.update_layout(
                    title=f"Merton Jump Diffusion: 1 Year Projection ({TICKER})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

        elif sim_type == "Heston Stochastic Volatility":
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Heston Params**")
                
                # Initialize Session State for Heston Params if not present
                if 'h_kappa' not in st.session_state: st.session_state.h_kappa = 2.0
                if 'h_theta' not in st.session_state: st.session_state.h_theta = 0.04
                if 'h_xi' not in st.session_state: st.session_state.h_xi = 0.3
                if 'h_rho' not in st.session_state: st.session_state.h_rho = -0.7
                if 'h_v0' not in st.session_state: st.session_state.h_v0 = 0.04

                # Calibration Button
                st.caption("Methodology: Historical Proxy Calibration (GARCH-based)")
                if st.button("Calibrate from History (Proxy)"):
                    with st.spinner("Calibrating Heston Model..."):
                        try:
                            calib_res = Calibrator.calibrate_heston(df_main['Log_Returns'])
                            st.session_state.h_kappa = float(calib_res['kappa'])
                            st.session_state.h_theta = float(calib_res['theta'])
                            st.session_state.h_xi = float(calib_res['xi'])
                            st.session_state.h_rho = float(calib_res['rho'])
                            st.session_state.h_v0 = float(calib_res['v0'])
                            st.success("Calibration Successful!")
                        except Exception as e:
                            st.error(f"Calibration Failed: {e}")

                # Sliders using Session State Keys Directly
                # Sliders using Session State Keys Directly
                # Increased max values to accommodate calibration results
                h_kappa = st.number_input("Kappa (Mean Rev Speed)", 0.01, 1000.0, key='h_kappa', format="%.4f")
                h_theta = st.number_input("Theta (Long Term Vol)", 0.0, 5.0, key='h_theta', format="%.6f")
                h_xi = st.number_input("Xi (Vol of Vol)", 0.01, 100.0, key='h_xi', format="%.4f")
                h_rho = st.slider("Rho (Correlation)", -0.99, 0.99, key='h_rho')
                h_v0 = st.number_input("Initial Variance", 0.0, 5.0, key='h_v0', format="%.6f")

                # Session state is automatically updated by the widgets via keys

            with col2:
                current_price = df_main['Close'].iloc[-1]
                # Pass mu_drift instead of rf_rate
                sim_prices, sim_vols = simulate_heston(current_price, 1.0, mu_drift, h_kappa, h_theta, h_xi, h_rho, h_v0, 252, 50)
                
                # Calculate Statistics
                mean_path = sim_prices.mean(axis=1)
                median_path = np.median(sim_prices, axis=1)
                p05_path = np.percentile(sim_prices, 5, axis=1)
                p95_path = np.percentile(sim_prices, 95, axis=1)
                
                final_mean = mean_path[-1]
                final_median = median_path[-1]
                
                m1, m2 = st.columns(2)
                m1.metric("Projected Mean (Avg)", f"{CURRENCY}{final_mean:,.2f}")
                m2.metric("Projected Median (50th %)", f"{CURRENCY}{final_median:,.2f}")
                
                # Plotly Chart for Prices
                fig_h = go.Figure()
                
                # Add Cone of Uncertainty (5th-95th)
                fig_h.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=np.concatenate([p95_path, p05_path[::-1]]),
                    fill='toself',
                    fillcolor='rgba(100, 100, 255, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='90% Confidence Interval',
                    showlegend=True
                ))
                
                # Add individual paths
                for i in range(min(20, sim_prices.shape[1])):
                    fig_h.add_trace(go.Scatter(
                        x=future_dates, 
                        y=sim_prices[:, i], 
                        mode='lines', 
                        line=dict(color='rgba(100, 100, 255, 0.05)', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                # Add Mean Path
                fig_h.add_trace(go.Scatter(
                    x=future_dates, 
                    y=mean_path, 
                    mode='lines', 
                    name='Mean Path',
                    line=dict(color='orange', width=3),
                    hovertemplate=f'Mean: {CURRENCY}%{{y:.2f}}'
                ))
                
                # Add Median Path
                fig_h.add_trace(go.Scatter(
                    x=future_dates, 
                    y=median_path, 
                    mode='lines', 
                    name='Median Path',
                    line=dict(color='white', width=3, dash='dash'),
                    hovertemplate=f'Median: {CURRENCY}%{{y:.2f}}'
                ))
                
                fig_h.update_layout(
                    title=f"Heston Price Paths ({TICKER})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    template="plotly_dark",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_h, use_container_width=True)
                
                # Volatility Plot (Optional, keep simple or upgrade too)
                st.write("**Stochastic Volatility Paths**")
                fig_v = go.Figure()
                for i in range(min(20, sim_vols.shape[1])):
                     fig_v.add_trace(go.Scatter(
                        x=future_dates, 
                        y=np.sqrt(sim_vols[:, i]), 
                        mode='lines', 
                        line=dict(color='rgba(255, 165, 0, 0.3)', width=1),
                        showlegend=False
                    ))
                fig_v.update_layout(
                    title="Volatility Process (Sigma)",
                    xaxis_title="Date",
                    yaxis_title="Volatility",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig_v, use_container_width=True)

    # ==========================================
    # TAB 4: KALMAN FILTER
    # ==========================================
    with tab4:
        st.write("### Kalman Filter Analysis")
        
        kf_mode = st.radio("Analysis Mode", ["Pairs Trading (Relative Value)", "Single Asset (Trend)"])
        
        if kf_mode == "Pairs Trading (Relative Value)":
            st.write(f"**{TICKER} vs {PAIR_TICKER}**")
            df_pair = load_data(PAIR_TICKER, start_date, end_date)
            
            if df_pair is not None:
                # Align data
                common_idx = df_main.index.intersection(df_pair.index)
                y = df_main.loc[common_idx, 'Close'].values
                x = df_pair.loc[common_idx, 'Close'].values
                
                if len(y) > 10:
                    kf = KalmanFilterReg(delta=1e-4, R=1e-3)
                    state_means, state_covs = kf.run_filter(y, x)
                    
                    alpha = state_means[:, 0]
                    beta = state_means[:, 1]
                    
                    # Plot Beta
                    fig_k, (ax_k1, ax_k2) = plt.subplots(2, 1, figsize=(12,8), sharex=True)
                    
                    dates = common_idx
                    ax_k1.plot(dates, beta, color='darkblue', label=f"Dynamic Beta ({TICKER}/{PAIR_TICKER})")
                    ax_k1.set_title("Kalman Estimated Hedge Ratio (Beta)")
                    ax_k1.legend()
                    format_plot_dates(ax_k1, dates) # Apply Date Formatting
                    
                    # Spread Analysis
                    spread_series = y - (alpha + beta * x)
                    z_score = (spread_series - np.mean(spread_series)) / np.std(spread_series)
                    
                    ax_k2.plot(dates, z_score, color='purple', label="Spread Z-Score")
                    ax_k2.axhline(2.0, color='red', linestyle='--')
                    ax_k2.axhline(-2.0, color='green', linestyle='--')
                    ax_k2.set_title("Kalman Residual Z-Score (Mean Reversion Signal)")
                    ax_k2.legend()
                    format_plot_dates(ax_k2, dates) # Apply Date Formatting
                    
                    st.pyplot(fig_k)
                    st.write(f"Current Hedge Ratio: **{beta[-1]:.4f}** (Long 1 {TICKER}, Short {beta[-1]:.4f} {PAIR_TICKER})")
                else:
                    st.error("Not enough overlapping data for pairs analysis.")
            else:
                st.error(f"Could not load data for {PAIR_TICKER}")
                
        elif kf_mode == "Single Asset (Trend)":
            st.write(f"**{TICKER} Trend Detection**")
            st.caption("Uses a Kalman Filter (Local Level Model) to separate the 'True' Price Trend from Market Noise.")
            st.markdown("[Reference: Time Series Analysis by State Space Methods (Durbin & Koopman)](https://global.oup.com/academic/product/time-series-analysis-by-state-space-methods-9780199641178)")
            
            # Parameters
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                proc_noise = st.select_slider("Trend Flexibility (Process Noise)", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
            with col_k2:
                meas_noise = st.select_slider("Noise Tolerance (Measurement Noise)", options=[1e-3, 1e-2, 1e-1, 1.0], value=1e-2)
            with col_k3:
                model_mode = st.radio("Model Type", ["Smoothed (New)", "Standard (Old)", "Compare Both"])
            
            prices = df_main['Close'].values
            kf_trend = KalmanFilterTrend(process_noise=proc_noise, measurement_noise=meas_noise)
            
            # Calculate based on mode
            if model_mode == "Standard (Old)":
                est_trend, _ = kf_trend.filter(prices)
                label_trend = "Kalman Trend (Standard)"
                color_trend = "blue"
            elif model_mode == "Smoothed (New)":
                est_trend, _ = kf_trend.smooth(prices)
                label_trend = "Kalman Trend (Smoothed)"
                color_trend = "purple"
            else: # Compare Both
                est_trend_smooth, _ = kf_trend.smooth(prices)
                est_trend_std, _ = kf_trend.filter(prices)
            
            # Plot
            fig_kt, ax_kt = plt.subplots(figsize=(12, 6))
            ax_kt.plot(df_main.index, prices, color='gray', alpha=0.5, label='Actual Price')
            
            if model_mode == "Compare Both":
                ax_kt.plot(df_main.index, est_trend_std, color='blue', linewidth=1.5, linestyle='--', label='Standard (Causal)')
                ax_kt.plot(df_main.index, est_trend_smooth, color='purple', linewidth=2, label='Smoothed (RTS)')
                current_trend = est_trend_smooth[-1] # Use smooth for metrics
            else:
                ax_kt.plot(df_main.index, est_trend, color=color_trend, linewidth=2, label=label_trend)
                current_trend = est_trend[-1]

            ax_kt.set_title(f"Kalman Filter Trend: {TICKER}")
            ax_kt.legend()
            format_plot_dates(ax_kt, df_main.index) # Apply Date Formatting
            
            # Improve Y-Axis Ticks (Equal Intervals)
            from matplotlib.ticker import MaxNLocator
            ax_kt.yaxis.set_major_locator(MaxNLocator(nbins=15)) # Force more granular ticks
            ax_kt.grid(True, which='major', linestyle='--', alpha=0.5)
            
            st.pyplot(fig_kt)
            
            # Signal & Metrics
            current_price = prices[-1]
            diff_pct = (current_price - current_trend) / current_trend * 100
            
            # Display Current Values
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Price", f"{CURRENCY}{current_price:.2f}")
            with c2:
                st.metric("Current Trend", f"{CURRENCY}{current_trend:.2f}")
            with c3:
                st.metric("Deviation", f"{diff_pct:.2f}%", delta=f"{diff_pct:.2f}%", delta_color="inverse")

            if diff_pct > 5.0:
                st.warning("Price significantly ABOVE Trend (Potential Overbought)")
            elif diff_pct < -5.0:
                st.success("Price significantly BELOW Trend (Potential Oversold)")
            else:
                st.info("Price near Trend (Neutral)")

    # ==========================================
    # TAB 5: MACRO FACTORS
    # ==========================================
    with tab5:
        st.write("### Macro Factor Sensitivity")
        st.markdown("Correlation of returns against key structural drivers.")
        
        macro_tickers = {
            'Crude Oil': 'CL=F',
            'Gold': 'GC=F',
            '10Y Yield': '^TNX',
            'US Dollar': 'DX-Y.NYB',
            'S&P 500': '^GSPC'
        }
        
        macro_data = {}
        for name, sym in macro_tickers.items():
            m_df = load_data(sym, start_date, end_date)
            if m_df is not None:
                macro_data[name] = m_df['Returns']
        
        # Add main ticker
        macro_data[TICKER] = df_main['Returns']
        
        df_macro = pd.DataFrame(macro_data).dropna()
        
        if not df_macro.empty:
            corr_matrix = df_macro.corr()
            
            # Simple Heatmap using matplotlib
            fig_hm, ax_hm = plt.subplots(figsize=(8,6))
            cax = ax_hm.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            fig_hm.colorbar(cax)
            
            ticks = np.arange(len(corr_matrix.columns))
            ax_hm.set_xticks(ticks)
            ax_hm.set_yticks(ticks)
            ax_hm.set_xticklabels(corr_matrix.columns, rotation=45)
            ax_hm.set_yticklabels(corr_matrix.columns)
            
            for (i, j), z in np.ndenumerate(corr_matrix):
                ax_hm.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
                
            ax_hm.set_title("Asset Class Correlations")
            st.pyplot(fig_hm)
            
            st.write(f"**Structural Thesis Check:**")
            oil_corr = corr_matrix.loc[TICKER, 'Crude Oil']
            rate_corr = corr_matrix.loc[TICKER, '10Y Yield']
            
            if oil_corr > 0.3:
                st.success(f"High correlation with Energy ({oil_corr:.2f}). Commodity cycle model relevant.")
            elif oil_corr < -0.3:
                st.info(f"Inverse correlation with Energy ({oil_corr:.2f}).")
            else:
                st.warning(f"Low sensitivity to Energy prices ({oil_corr:.2f}).")

    # ==========================================
    # TAB 6: STRUCTURAL
    # ==========================================
    with tab6:
        st.write("### Structural Decomposition")
        # Need freq for decomposition. 
        # Business days ~ 5 (weekly), 21 (monthly), 252 (yearly)
        period = st.selectbox("Seasonality Period", [5, 21, 63, 252], index=1)
        
        if len(df_main) > period * 2:
            decomp = seasonal_decompose(df_main['Close'], model='multiplicative', period=period)
            
            fig_dec, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            
            decomp.trend.plot(ax=ax1, title='Trend')
            decomp.seasonal.plot(ax=ax2, title='Seasonal Component')
            decomp.resid.plot(ax=ax3, title='Residuals')
            
            format_plot_dates(ax3, df_main.index) # Apply Date Formatting to the shared x-axis
            
            st.pyplot(fig_dec)
        else:
            st.warning("Insufficient data for decomposition with selected period.")

    # ==========================================
    # TAB 7: BACKTEST
    # ==========================================
    with tab7:
        st.write("### 🛠️ Strategy Backtest")
        
        # Strategy Selector
        strategy_type = st.radio("Select Strategy", ["Regime Switching (Trend Following)", "Kalman Filter (Trend Crossover)"], horizontal=True)
        
        # Common Backtest Params
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            trailing_stop = st.slider("Trailing Stop Loss (%)", 0.0, 20.0, 0.0, step=0.5) / 100
        with col_b2:
            initial_cap = st.number_input("Initial Capital", 1000, 1000000, 10000)
        
        # Date Selection
        with col_b3:
            default_start = datetime.now() - timedelta(days=365)
            bt_start_date = st.date_input("Backtest Start", default_start)
            bt_end_date = st.date_input("Backtest End", datetime.now())

        # Data Prep
        if bt_start_date >= bt_end_date:
            st.error("Start date must be before end date.")
            st.stop()
            
        df_bt = load_data(TICKER, bt_start_date, bt_end_date)
        
        if df_bt is None or df_bt.empty:
            st.error("Could not load data for backtest. Check dates and ticker.")
            st.stop()
            
        returns_bt = df_bt['Returns']
        prices_bt = df_bt['Close']
        model_data_bt = returns_bt.dropna() * 100
        
        signals = None
        strat_prices = prices_bt
        
        if strategy_type == "Regime Switching (Trend Following)":
            
            # Regime Parameters
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                bt_n_regimes = st.slider("Number of Regimes", 2, 4, 2, key="bt_n_regimes")
            with col_r2:
                bt_stability = st.slider("Signal Stability (Smoothing)", 0, 10, 4, key="bt_stability")
            with col_r3:
                bt_freq = st.selectbox("Frequency", ["Weekly", "Daily"], key="bt_freq")
            
            col_r4, col_r5 = st.columns(2)
            with col_r4:
                bt_switch_trend = st.checkbox("Switching Mean", value=True, key="bt_switch_trend")
            with col_r5:
                bt_switch_vol = st.checkbox("Switching Volatility", value=True, key="bt_switch_vol")

            # Signal Method Selection
            signal_method = st.radio("Signal Method", ["Regime Weighted Expected Return", "Regime Probability", "Regime Switching Period"], horizontal=True)

            if signal_method == "Regime Weighted Expected Return":
                st.markdown("**Strategy:** Long when **Expected Return > 0**. Sell when **Expected Return < 0**.")
            elif signal_method == "Regime Probability":
                st.markdown("**Strategy:** Long when **Bull Probability** crosses above others (Dominant Regime). Sell otherwise.")
            else:
                st.markdown("**Strategy:** Long immediately when entering **Bull Regime**. Sell immediately when exiting.")

            # Resample if Weekly
            if bt_freq == "Weekly":
                # Resample Prices to Weekly (Last Close)
                prices_bt_resampled = prices_bt.resample('W').last().dropna()
                # Recalculate Returns from resampled prices
                returns_bt_resampled = prices_bt_resampled.pct_change().dropna()
                strat_prices = prices_bt_resampled
            else:
                prices_bt_resampled = prices_bt
                returns_bt_resampled = returns_bt

            # Apply Smoothing if requested
            if bt_stability > 0:
                model_data_bt = returns_bt_resampled.ewm(span=bt_stability, adjust=False).mean().dropna() * 100
            else:
                model_data_bt = returns_bt_resampled.dropna() * 100

            with st.spinner("Fitting Regime Model..."):
                # Fit Model
                res_bt = fit_regime_model(model_data_bt, bt_n_regimes, bt_switch_vol, bt_switch_trend)
                
                if res_bt:
                    # Identify Regimes (Sort by Mean)
                    regime_means = []
                    for i in range(bt_n_regimes):
                        if f'const[{i}]' in res_bt.params:
                            mean_val = res_bt.params[f'const[{i}]']
                        else:
                            mean_val = res_bt.params.get('const', 0.0)
                        regime_means.append((i, mean_val))
                    
                    # Sort regimes by mean return (High to Low) -> Index 0 is Bull
                    sorted_regimes = sorted(regime_means, key=lambda x: x[1], reverse=True)
                    bull_regime_idx = sorted_regimes[0][0]
                    
                    # Get Filtered Probabilities
                    probs_df = res_bt.filtered_marginal_probabilities
                    
                    if signal_method == "Regime Weighted Expected Return":
                        # Calculate Expected Return
                        expected_ret = pd.Series(0.0, index=model_data_bt.index)
                        for i in range(bt_n_regimes):
                            # Get Regime Mean
                            if f'const[{i}]' in res_bt.params:
                                mean_val = res_bt.params[f'const[{i}]']
                            else:
                                mean_val = res_bt.params.get('const', 0.0)
                                
                            # Get Filtered Probability
                            prob = probs_df.iloc[:, i]
                            expected_ret += prob * mean_val
                        
                        # Align indices
                        common_idx = strat_prices.index.intersection(expected_ret.index)
                        expected_ret = expected_ret.loc[common_idx]
                        
                        # Generate Signals (1 = Long if Exp Ret > 0, else 0)
                        signals = (expected_ret > 0).astype(int)
                        
                        # Plot Context
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            ax_ctx.plot(expected_ret.index, expected_ret, color='purple', label='Expected Return')
                            ax_ctx.axhline(0, color='black', linestyle='--', linewidth=1)
                            ax_ctx.fill_between(expected_ret.index, 0, expected_ret, where=(expected_ret>0), color='green', alpha=0.3, label='Long Zone')
                            ax_ctx.fill_between(expected_ret.index, 0, expected_ret, where=(expected_ret<0), color='red', alpha=0.3, label='Cash/Short Zone')
                            format_plot_dates(ax_ctx, expected_ret.index)
                            ax_ctx.set_title("Regime-Weighted Expected Return")
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    elif signal_method == "Regime Probability":
                        # Logic: Long if Bull Probability is the highest (Dominant)
                        # Or specifically: Bull > Bear (and Bull > Normal)
                        
                        bull_probs = probs_df.iloc[:, bull_regime_idx]
                        
                        # Determine if Bull is dominant
                        # We can just check if argmax is the bull index
                        dominant_regime = probs_df.idxmax(axis=1) # Returns column name (0, 1, etc)
                        
                        # Align indices
                        common_idx = strat_prices.index.intersection(dominant_regime.index)
                        dominant_regime = dominant_regime.loc[common_idx]
                        bull_probs = bull_probs.loc[common_idx]
                        
                        # Signal: 1 if Dominant Regime is Bull, else 0
                        signals = (dominant_regime == bull_regime_idx).astype(int)
                        
                        # Plot Context
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            
                            # Plot Bull Probability
                            ax_ctx.plot(bull_probs.index, bull_probs, color='green', label='Bull Probability')
                            
                            # Plot Others
                            for r_idx, r_mean in sorted_regimes:
                                if r_idx != bull_regime_idx:
                                    other_probs = probs_df.iloc[:, r_idx].loc[common_idx]
                                    ax_ctx.plot(other_probs.index, other_probs, linestyle='--', alpha=0.6, label=f'Regime {r_idx} Prob')
                            
                            # Highlight Long Zones
                            ax_ctx.fill_between(bull_probs.index, 0, 1, where=(signals==1), color='green', alpha=0.1, label='Long Signal')
                            
                            format_plot_dates(ax_ctx, bull_probs.index)
                            ax_ctx.set_title(f"Regime Probability Crossover (Bull Regime: {bull_regime_idx})")
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    else: # Regime Switching Period
                        # Logic: Long if Bull Probability is the highest (Dominant)
                        # This is similar to 'Regime Probability' but framed as "Period"
                        # We ensure it's strictly 1 or 0 based on dominance.
                        
                        dominant_regime = probs_df.idxmax(axis=1)
                        
                        # Align indices
                        common_idx = strat_prices.index.intersection(dominant_regime.index)
                        dominant_regime = dominant_regime.loc[common_idx]
                        
                        # Signal: 1 if Dominant Regime is Bull
                        signals = (dominant_regime == bull_regime_idx).astype(int)
                        
                        # Plot Context
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            
                            # Plot Price
                            ax_ctx.plot(strat_prices.index, strat_prices, color='gray', alpha=0.5, label='Price')
                            
                            # Highlight Bull Periods
                            ax_ctx.fill_between(strat_prices.index, strat_prices.min(), strat_prices.max(), 
                                                where=(signals==1), color='green', alpha=0.2, label='Bull Regime Period')
                            
                            format_plot_dates(ax_ctx, strat_prices.index)
                            ax_ctx.set_title(f"Regime Switching Periods (Bull Regime: {bull_regime_idx})")
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    # Debug Dataframe
                    with st.expander("🔍 Debug: Signal Details"):
                        debug_df = pd.DataFrame({
                            "Price": strat_prices,
                            "Signal": signals
                        }).dropna()
                        st.dataframe(debug_df.style.format({
                            "Price": "{:.2f}",
                            "Signal": "{:.0f}"
                        }), use_container_width=True)
                        
                else:
                    st.error("Regime model fitting failed.")

        elif strategy_type == "Kalman Filter (Trend Crossover)":
            st.markdown("**Strategy:** Long when Price crosses **ABOVE** Kalman Trend. Sell when Price crosses **BELOW**.")
            
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                kf_noise = st.select_slider("Trend Sensitivity", options=[1e-5, 1e-4, 1e-3], value=1e-4, 
                                            format_func=lambda x: f"{x} (Standard)" if x==1e-4 else str(x))
            with col_k2:
                confirm_days = st.slider("Signal Confirmation (Days)", 1, 5, 1, help="Consecutive days required to confirm a trend change.")

            with st.spinner("Running Kalman Filter..."):
                # Run Kalman Filter (Standard/Causal to avoid lookahead)
                kf = KalmanFilterTrend(process_noise=kf_noise, measurement_noise=1e-2)
                trend_est, _ = kf.filter(prices_bt.values)
                trend_series = pd.Series(trend_est, index=prices_bt.index)
                
                # Generate Signals with Confirmation Logic
                sig_list = []
                position = 0
                
                # Counters for consecutive days
                days_above = 0
                days_below = 0
                
                for price, trend in zip(prices_bt, trend_series):
                    if price > trend:
                        days_above += 1
                        days_below = 0
                    else:
                        days_below += 1
                        days_above = 0
                    
                    # Trading Logic
                    if position == 0:
                        if days_above >= confirm_days:
                            position = 1 # Buy
                    elif position == 1:
                        if days_below >= confirm_days:
                            position = 0 # Sell
                            
                    sig_list.append(position)
                
                signals = pd.Series(sig_list, index=prices_bt.index)
                
                # Plot Strategy Context
                with st.expander("See Strategy Context"):
                    fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                    ax_ctx.plot(prices_bt.index, prices_bt, color='gray', alpha=0.5, label='Price')
                    ax_ctx.plot(trend_series.index, trend_series, color='blue', label='Kalman Trend')
                    
                    # Highlight Long Zones
                    ax_ctx.fill_between(trend_series.index, prices_bt.min(), prices_bt.max(), 
                                        where=(signals==1), color='green', alpha=0.1, label='Long Zone')
                    
                    format_plot_dates(ax_ctx, prices_bt.index)
                    ax_ctx.legend()
                    st.pyplot(fig_ctx)

        # Run Backtest Engine if signals exist
        if signals is not None:
            bt_results = BacktestEngine.run_strategy(strat_prices, signals, initial_cap, trailing_stop)
            
            # Metrics
            strat_metrics = BacktestEngine.calculate_metrics(bt_results['returns'], rf_rate)
            bench_metrics = BacktestEngine.calculate_metrics(strat_prices.pct_change().dropna(), rf_rate)
            
            # Display Metrics
            st.write("#### 📊 Performance Metrics")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            
            with met_col1:
                st.metric("Total Return (Strategy)", f"{(bt_results['equity_curve'].iloc[-1]/initial_cap - 1)*100:.2f}%")
            with met_col2:
                st.metric("Sharpe Ratio", f"{strat_metrics.get('Sharpe Ratio', 0):.2f}")
            with met_col3:
                st.metric("Max Drawdown", f"{strat_metrics.get('Max Drawdown', 0)*100:.2f}%")
            with met_col4:
                st.metric("Benchmark Return", f"{(bt_results['benchmark_curve'].iloc[-1]/initial_cap - 1)*100:.2f}%")
                
            # Equity Curve Plot
            st.write("#### 📈 Equity Curve")
            fig_bt, ax_bt = plt.subplots(figsize=(12, 6))
            ax_bt.plot(bt_results['equity_curve'], label=f'Strategy ({strategy_type})', color='green', linewidth=2)
            ax_bt.plot(bt_results['benchmark_curve'], label='Buy & Hold (Benchmark)', color='gray', linestyle='--', alpha=0.7)
            ax_bt.set_title(f"Strategy Performance: {TICKER}")
            ax_bt.legend()
            format_plot_dates(ax_bt, bt_results['equity_curve'].index)
            st.pyplot(fig_bt)
            
            # Trade Log
            st.write("#### 📝 Trade Log")
            trades_df = bt_results['trades']
            if not trades_df.empty:
                # Format dates
                trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.date
                trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).apply(lambda x: x.date() if pd.notnull(x) else "Open")
                
                st.dataframe(trades_df.style.format({
                    "Buy Price": "{:.2f}",
                    "Sell Price": "{:.2f}",
                    "PnL (%)": "{:.2f}%"
                }), use_container_width=True)
            else:
                st.info("No closed trades generated by the strategy.")


else:
    st.info("Enter a ticker and ensure data is loaded to begin analysis.")

# Footer
st.markdown("---")
st.caption("Generated via Gemini 2.0 Flash | Robust Financial Thesis Implementation")
