import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import smtplib
from email.message import EmailMessage

# Default historical/non-live anchor date
DEFAULT_NONLIVE_START = datetime(2024, 1, 1)
# Try importing export libraries
try:
    from fpdf import FPDF
    import xlsxwriter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False

# Try importing arch, handle if missing
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Try importing sklearn
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Statsmodels Diagnostic Imports

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

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

def highlight_plotly_zones(fig, mask, color, opacity=0.15, row=None, col=None):
    """
    Highlights contiguous boolean True blocks in Plotly charts using vertical rectangles.
    """
    if not isinstance(mask, pd.Series) or not mask.any():
        return
    blocks = (~mask).cumsum()
    for _, group in mask[mask].groupby(blocks[mask]):
        if len(group) > 0:
            x0 = group.index[0]
            x1 = group.index[-1]
            if x0 == x1:
                x1 = x0 + pd.Timedelta(days=1)
            if row is not None and col is not None:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity, layer="below", line_width=0, row=row, col=col)
            else:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity, layer="below", line_width=0)

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

class RealizedVolatility:
    @staticmethod
    def realized_variance(returns):
        """Standard Realized Variance (sum of squared returns)."""
        return np.sum(returns**2)

    @staticmethod
    def bipower_variation(returns):
        """Bipower Variation (robust to jumps).
        BV = (pi/2) * sum(|rt| * |rt-1|)
        """
        abs_rets = np.abs(returns)
        # Vectorized scalar product of lagged absolutes
        if len(returns) < 2: return 0.0
        return (np.pi / 2) * np.sum(abs_rets[1:] * abs_rets[:-1])

    @staticmethod
    def jump_component(returns):
        """Tests for jumps using RV and BV interaction (Barndorff-Nielsen & Shephard)."""
        rv = RealizedVolatility.realized_variance(returns)
        bv = RealizedVolatility.bipower_variation(returns)
        
        # Jump contribution is difference between Total Vol (RV) and Continuous Vol (BV)
        jump_var = max(rv - bv, 0)
        jump_ratio = jump_var / rv if rv > 0 else 0.0
        
        # Simplified significance test (Heuristic)
        # Z = (RV - BV) / (RV * sqrt(theta * max(1/N, some_const)))
        n = len(returns)
        if n < 10: 
            return {'jump_ratio': 0.0, 'p_value': 1.0, 'z_score': 0.0}
        
        # Heuristic Z-score for significance
        # A jump ratio > 0.5 with high data count is usually significant
        z_score = (jump_ratio - 0.05) * np.sqrt(n/2) # Simple heuristic scaling
        p_value = 1 - stats.norm.cdf(z_score)
        
        return {'jump_ratio': jump_ratio, 'p_value': p_value, 'z_score': z_score}

class HawkesVolatility:
    def __init__(self):
        self.mu = 0.5
        self.alpha = 0.5 # Excitation
        self.beta = 2.0  # Decay
        self.metrics = {}

    def fit(self, returns):
        """
        Fits a simple univariate Hawkes process to Volatility Peaks (POT).
        """
        # 1. Identify "Events" (Extreme Volatility)
        # Use simple Peak Over Threshold (POT) on absolute returns
        vol_proxy = np.abs(returns)
        if len(vol_proxy) < 20:
             return self
             
        threshold = np.percentile(vol_proxy, 90) # Top 10% events
        events = np.where(vol_proxy > threshold)[0]
        
        if len(events) < 5:
             # Not enough data
             return self
             
        # LL Function for Hawkes: sum(log(lambda(ti))) - integral(lambda(t))
        # lambda(t) = mu + sum(alpha * exp(-beta * (t - ti)))
        
        def neg_log_likelihood(params):
            mu_p, alpha_p, beta_p = params
            if mu_p <= 0 or alpha_p < 0 or beta_p <= alpha_p: return 1e9
            
            t = events
            n = len(t)
            T_end = len(returns) # total duration in days
            
            # Recursive calculation of R(k) = sum(exp(-beta*(tk - ti)))
            # R(k) = exp(-beta*(tk - tk-1)) * (1 + R(k-1))
            R = np.zeros(n)
            for i in range(1, n):
                dt = t[i] - t[i-1]
                R[i] = np.exp(-beta_p * dt) * (1 + R[i-1])
            
            # Avoid log(0)
            intensities = mu_p + alpha_p * R
            if np.any(intensities <= 0): return 1e9
            
            term1 = np.sum(np.log(intensities))
            
            # Integral term: int(mu) + sum(alpha/beta * (1 - exp(-beta*(T - ti))))
            term2 = mu_p * T_end + (alpha_p / beta_p) * np.sum(1 - np.exp(-beta_p * (T_end - t)))
            
            return -(term1 - term2)
            
        try:
            # Bounds: mu>0, alpha>0, beta>alpha (stationarity)
            res = minimize(neg_log_likelihood, [0.1, 0.2, 1.0], 
                           bounds=[(1e-4, 2.0), (1e-4, 5.0), (0.1, 10.0)], method='L-BFGS-B')
            self.mu, self.alpha, self.beta = res.x
        except:
            pass # Keep defaults
            
        return self

    def branching_ratio(self):
        """Measure of self-excitement intensity (alpha/beta). <1 is stable."""
        if self.beta == 0: return 0.0
        return self.alpha / self.beta

    def half_life(self):
        """Time for a shock to decay by half (ln(2)/beta)."""
        if self.beta == 0: return 0.0
        return np.log(2) / self.beta

class AdvancedRegimeDetector:
    def __init__(self, log_returns):
        self.data = log_returns.values.reshape(-1, 1) if hasattr(log_returns, 'values') else log_returns
        self.dates = log_returns.index if hasattr(log_returns, 'index') else np.arange(len(log_returns))
        self.metrics = {}
        self.regimes = {}
        self.regime_characteristics = []

    def fit_all(self, n_states=3):
        """Fits both HMM and Bayesian Changepoint logic."""
        
        # 1. HMM (using GMM Proxy if sklearn available)
        if SKLEARN_AVAILABLE:
            from sklearn.mixture import GaussianMixture
            
            # Fit GMM as HMM proxy (Regime Clustering)
            model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
            model.fit(self.data)
            
            hidden_states = model.predict(self.data)
            probs = model.predict_proba(self.data)
            
            # Re-order states by mean volatility (0=Low, 1=Med, 2=High)
            state_vars = []
            for i in range(n_states):
                # Filter data for this state
                mask = (hidden_states == i)
                if np.sum(mask) > 0:
                    state_vars.append(np.std(self.data[mask]))
                else:
                    state_vars.append(0)
                
            # Sort indices: low vol -> high vol
            sorted_idx = np.argsort(state_vars)
            
            # Create mapping: old_id -> new_id (0,1,2)
            map_dict = {old: new for new, old in enumerate(sorted_idx)}
            
            # Apply mapping to states and probs
            sorted_states = np.vectorize(map_dict.get)(hidden_states)
            sorted_probs = probs[:, sorted_idx]
            
            self.regimes['hmm_states'] = sorted_states
            self.regimes['hmm_probs'] = sorted_probs
            self.metrics['hmm_aic'] = model.aic(self.data)
            
            # Calculate Characteristics
            self._calculate_characteristics(sorted_states)
        else:
            # Fallback if no sklearn
            self.regimes['hmm_probs'] = np.zeros((len(self.data), n_states))
            self.metrics['hmm_aic'] = 0
        
        # 2. Bayesian Changepoint Detection (Simplified Proxy)
        self.regimes['changepoint_probs'] = self._bayesian_changepoint_proxy(self.data.flatten())
        
    def _bayesian_changepoint_proxy(self, data):
        """
        Fast proxy for changepoint probability using rolling volatility regime shifts.
        True BCP is computationally heavy for Streamlit.
        """
        vol = pd.Series(data).rolling(window=22).std().bfill()
        
        # Detect shifts in volatility (Z-score of vol change)
        vol_change = vol.diff().abs()
        mean_change = vol_change.rolling(252, min_periods=20).mean()
        std_change = vol_change.rolling(252, min_periods=20).std()
        
        # Z-score
        z = (vol_change - mean_change) / (std_change + 1e-8)
        
        # Sigmoid probability transform: Z > 2 implies likely shift
        probs = 1 / (1 + np.exp(-(z - 2.0)))
        return probs.fillna(0).values

    def _calculate_characteristics(self, states):
        df = pd.DataFrame(self.data, columns=['ret'])
        df['state'] = states
        
        # Group by state
        stats_df = df.groupby('state')['ret'].agg(['mean', 'std', 'count'])
        
        self.regime_characteristics = []
        labels = ['Bull/Calm', 'Normal/Transition', 'Bear/Crisis'] 
        
        for i in range(len(stats_df)):
            if i >= len(labels): cn = f"State {i}"
            else: cn = labels[i]
            
            s = stats_df.iloc[i]
            self.regime_characteristics.append({
                'label': cn,
                'mean_return': s['mean'] * 252, # Annualized
                'volatility': s['std'] * np.sqrt(252),
                'frequency': s['count'] / len(df),
                'avg_duration': 0.0, # Placeholder
                'max_drawdown': 0.0 # Placeholder
            })
            
    def get_trading_signal(self):
        """Derives signal from latest regime probabilities."""
        if 'hmm_probs' not in self.regimes:
            return "N/A", {'label': 'No Model', 'confidence': 0.0}
            
        probs = self.regimes['hmm_probs'][-1] # [Low, Med, High] sorted
        
        # Logic: 
        # High Vol (Idx 2) > 50% => RISK OFF
        # Low Vol (Idx 0) > 60% => RISK ON
        # Else => NEUTRAL
        
        n_states = len(probs)
        if n_states < 3:
             return "NEUTRAL", {'label': 'Unknown', 'confidence': 0.0}
             
        p_safe = probs[0]
        p_danger = probs[-1]
        
        if p_danger > 0.5:
            return "DEFENSIVE / SHORT", {'label': 'High Volatility', 'confidence': p_danger}
        elif p_safe > 0.6:
            return "AGGRESSIVE LONG", {'label': 'Low Volatility', 'confidence': p_safe}
        else:
            return "NEUTRAL / HEDGED", {'label': 'Transition/Mixed', 'confidence': max(probs)}

class ProRegimeDetector:
    """
    Institutional-grade Regime Detection using Multi-Factor Feature Vectors.
    Analyzes Returns, Volatility, and Trend-Deviation simultaneously.
    """
    def __init__(self, prices, log_returns):
        self.prices = prices if isinstance(prices, pd.Series) else pd.Series(prices)
        self.returns = log_returns if isinstance(log_returns, pd.Series) else pd.Series(log_returns)
        self.features = None
        self.regimes = {}
        self.metrics = {}
        self.state_labels = {}

    def _prepare_features(self):
        # 1. Momentum (Short-term smoothed returns)
        f1 = self.returns.rolling(window=5).mean().fillna(0)
        
        # 2. Volatility Cluster (Z-scored 20d Vol)
        vol = self.returns.rolling(window=20).std().bfill()
        v_mean = vol.rolling(252, min_periods=20).mean()
        v_std = vol.rolling(252, min_periods=20).std()
        f2 = (vol - v_mean) / (v_std + 1e-9)
        f2 = f2.fillna(0)
        
        # 3. Structural Deviation (Price vs EMA20)
        ema = self.prices.ewm(span=20).mean()
        f3 = (self.prices - ema) / (ema + 1e-9)
        f3 = f3.fillna(0)
        
        self.features = np.column_stack([f1.values, f2.values, f3.values])
        return self.features

    def fit(self, n_states=4):
        X = self._prepare_features()
        if SKLEARN_AVAILABLE:
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Multivariate GMM
            model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=123, max_iter=200)
            model.fit(X_scaled)
            
            states = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)
            
            # --- INSTITUTIONAL STATE MAPPING ---
            # We characterize states by their Mean Returns and Mean Volatility
            state_stats = []
            for i in range(n_states):
                mask = (states == i)
                if np.sum(mask) > 0:
                    m_ret = np.mean(self.features[mask, 0])
                    m_vol = np.mean(self.features[mask, 1])
                    state_stats.append({'id': i, 'ret': m_ret, 'vol': m_vol})
                else:
                    state_stats.append({'id': i, 'ret': -999, 'vol': 999})
            
            # Logic-based labeling:
            # 1. Bull/Quiet: High Ret, Low Vol
            # 2. Bull/Volatile: High Ret, High Vol (Overextended)
            # 3. Bear/Quiet: Low Ret, Low Vol (Distribution)
            # 4. Bear/Volatile: Low Ret, High Vol (Panic/Crisis)
            
            # Sort by returns
            # Sort states by returns for logical mapping
            sorted_stats = sorted(state_stats, key=lambda x: x['ret'], reverse=True)
            
            if n_states == 4:
                bulls = sorted_stats[:2]
                bears = sorted_stats[2:]
                bull_low = min(bulls, key=lambda x: x['vol'])
                bull_high = max(bulls, key=lambda x: x['vol'])
                bear_low = min(bears, key=lambda x: x['vol'])
                bear_high = max(bears, key=lambda x: x['vol'])
                self.state_labels = {
                    bull_low['id']: "BULL / QUIET (Conviction)",
                    bull_high['id']: "BULL / VOLATILE (Exhaustion)",
                    bear_low['id']: "BEAR / QUIET (Distribution)",
                    bear_high['id']: "BEAR / VOLATILE (Panic/Crisis)"
                }
            elif n_states == 2:
                self.state_labels = {
                    sorted_stats[0]['id']: "BULL REGIME (Accumulation)",
                    sorted_stats[1]['id']: "BEAR REGIME (Distribution)"
                }
            elif n_states == 3:
                self.state_labels = {
                    sorted_stats[0]['id']: "BULL REGIME (Conviction)",
                    sorted_stats[1]['id']: "NEUTRAL / TRANSITION",
                    sorted_stats[2]['id']: "BEAR REGIME (Panic)"
                }
            
            self.regimes['states'] = states
            self.regimes['probs'] = probs
            # Calculate BIC manually for GMM (Scikit-learn model has .bic())
            self.metrics['aic'] = model.aic(X_scaled)
            self.metrics['bic'] = model.bic(X_scaled)
            self.metrics['n_states'] = n_states
        else:
            self.regimes['states'] = np.zeros(len(X))
            self.regimes['probs'] = np.ones((len(X), 1))

    def fit_optimized(self, state_choices=[2, 3, 4]):
        """Runs multiple models and picks the best one by BIC."""
        best_bic = float('inf')
        best_n = 4
        
        # We perform a quick search for best BIC
        for n in state_choices:
            try:
                temp_model = ProRegimeDetector(self.prices, self.returns)
                temp_model.fit(n_states=n)
                if temp_model.metrics.get('bic', float('inf')) < best_bic:
                    best_bic = temp_model.metrics['bic']
                    best_n = n
            except:
                continue
        
        # Final fit with best N
        self.fit(n_states=best_n)
        return best_n

    def get_latest_verdict(self):
        if 'states' not in self.regimes or not self.state_labels:
            return "NEUTRAL", 0.0, "N/A"
            
        last_state = self.regimes['states'][-1]
        last_prob = np.max(self.regimes['probs'][-1])
        label = self.state_labels.get(last_state, "Unknown")
        
        if "BULL" in label:
            verdict = "ACCUMULATE / LONG" if "QUIET" in label else "HEDGE / CAUTION"
        elif "BEAR" in label:
            verdict = "DEFENSIVE / SHORT" if "VOLATILE" in label else "REDUCE EXPOSURE"
        else:
            verdict = "NEUTRAL"
            
        return verdict, last_prob, label

class SMLAnalyzer:
    def __init__(self, ticker_returns, benchmark_returns, rf_annual=0.04):
        self.r_asset = ticker_returns
        self.r_bench = benchmark_returns
        self.rf_annual = rf_annual
        self.rf_daily = rf_annual / 252
        
    def calculate_metrics(self, window=90):
        # Align data
        common_idx = self.r_asset.index.intersection(self.r_bench.index)
        y = self.r_asset.loc[common_idx] - self.rf_daily # Excess Asset Returns
        x = self.r_bench.loc[common_idx] - self.rf_daily # Excess Market Returns
        
        df = pd.DataFrame({'asset_ex': y, 'mkt_ex': x}, index=common_idx)
        
        # 1. Rolling Beta & Alpha (HAC Robust)
        betas = []
        alphas = []
        
        # Pre-allocate array for speed
        beta_arr = np.full(len(df), np.nan)
        alpha_arr = np.full(len(df), np.nan)
        
        # Need at least 'window' points
        for i in range(window, len(df)):
            window_slice = df.iloc[i-window:i]
            y_win = window_slice['asset_ex']
            x_win = sm.add_constant(window_slice['mkt_ex'])
            
            try:
                # OLS with HAC (Heteroskedasticity & Autocorrelation Consistent) Standard Errors
                # maxlags=1 is usually sufficient for daily stock data
                model = sm.OLS(y_win, x_win).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
                alpha_arr[i] = model.params.get('const', np.nan)
                beta_arr[i] = model.params.get('mkt_ex', np.nan)
            except:
                pass # nan default
                
        df['Beta'] = beta_arr
        df['Alpha_Daily'] = alpha_arr
        
        # 2. CAPM Checks
        # E(Ri) = Rf + Beta(E(Rm) - Rf)
        # We use realized market return over the window as proxy for E(Rm)
        rolling_mkt_ret_ann = df['mkt_ex'].rolling(window).mean() * 252
        
        df['SML_Exp_Return'] = self.rf_annual + (df['Beta'] * rolling_mkt_ret_ann)
        df['Actual_Return_Ann'] = (df['asset_ex'].rolling(window).mean() * 252) + self.rf_annual
        
        # Mispricing (Alpha in return space)
        df['Mispricing_Spread'] = df['Actual_Return_Ann'] - df['SML_Exp_Return']
        
        return df.dropna()

class MADTrendModes:
    """
    Translates the 'MAD Trend Modes' logic from Pine Script.
    Includes Mean Absolute Deviation (MAD), "For Loop" system, and various MAs.
    """
    @staticmethod
    def sma(series, length):
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def wma(series, length):
        weights = np.arange(1, length + 1)
        return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    @staticmethod
    def hma(series, length):
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        wma_half = MADTrendModes.wma(series, half_length)
        wma_full = MADTrendModes.wma(series, length)
        combined = 2 * wma_half - wma_full
        return MADTrendModes.wma(combined.dropna(), sqrt_length).reindex_like(series)

    @staticmethod
    def rma(series, length):
        """
        RMA = 1/L * src + (1 - 1/L) * RMA[1]
        Equivalent to EMA with alpha = 1/L
        """
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def alma(series, length, offset=0.85, sigma=6):
        """
        Arnaud Legoux Moving Average
        """
        m = offset * (length - 1)
        s = length / sigma
        weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        weights /= weights.sum()
        return series.rolling(window=length).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def lsma(series, length):
        """
        Least Squares Moving Average (Linear Regression)
        """
        def linreg_end(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            return slope * (len(y) - 1) + intercept
        return series.rolling(window=length).apply(linreg_end, raw=True)

    @staticmethod
    def ma_switch(series, length, avg_type):
        if avg_type == "SMA": return MADTrendModes.sma(series, length)
        if avg_type == "EMA": return MADTrendModes.ema(series, length)
        if avg_type == "WMA": return MADTrendModes.wma(series, length)
        if avg_type == "HMA": return MADTrendModes.hma(series, length)
        if avg_type == "RMA": return MADTrendModes.rma(series, length)
        if avg_type == "ALMA": return MADTrendModes.alma(series, length)
        if avg_type == "LSMA": return MADTrendModes.lsma(series, length)
        # Fallback to SMA if type not implemented
        return MADTrendModes.sma(series, length)

    @staticmethod
    def calculate_mad(series, benchmark, length):
        """
        Vectorized MAD: sum(|src[t-i] - benchmark[t]|) / length
        """
        # Using numpy stride tricks for ultra-fast windowing
        from numpy.lib.stride_tricks import sliding_window_view
        
        vals = series.values
        bench_vals = benchmark.values
        
        # Create sliding windows of size 'length'
        if len(vals) < length:
            return pd.Series(np.nan, index=series.index)
            
        windows = sliding_window_view(vals, length)
        # windows[i] is the window ending at index i + length - 1
        
        # We need to subtract bench_vals[i + length - 1] from each element in windows[i]
        # broad casting: windows shape (N-L+1, L), bench_vals[L-1:] shape (N-L+1,)
        diffs = np.abs(windows - bench_vals[length-1:, np.newaxis])
        res_vals = np.mean(diffs, axis=1)
        
        # Pad with NaNs for the beginning
        res = np.full(len(series), np.nan)
        res[length-1:] = res_vals
        return pd.Series(res, index=series.index)

    @staticmethod
    def system_score(series, a, b):
        """
        Vectorized system: sum(sign(src[t] - src[t-i])) for i in a..b
        """
        total = pd.Series(0.0, index=series.index)
        for i in range(a, b + 1):
            shifted = series.shift(i)
            # Use np.sign logic: (series > shifted) - (series < shifted)
            total += np.sign(series - shifted).fillna(0)
        return total

    @staticmethod
    def get_signals(df, params):
        """
        Generates strategy signals based on parameters.
        """
        src = df['Close']
        mode = params.get('signal_mode', 'Bollinger Bands')
        
        # BB Params
        bb_ma_type = params.get('bb_ma_type', 'EMA')
        bb_len = params.get('bb_len', 25)
        bb_mult_p = params.get('bb_mult_p', 1.4)
        bb_mult_n = params.get('bb_mult_n', 1.0)
        
        # for loop params
        fl_ma_type = params.get('fl_ma_type', 'ALMA') # Fallback to SMA if not impl
        fl_len = params.get('fl_len', 10)
        fl_a = params.get('fl_a', 10)
        fl_b = params.get('fl_b', 60)
        fl_thresh_l = params.get('fl_thresh_l', 23)
        fl_thresh_s = params.get('fl_thresh_s', 3)
        
        # combined params
        c_thresh_l = params.get('c_thresh_l', 0.0)
        c_thresh_s = params.get('c_thresh_s', 0.0)

        # 1. BB Calculations
        avg_bb = MADTrendModes.ma_switch(src, bb_len, bb_ma_type)
        mad_bb = MADTrendModes.calculate_mad(src, avg_bb, bb_len)
        bb_up = avg_bb + (mad_bb * bb_mult_p)
        bb_dn = avg_bb - (mad_bb * bb_mult_n)
        
        # 2. FL Calculations
        avg_fl = MADTrendModes.ma_switch(src, fl_len, fl_ma_type)
        mad_fl_val = MADTrendModes.calculate_mad(src, avg_fl, fl_len)
        
        # Weighted source for system
        # mad_w_src = ma_switch(source*mad2, mad_length_fl, ma_benchmark_type_fl) / ma_switch(mad2, mad_length_fl, ma_benchmark_type_fl)
        num = MADTrendModes.ma_switch(src * mad_fl_val, fl_len, fl_ma_type)
        den = MADTrendModes.ma_switch(mad_fl_val, fl_len, fl_ma_type)
        mad_w_src = num / den
        
        sys_score = MADTrendModes.system_score(mad_w_src, fl_a, fl_b)
        
        # Crossovers
        bb_long = (src > bb_up) & (src.shift(1) <= bb_up.shift(1))
        bb_short = (src < bb_dn) & (src.shift(1) >= bb_dn.shift(1))

        # Stateful Signal logic
        def get_stateful_signal(long_cond, short_cond, index):
            sig = pd.Series(np.nan, index=index)
            sig.loc[long_cond] = 1
            sig.loc[short_cond] = -1
            return sig.ffill().fillna(0)

        bb_score = get_stateful_signal(bb_long, bb_short, src.index)
        
        fl_long = (sys_score > fl_thresh_l) & (sys_score.shift(1) <= fl_thresh_l)
        fl_short = (sys_score < fl_thresh_s) & (sys_score.shift(1) >= fl_thresh_s)
        fl_score = get_stateful_signal(fl_long, fl_short, src.index)
        
        c_signal = (bb_score + fl_score) / 2
        c_long = (c_signal > c_thresh_l) & (c_signal.shift(1) <= c_thresh_l)
        c_short = (c_signal < c_thresh_s) & (c_signal.shift(1) >= c_thresh_s)
        combined_score = get_stateful_signal(c_long, c_short, src.index)
            
        # Final Selection
        if mode == "Bollinger Bands":
            final_score = bb_score
        elif mode == "For Loop":
            final_score = fl_score
        else: # Combined
            final_score = combined_score
            
        return (final_score == 1).astype(int)

def rolling_hurst(prices, window=100, max_lag=20):
    """Calculates the rolling Hurst Exponent using log-variance approximation."""
    # Use log prices for accurate variance scaling of returns
    log_prices = np.log(prices)
    def hurst_val(x):
        lags = range(2, max_lag)
        tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
        tau = [t if t > 0 else 1e-8 for t in tau]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    return log_prices.rolling(window).apply(hurst_val, raw=True)

class EhlersFilters:
    @staticmethod
    def super_smoother(prices, period=15):
        """
        Ehlers SuperSmoother Filter (2-pole Butterworth with SMA)
        """
        a1 = np.exp(-1.414 * np.pi / period)
        b1 = 2 * a1 * np.cos(1.414 * np.pi / period)
        c2 = b1
        c3 = -a1 * a1
        c1 = 1 - c2 - c3
        
        filt = np.zeros(len(prices))
        prices_vals = prices.values
        
        for i in range(len(prices)):
            if i < 2:
                filt[i] = prices_vals[i]
            else:
                filt[i] = c1 * (prices_vals[i] + prices_vals[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
                
        return pd.Series(filt, index=prices.index)
        
    @staticmethod
    def simple_decycler(prices, period=60):
        """
        Ehlers Simple Decycler
        Removes high frequency components to leave underlying trend
        """
        alpha1 = (np.cos(0.707 * 2 * np.pi / period) + np.sin(0.707 * 2 * np.pi / period) - 1) / np.cos(0.707 * 2 * np.pi / period)
        
        hp = np.zeros(len(prices))
        prices_vals = prices.values
        
        for i in range(len(prices)):
            if i < 2:
                hp[i] = 0
            else:
                hp[i] = ((1 - alpha1 / 2)**2) * (prices_vals[i] - 2 * prices_vals[i-1] + prices_vals[i-2]) + \
                        2 * (1 - alpha1) * hp[i-1] - ((1 - alpha1)**2) * hp[i-2]
                        
        decycler = prices_vals - hp
        return pd.Series(decycler, index=prices.index)

@st.cache_data(ttl=300, show_spinner=False)
def get_iv_metrics(ticker: str) -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="252d", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 30:
            return None
        current_price = float(hist['Close'].iloc[-1])
        if current_price <= 0:
            return None
        log_rets = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        hv_30 = float(log_rets.tail(21).std() * np.sqrt(252) * 100)
        hv_60 = float(log_rets.tail(42).std() * np.sqrt(252) * 100) if len(log_rets) >= 42 else hv_30
        hv_252 = float(log_rets.std() * np.sqrt(252) * 100)
        expirations = tk.options
        if not expirations:
            return None
        now = datetime.now()
        valid_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - now).days
            if 7 <= dte <= 90:
                valid_exps.append((exp, dte))
        if not valid_exps:
            return None
        all_call_iv, all_put_iv, atm_call_iv_list, atm_put_iv_list = [], [], [], []
        total_call_vol, total_put_vol, total_call_oi, total_put_oi = 0, 0, 0, 0
        skew_readings = []
        for exp, dte in valid_exps[:4]:
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls[(chain.calls['impliedVolatility'] > 0.01) & (chain.calls['impliedVolatility'] < 5.0)].copy()
                puts = chain.puts[(chain.puts['impliedVolatility'] > 0.01) & (chain.puts['impliedVolatility'] < 5.0)].copy()
                if calls.empty or puts.empty:
                    continue
                atm_range = (current_price * 0.95, current_price * 1.05)
                atm_calls = calls[(calls['strike'] >= atm_range[0]) & (calls['strike'] <= atm_range[1])]
                atm_puts = puts[(puts['strike'] >= atm_range[0]) & (puts['strike'] <= atm_range[1])]
                if not atm_calls.empty:
                    atm_call_iv_list.append(float(atm_calls['impliedVolatility'].median()))
                if not atm_puts.empty:
                    atm_put_iv_list.append(float(atm_puts['impliedVolatility'].median()))
                otm_puts = puts[(puts['strike'] >= current_price * 0.85) & (puts['strike'] < current_price * 0.95)]
                otm_calls = calls[(calls['strike'] > current_price * 1.05) & (calls['strike'] <= current_price * 1.15)]
                if not otm_puts.empty and not atm_calls.empty:
                    put_iv = float(otm_puts['impliedVolatility'].median())
                    call_iv_atm = float(atm_calls['impliedVolatility'].median())
                    skew_readings.append(put_iv - call_iv_atm)
                total_call_vol += int(calls['volume'].fillna(0).sum())
                total_put_vol += int(puts['volume'].fillna(0).sum())
                total_call_oi += int(calls['openInterest'].fillna(0).sum())
                total_put_oi += int(puts['openInterest'].fillna(0).sum())
            except Exception:
                continue
        if not atm_call_iv_list:
            return None
        current_atm_iv = float(np.mean(atm_call_iv_list) * 100)
        if len(log_rets) >= 252:
            rolling_vols = log_rets.rolling(21).std().dropna() * np.sqrt(252) * 100
            iv_52w_low = float(rolling_vols.min())
            iv_52w_high = float(rolling_vols.max())
            iv_percentile = float((rolling_vols < current_atm_iv).mean() * 100)
        else:
            iv_52w_low = min(hv_30, hv_252) * 0.8
            iv_52w_high = max(hv_30, hv_252) * 1.3
            iv_percentile = float(np.clip(((current_atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low + 1e-6)) * 100, 0, 100))
        iv_rank = float(np.clip(((current_atm_iv - iv_52w_low) / (iv_52w_high - iv_52w_low + 1e-6)) * 100, 0, 100))
        pc_ratio = total_put_vol / (total_call_vol + 1e-6)
        pc_oi_ratio = total_put_oi / (total_call_oi + 1e-6)
        skew = float(np.mean(skew_readings)) * 100 if skew_readings else 0.0
        iv_hv_ratio = current_atm_iv / (hv_30 + 1e-6)
        term_structure_slope = (atm_call_iv_list[-1] - atm_call_iv_list[0]) * 100 if len(atm_call_iv_list) >= 2 else 0.0
        
        score = 0.0
        signals = []
        if iv_rank < 30 and iv_hv_ratio > 1.05:
            score += 2.5
            signals.append(("IV Expansion from Low Base", "green", f"IVR={iv_rank:.0f} (low), IV/HV={iv_hv_ratio:.2f} (rising)"))
        if skew < -2.0:
            score += 2.0
            signals.append(("Bullish Call Skew", "green", f"Skew={skew:.1f}% (calls premium over puts)"))
        elif skew > 5.0:
            score -= 1.0
            signals.append(("Bearish Put Skew", "red", f"Skew={skew:.1f}% (puts heavily bid — hedging)"))
        if pc_ratio < 0.6 and current_atm_iv > hv_30:
            score += 2.0
            signals.append(("Call Buying Dominance", "green", f"P/C={pc_ratio:.2f} (call-heavy), IV > HV"))
        if iv_rank > 70 and iv_hv_ratio > 1.3:
            score += 1.5
            signals.append(("IV Crush Setup", "orange", f"IVR={iv_rank:.0f} (elevated), IV/HV={iv_hv_ratio:.2f}"))
        if term_structure_slope > 1.0:
            score += 1.0
            signals.append(("Contango IV Structure", "green", f"Near→Far slope: +{term_structure_slope:.1f}%"))
        elif term_structure_slope < -3.0:
            score -= 1.0
            signals.append(("Backwardation IV Structure", "orange", f"Near→Far slope: {term_structure_slope:.1f}% (event risk near)"))
        if pc_oi_ratio < 0.5:
            score += 1.0
            signals.append(("Heavy Call OI", "green", f"P/C OI={pc_oi_ratio:.2f} (call-heavy positioning)"))
        if iv_hv_ratio < 0.7:
            score -= 1.5
            signals.append(("IV Suppressed vs HV", "red", f"IV/HV={iv_hv_ratio:.2f} (options underpricing risk)"))
            
        if score >= 4.0:
            verdict, verdict_color = "STRONG BUY", "#00ff88"
        elif score >= 2.5:
            verdict, verdict_color = "BUY", "#44cc66"
        elif score >= 1.0:
            verdict, verdict_color = "WATCH", "#ffcc00"
        elif score <= -1.0:
            verdict, verdict_color = "AVOID", "#ff4444"
        else:
            verdict, verdict_color = "NEUTRAL", "#aaaaaa"
            
        return {
            'ticker': ticker, 'price': current_price, 'atm_iv': current_atm_iv, 'iv_rank': iv_rank,
            'iv_percentile': iv_percentile, 'hv_30': hv_30, 'hv_60': hv_60, 'hv_252': hv_252,
            'iv_hv_ratio': iv_hv_ratio, 'skew': skew, 'pc_ratio': pc_ratio, 'pc_oi_ratio': pc_oi_ratio,
            'term_structure_slope': term_structure_slope, 'call_vol': total_call_vol, 'put_vol': total_put_vol,
            'call_oi': total_call_oi, 'put_oi': total_put_oi, 'score': score, 'verdict': verdict,
            'verdict_color': verdict_color, 'signals': signals, 'mkt_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'), 'name': info.get('shortName', ticker)
        }
    except Exception:
        return None

def build_iv_surface(ticker: str, current_price: float):
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        now = datetime.now()
        surface_rows = []
        for exp in expirations[:6]:
            exp_date = datetime.strptime(exp, "%Y-%m-%d")
            dte = (exp_date - now).days
            if dte < 5:
                continue
            try:
                chain = tk.option_chain(exp)
                calls = chain.calls[(chain.calls['impliedVolatility'] > 0.01) & (chain.calls['volume'].fillna(0) > 0)].copy()
                for _, row in calls.iterrows():
                    moneyness = (row['strike'] / current_price - 1) * 100
                    if -25 <= moneyness <= 25:
                        surface_rows.append({'DTE': dte, 'Moneyness': round(moneyness, 1), 'IV': round(row['impliedVolatility'] * 100, 2)})
            except Exception:
                continue
        return pd.DataFrame(surface_rows) if surface_rows else None
    except Exception:
        return None

@st.cache_data(ttl=86400)
def get_sp500():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","BRK-B","UNH","JNJ", "JPM","V","PG","MA","HD","CVX","MRK","ABBV","PEP","COST"]

@st.cache_data(ttl=86400)
def get_nasdaq100():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for t in tables:
            if 'Ticker' in t.columns:
                return t['Ticker'].tolist()
        return tables[4].iloc[:, 1].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","AVGO","PEP","COST"]


class BacktestEngine:
    """
    Handles simple vectorised backtesting for regime-based strategies.
    """
    @staticmethod
    def run_strategy(prices, signals, initial_capital=10000.0, trailing_stop_pct=0.0, stop_loss_pct=0.0):
        """
        prices: Series of asset prices
        signals: Series of 1 (Long) or 0 (Cash/Neutral). Index must match prices.
        trailing_stop_pct: Float (e.g., 0.05 for 5%). If > 0, applies trailing stop.
        stop_loss_pct: Float (e.g., 0.08 for 8%). If > 0, applies a hard stop loss.

        Important:
        The equity curve, performance metrics, and trade log all come from the same
        account-level accounting below. Individual trade PnL can differ from total
        strategy return because the account compounds and may use fractional exposure.
        """
        # Align and clean
        common_idx = prices.index.intersection(signals.index)
        prices = pd.Series(prices.loc[common_idx]).replace([np.inf, -np.inf], np.nan).dropna()
        signals = pd.Series(signals).reindex(prices.index).ffill().fillna(0.0).astype(float).clip(0.0, 1.0)

        if len(prices) == 0:
            empty = pd.Series(dtype=float)
            return {
                'equity_curve': empty,
                'benchmark_curve': empty,
                'trades': pd.DataFrame(),
                'returns': empty
            }

        returns = prices.pct_change().fillna(0.0)

        equity_vals = []
        trades = []

        position = 0  # 0: Cash, 1: Long
        entry_price = 0.0
        entry_date = None
        entry_equity = initial_capital
        entry_signal = 0.0
        max_price_since_entry = 0.0

        cash = float(initial_capital)
        holdings = 0.0
        cooldown_bars = 0

        def current_equity(price):
            return float(cash + holdings * price)

        def record_trade(exit_date, exit_price, status_msg):
            nonlocal cash, holdings, position, entry_price, entry_date, entry_equity, entry_signal

            trade_return_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price else 0.0
            exit_equity = current_equity(exit_price)
            account_return_pct = ((exit_equity / entry_equity) - 1.0) * 100.0 if entry_equity else 0.0
            cumulative_return_pct = ((exit_equity / initial_capital) - 1.0) * 100.0 if initial_capital else 0.0

            trades.append({
                'Side': 'Long',
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Buy Price': float(entry_price),
                'Sell Price': float(exit_price),
                'PnL (%)': float(trade_return_pct),              # single trade price return
                'Cumulative Return (%)': round(float(cumulative_return_pct), 2),  # total account return after this trade, rounded to 2 decimals for clean display
                'Status': status_msg
            })

        for date, price in prices.items():
            price = float(price)
            desired_signal = float(signals.loc[date])

            # Cooldown logic to prevent immediate re-entry after stop loss/trailing stop
            if cooldown_bars > 0:
                cooldown_bars -= 1
                if desired_signal == 0:
                    cooldown_bars = 0

            # Stop checks are based on the same account state used by the equity curve
            if position == 1:
                stop_out = False
                status_msg = ""

                if stop_loss_pct > 0:
                    hard_stop_price = entry_price * (1 - stop_loss_pct)
                    if price <= hard_stop_price:
                        stop_out = True
                        status_msg = 'Stop Loss'

                if not stop_out and trailing_stop_pct > 0:
                    max_price_since_entry = max(max_price_since_entry, price)
                    stop_price = max_price_since_entry * (1 - trailing_stop_pct)
                    if price <= stop_price:
                        stop_out = True
                        status_msg = 'Trailing Stop'

                if stop_out:
                    cash += holdings * price
                    holdings = 0.0
                    record_trade(date, price, status_msg)
                    position = 0
                    cooldown_bars = 5
                    equity_vals.append(cash)
                    continue

            # Signal processing
            if position == 0 and desired_signal > 0 and cooldown_bars == 0:
                position = 1
                entry_price = price
                entry_date = date
                entry_equity = current_equity(price)
                entry_signal = desired_signal
                max_price_since_entry = price

                invest_amt = cash * desired_signal
                holdings = invest_amt / price
                cash -= invest_amt

            elif position == 1 and desired_signal == 0:
                cash += holdings * price
                holdings = 0.0
                record_trade(date, price, 'Closed')
                position = 0

            equity_vals.append(current_equity(price))

        # Capture open position as mark-to-market, using TOTAL account equity
        # (cash + holdings), not holdings-only.
        if position == 1:
            current_price = float(prices.iloc[-1])
            record_trade(None, current_price, 'Open')
            equity_vals[-1] = current_equity(current_price)

        equity_curve_series = pd.Series(equity_vals, index=prices.index, dtype=float)
        benchmark_curve = initial_capital * (1 + returns).cumprod()
        strat_returns = equity_curve_series.pct_change().fillna(0.0)

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



def prepare_execution_prices_for_regime(prices_model, signals, raw_prices, use_weekly_execution=False):
    """
    For weekly regime signals, keep the weekly model decision, but move each weekly
    signal timestamp to the REAL latest raw trading candle inside that week.

    This fixes the confusing issue where an in-progress weekly bar is labeled as the
    upcoming Friday/Sunday even though the actual latest candle is Tuesday/Wednesday.
    Strategy logic is unchanged; only the execution/trade-log timestamp is mapped to
    the actual available raw date.
    """
    try:
        prices_model = pd.Series(prices_model).replace([np.inf, -np.inf], np.nan).dropna()
        signals = pd.Series(signals).reindex(prices_model.index).ffill().fillna(0).clip(0, 1)
        if not use_weekly_execution:
            return prices_model, signals

        raw = pd.Series(raw_prices).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return prices_model, signals

        raw.index = pd.to_datetime(raw.index)
        prices_model.index = pd.to_datetime(prices_model.index)
        signals.index = pd.to_datetime(signals.index)

        # Start close to the first weekly model bar, but include the whole first week so
        # a signal dated at a weekly period-end can execute on the actual raw date inside it.
        first_model_date = prices_model.index.min()
        start_floor = first_model_date - pd.Timedelta(days=7)
        raw = raw.loc[raw.index >= start_floor]
        if raw.empty:
            return prices_model, signals

        # Map each weekly signal date to the last actual raw trading date available in
        # that same weekly period. For a live/incomplete week, this becomes today's/latest
        # raw candle instead of the future Friday/Sunday period label.
        mapped_points = []
        for sig_date, sig_val in signals.items():
            sig_date = pd.Timestamp(sig_date)

            # Determine the weekly bucket using the signal label as period end.
            # Works for W-FRI labels and default W-SUN labels.
            bucket_start = sig_date - pd.Timedelta(days=6)
            bucket_raw = raw.loc[(raw.index >= bucket_start) & (raw.index <= sig_date)]

            # If the weekly label is in the future/current incomplete week, use all raw
            # data up to the latest available candle in that bucket.
            if bucket_raw.empty and sig_date > raw.index.max():
                bucket_raw = raw.loc[raw.index <= raw.index.max()]
                bucket_raw = bucket_raw.loc[bucket_raw.index >= bucket_start]

            if bucket_raw.empty:
                # Safe fallback: closest raw candle at or before the signal date.
                prior_raw = raw.loc[raw.index <= sig_date]
                if prior_raw.empty:
                    continue
                actual_date = prior_raw.index[-1]
            else:
                actual_date = bucket_raw.index[-1]

            mapped_points.append((actual_date, float(sig_val)))

        if not mapped_points:
            exec_signals = signals.reindex(raw.index, method='ffill').fillna(0).clip(0, 1)
            return raw, exec_signals

        mapped_sig = pd.Series(
            [v for _, v in mapped_points],
            index=pd.DatetimeIndex([d for d, _ in mapped_points]),
            dtype=float
        )
        # If multiple weekly labels map to the same latest raw candle, keep the newest value.
        mapped_sig = mapped_sig.groupby(mapped_sig.index).last().sort_index()

        raw = raw.loc[raw.index >= mapped_sig.index.min()]
        exec_signals = mapped_sig.reindex(raw.index).ffill().fillna(0).clip(0, 1)
        return raw, exec_signals
    except Exception:
        return prices_model, signals


def map_weekly_trade_log_dates_only(trades_df, raw_prices):
    """Display-only weekly date fix. Does not change returns, PnL, prices, stops, or metrics."""
    try:
        if trades_df is None or trades_df.empty:
            return trades_df
        raw = pd.Series(raw_prices).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return trades_df
        raw.index = pd.to_datetime(raw.index)

        def _map_one(dt):
            if pd.isna(dt) or dt == "Open":
                return dt
            d = pd.Timestamp(dt)
            bucket_start = d - pd.Timedelta(days=6)
            bucket = raw.loc[(raw.index >= bucket_start) & (raw.index <= d)]
            if bucket.empty and d > raw.index.max():
                bucket = raw.loc[(raw.index >= bucket_start) & (raw.index <= raw.index.max())]
            if bucket.empty:
                prior = raw.loc[raw.index <= d]
                return prior.index[-1] if not prior.empty else d
            return bucket.index[-1]

        out = trades_df.copy()
        if 'Entry Date' in out.columns:
            out['Entry Date'] = out['Entry Date'].apply(_map_one)
        if 'Exit Date' in out.columns:
            out['Exit Date'] = out['Exit Date'].apply(_map_one)
        return out
    except Exception:
        return trades_df



def apply_weekly_live_trigger_display_overrides(trades_df, raw_prices, signals, ticker="", strategy_name=""):
    """
    DISPLAY-ONLY helper for Weekly Regime Switching.

    Weekly bars are period-labeled, so a live weekly signal can appear with an upcoming
    Friday/week-end label. This helper remembers the first raw trading day/price when
    the app actually observes a live weekly BUY/SELL flip, then displays that date/price
    in the trade log for open/latest trades. It does not change backtest metrics,
    equity curve, or historical strategy logic.

    Important: it can only know the exact live trigger date if the app was running when
    the trigger first appeared. Otherwise it uses the first refresh where it sees the flip.
    """
    try:
        if trades_df is None or trades_df.empty:
            return trades_df
        raw = pd.Series(raw_prices).replace([np.inf, -np.inf], np.nan).dropna()
        sig = pd.Series(signals).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty or sig.empty:
            return trades_df

        raw.index = pd.to_datetime(raw.index)
        sig.index = pd.to_datetime(sig.index)
        latest_raw_date = raw.index[-1]
        latest_raw_price = float(raw.iloc[-1])
        latest_sig = float(sig.iloc[-1]) > 0

        state_key = f"weekly_live_trigger_state::{ticker}::{strategy_name}"
        state = st.session_state.get(state_key, {
            "last_sig": None,
            "open_entry_date": None,
            "open_entry_price": None,
            "last_exit_date": None,
            "last_exit_price": None,
        })

        last_sig = state.get("last_sig")

        # First time we see this ticker/strategy, initialize state but do not fake a past trigger.
        if last_sig is None:
            state["last_sig"] = latest_sig
            if latest_sig:
                # This is the first observed live long state. Use this refresh as the live trigger.
                state["open_entry_date"] = latest_raw_date
                state["open_entry_price"] = latest_raw_price
            st.session_state[state_key] = state
        else:
            # Live BUY flip observed while app is running.
            if (not bool(last_sig)) and latest_sig:
                state["open_entry_date"] = latest_raw_date
                state["open_entry_price"] = latest_raw_price
                state["last_exit_date"] = None
                state["last_exit_price"] = None
            # Live SELL flip observed while app is running.
            elif bool(last_sig) and (not latest_sig):
                state["last_exit_date"] = latest_raw_date
                state["last_exit_price"] = latest_raw_price
            state["last_sig"] = latest_sig
            st.session_state[state_key] = state

        out = trades_df.copy()

        # If current/latest trade is open, use remembered live entry date/price and latest raw price.
        if latest_sig and "Status" in out.columns:
            open_mask = out["Status"].astype(str).str.lower().eq("open")
            if open_mask.any():
                # newest open row after sorting may be row 0, but use the first open row found safely.
                i = out.index[open_mask][0]
                entry_date = state.get("open_entry_date")
                entry_price = state.get("open_entry_price")
                if entry_date is not None and entry_price is not None and float(entry_price) > 0:
                    out.loc[i, "Entry Date"] = pd.Timestamp(entry_date)
                    out.loc[i, "Buy Price"] = float(entry_price)
                    out.loc[i, "Sell Price"] = latest_raw_price
                    out.loc[i, "PnL (%)"] = ((latest_raw_price - float(entry_price)) / float(entry_price)) * 100.0

        # If a SELL flip was observed live, adjust the latest closed row display date/price only.
        # Metrics/returns remain unchanged.
        if (not latest_sig) and state.get("last_exit_date") is not None and "Status" in out.columns:
            closed_mask = out["Status"].astype(str).str.lower().isin(["closed", "stop loss", "trailing stop"])
            if closed_mask.any():
                i = out.index[closed_mask][0]
                out.loc[i, "Exit Date"] = pd.Timestamp(state.get("last_exit_date"))
                if state.get("last_exit_price") is not None and "Sell Price" in out.columns:
                    out.loc[i, "Sell Price"] = float(state.get("last_exit_price"))
                    bp = float(out.loc[i, "Buy Price"]) if "Buy Price" in out.columns else 0.0
                    if bp > 0:
                        out.loc[i, "PnL (%)"] = ((float(state.get("last_exit_price")) - bp) / bp) * 100.0

        return out
    except Exception:
        return trades_df


def make_stateful_position(entry_cond, exit_cond, index):
    """
    Converts entry/exit booleans into a 0/1 long-only position series.
    Entry = 1, Exit = 0, then forward-filled.
    """
    pos = pd.Series(np.nan, index=index, dtype=float)
    entry_cond = pd.Series(entry_cond, index=index).fillna(False)
    exit_cond = pd.Series(exit_cond, index=index).fillna(False)
    pos.loc[entry_cond] = 1.0
    pos.loc[exit_cond] = 0.0
    return pos.ffill().fillna(0.0).clip(lower=0, upper=1)


def enforce_min_hold_period(raw_signal, min_hold=1):
    """
    Applies a minimum hold period to a 0/1 signal so the strategy does not flip too quickly.
    This makes the Regime Switching Period method materially different from a simple probability rule.
    """
    raw_signal = pd.Series(raw_signal).fillna(0).astype(float).clip(0, 1)
    min_hold = max(1, int(min_hold))
    out = []
    position = 0.0
    bars_held = min_hold
    for desired in raw_signal.values:
        desired = 1.0 if desired > 0 else 0.0
        if desired != position and bars_held >= min_hold:
            position = desired
            bars_held = 1
        else:
            bars_held += 1
        out.append(position)
    return pd.Series(out, index=raw_signal.index, dtype=float)



def get_price_trend_override(prices_index, model_index, strat_prices):
    """
    Causal price trend override.
    Stays long when price structure is clearly bullish
    regardless of regime uncertainty.
    """
    try:
        px = pd.Series(strat_prices).replace([np.inf, -np.inf], np.nan).dropna()
        ema20 = px.ewm(span=20, adjust=False).mean()
        ema50 = px.ewm(span=50, adjust=False).mean()
        ema200 = px.ewm(span=200, adjust=False).mean()
        mom_20 = px.pct_change(20).fillna(0)
        mom_60 = px.pct_change(60).fillna(0)

        strong_trend = (
            (px > ema20) &
            (ema20 > ema50) &
            (ema50 > ema200) &
            (mom_20 > 0.02) &
            (mom_60 > 0.05)
        ).astype(float)

        return strong_trend.reindex(prices_index).ffill().fillna(0)
    except Exception:
        return pd.Series(0.0, index=prices_index)

def build_regime_backtest_signal(res_model, model_index, prices_index, n_regimes, signal_method, conviction=0.65, min_hold=1):
    """
    Converts Markov filtered probabilities into a tradable long/cash signal.
    Uses filtered probabilities only, so it is causal for closed bars.
    """
    probs_df = res_model.filtered_marginal_probabilities.copy()
    probs_df.index = model_index

    regime_means = []
    for i in range(n_regimes):
        if f'const[{i}]' in res_model.params:
            mean_val = res_model.params[f'const[{i}]']
        else:
            mean_val = res_model.params.get('const', 0.0)
        regime_means.append((i, float(mean_val)))
    bull_regime_idx = sorted(regime_means, key=lambda x: x[1], reverse=True)[0][0]

    bull_probs = probs_df.iloc[:, bull_regime_idx]
    dominant_regime = probs_df.idxmax(axis=1)

    if signal_method == "Regime Weighted Expected Return":
        expected_ret = pd.Series(0.0, index=model_index)
        for i in range(n_regimes):
            if f'const[{i}]' in res_model.params:
                mean_val = float(res_model.params[f'const[{i}]'])
            else:
                mean_val = float(res_model.params.get('const', 0.0))
            expected_ret += probs_df.iloc[:, i] * mean_val
        soft_conviction = float(conviction) * 0.70
        trend_participating = (
            (expected_ret > 0) & (bull_probs > soft_conviction)
        ) | (
            (bull_probs > float(conviction))
        )
        raw_signal = trend_participating.astype(float)
        context = {"expected_ret": expected_ret, "bull_probs": bull_probs, "dominant_regime": dominant_regime, "bull_regime_idx": bull_regime_idx}

    elif signal_method == "Regime Probability":
        # User-requested conviction threshold: bull probability must clear the threshold.
        raw_signal = (bull_probs > float(conviction)).astype(float)
        context = {"bull_probs": bull_probs, "dominant_regime": dominant_regime, "bull_regime_idx": bull_regime_idx}

    else:  # Regime Switching Period
        # Different from Regime Probability: require bull dominance + conviction, then enforce a minimum hold period.
        raw_signal = ((dominant_regime == bull_regime_idx) & (bull_probs > float(conviction))).astype(float)
        raw_signal = enforce_min_hold_period(raw_signal, min_hold=min_hold)
        context = {"bull_probs": bull_probs, "dominant_regime": dominant_regime, "bull_regime_idx": bull_regime_idx}

    signal = raw_signal.reindex(prices_index).ffill().fillna(0).clip(0, 1)
    return signal, context


@st.cache_data(ttl=900, show_spinner=False)
def walk_forward_regime_selection(prices, returns, n_regimes=2, switch_vol=True, switch_trend=True, train_window=126, forward_window=21, conviction=0.65, min_hold=3, initial_capital=10000.0, trailing_stop_pct=0.0, stop_loss_pct=0.0, confirmed_bar=True, use_strong_runner_override=True, activity_mode="Conservative", use_return_booster=True, return_booster_mode="Balanced"):
    """
    Walk-forward validation for the Regime Switching backtest.
    Each forward block fits only on the trailing training window, selects the best regime signal method
    and, when n_regimes='Auto', the best number of regimes from 2/3/4 on that training window.
    It then applies the latest confirmed exposure to the next unseen block.
    """
    prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    returns = pd.Series(returns).reindex(prices.index).replace([np.inf, -np.inf], np.nan).dropna()
    common_idx = prices.index.intersection(returns.index)
    prices = prices.loc[common_idx]
    returns = returns.loc[common_idx]

    train_window = int(train_window)
    forward_window = int(forward_window)

    # Practical WFO sizing for live/short windows:
    # Do NOT fall back to full-history just because the requested 252/21 window
    # is too large. Shrink the WFO windows so we still get a real out-of-sample
    # test. The effective values are returned to the UI.
    n_obs = len(prices)
    if n_obs < 15:
        return None

    train_window = max(8, train_window)
    forward_window = max(3, forward_window)

    if n_obs < train_window + forward_window:
        # About 65% train / 20% forward, with enough remaining bars to evaluate.
        train_window = max(8, int(n_obs * 0.65))
        forward_window = max(3, int(n_obs * 0.20))
        if train_window + forward_window > n_obs:
            forward_window = max(2, n_obs - train_window)
        if train_window < 8 or forward_window < 2 or train_window + forward_window > n_obs:
            return None

    base_methods = ["Regime Weighted Expected Return", "Regime Probability", "Regime Switching Period"]
    # Auto means WFO chooses the best regime count per forward block using only the training window.
    if isinstance(n_regimes, str) and n_regimes.lower() == "auto":
        regime_candidates = [2, 3, 4]
    else:
        regime_candidates = [int(n_regimes)]
    idx = prices.index
    wf_signal = pd.Series(np.nan, index=idx, dtype=float)
    rows = []
    sequence = []

    start = train_window
    period_no = 1
    while start < len(idx):
        train_idx = idx[start-train_window:start]
        test_idx = idx[start:min(start+forward_window, len(idx))]
        if len(test_idx) < 2:
            break

        train_returns = (returns.loc[train_idx].dropna() * 100)
        # n_regimes can be "Auto". Markov candidates need enough returns to converge,
        # but non-Markov fallback candidates like Strong Runner can still be tested on
        # short/noisy windows. Do NOT skip the whole WFO period just because Markov
        # lacks enough observations.
        has_markov_training_data = len(train_returns) >= 20
        train_returns = pd.Series(train_returns.values.flatten().astype(float), index=train_returns.index)

        train_scores = []
        markov_candidate_count = 0

        # 1) Markov regime candidates: WFO can choose 2, 3, or 4 regimes per stock/period.
        for n_candidate in regime_candidates:
            try:
                n_candidate = int(n_candidate)
                if (not has_markov_training_data) or len(train_returns) < max(20, n_candidate * 8):
                    continue
                res = fit_regime_model(train_returns, n_candidate, switch_vol, switch_trend, search_reps=3)
                if res is None:
                    continue

                for method in base_methods:
                    try:
                        sig_train, _ = build_regime_backtest_signal(
                            res, train_returns.index, prices.loc[train_idx].index,
                            n_candidate, method, conviction=float(conviction), min_hold=int(min_hold)
                        )
                        if confirmed_bar:
                            sig_train = sig_train.shift(1).ffill().fillna(0).clip(0, 1)
                        score = evaluate_strategy_candidate(
                            prices.loc[train_idx], sig_train,
                            initial_capital=initial_capital,
                            trailing_stop_pct=trailing_stop_pct,
                            stop_loss_pct=stop_loss_pct
                        )
                        if score is None:
                            continue
                        score["Institutional Score"] = risk_adjusted_candidate_score(score, activity_mode=str(activity_mode))
                        markov_candidate_count += 1
                        train_scores.append({
                            "method": method,
                            "n_regimes": n_candidate,
                            "score": score,
                            "signal": sig_train,
                            "activity_variant": "Base"
                        })

                        # Activity-aware variant: this is what makes Conservative/Balanced/Active truly different.
                        # It is scored on the training window only, then re-created causally for the forward window.
                        if str(activity_mode).lower() != "conservative":
                            act_sig_train = combine_regime_activity_signal(
                                sig_train, prices.loc[train_idx], mode=str(activity_mode)
                            )
                            if confirmed_bar:
                                act_sig_train = act_sig_train.shift(1).ffill().fillna(0).clip(0, 1)
                            act_score = evaluate_strategy_candidate(
                                prices.loc[train_idx], act_sig_train,
                                initial_capital=initial_capital,
                                trailing_stop_pct=trailing_stop_pct,
                                stop_loss_pct=stop_loss_pct
                            )
                            if act_score is not None:
                                act_score["Institutional Score"] = risk_adjusted_candidate_score(act_score, activity_mode=str(activity_mode))
                                train_scores.append({
                                    "method": f"{method} + {str(activity_mode)} Activity",
                                    "base_method": method,
                                    "n_regimes": n_candidate,
                                    "score": act_score,
                                    "signal": act_sig_train,
                                    "activity_variant": str(activity_mode)
                                })
                    except Exception:
                        continue
            except Exception:
                continue

        # 2) Strong runner candidate is not a Markov model, so add it once per training window.
        # This is still a true WFO candidate because it is scored only on the training window
        # and then tested only on the next unseen forward window. It is NOT a full-history fallback.
        if use_strong_runner_override:
            try:
                sig_train = strong_runner_trend_hold_signal(prices.loc[train_idx])
                if confirmed_bar:
                    sig_train = sig_train.shift(1).ffill().fillna(0).clip(0, 1)
                score = evaluate_strategy_candidate(
                    prices.loc[train_idx], sig_train,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_loss_pct=stop_loss_pct
                )
                if score is not None:
                    score["Institutional Score"] = risk_adjusted_candidate_score(score, activity_mode=str(activity_mode))
                    train_scores.append({
                        "method": "Strong Runner Trend Hold",
                        "n_regimes": "Trend",
                        "score": score,
                        "signal": sig_train,
                        "activity_variant": "Base"
                    })
            except Exception:
                pass

        # Pure activity candidate: useful when Markov is too defensive and does not trade enough.
        # This is still true WFO: it is chosen using training data only, then tested on unseen forward data.
        if str(activity_mode).lower() != "conservative":
            try:
                pulse_train = regime_activity_pulse_signal(prices.loc[train_idx], mode=str(activity_mode))
                if confirmed_bar:
                    pulse_train = pulse_train.shift(1).ffill().fillna(0).clip(0, 1)
                pulse_score = evaluate_strategy_candidate(
                    prices.loc[train_idx], pulse_train,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_loss_pct=stop_loss_pct
                )
                if pulse_score is not None:
                    pulse_score["Institutional Score"] = risk_adjusted_candidate_score(pulse_score, activity_mode=str(activity_mode))
                    train_scores.append({
                        "method": f"{str(activity_mode)} Trend Pulse",
                        "n_regimes": "Pulse",
                        "score": pulse_score,
                        "signal": pulse_train,
                        "activity_variant": str(activity_mode)
                    })
            except Exception:
                pass

        # 4) Benchmark-aware return booster candidate.
        # Goal: get closer to buy-and-hold during strong trends while still using a trend-break exit
        # to protect drawdown. It is scored on training only and tested forward unseen.
        if use_return_booster:
            try:
                booster_train = benchmark_aware_trend_participation_signal(
                    prices.loc[train_idx], mode=str(return_booster_mode)
                )
                if confirmed_bar:
                    booster_train = booster_train.shift(1).ffill().fillna(0).clip(0, 1)
                booster_score = evaluate_strategy_candidate(
                    prices.loc[train_idx], booster_train,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_loss_pct=stop_loss_pct
                )
                if booster_score is not None:
                    # Score wants benchmark participation, but still punishes drawdown.
                    booster_score["Institutional Score"] = risk_adjusted_candidate_score(
                        booster_score, activity_mode="ReturnBooster"
                    )
                    train_scores.append({
                        "method": f"Benchmark-Aware Return Booster ({str(return_booster_mode)})",
                        "n_regimes": "Booster",
                        "score": booster_score,
                        "signal": booster_train,
                        "activity_variant": "ReturnBooster"
                    })
            except Exception:
                pass

        if not train_scores:
            start += forward_window
            period_no += 1
            continue

        train_scores = sorted(
            train_scores,
            key=lambda x: x["score"].get("Institutional Score", -1e9),
            reverse=True
        )
        chosen = train_scores[0]
        chosen_method = chosen["method"]
        chosen_n_regimes = chosen.get("n_regimes", n_regimes)
        sequence.append(f"{chosen_method} | {chosen_n_regimes}R")

        # No-lookahead forward execution.
        # For Markov signals, carry the latest training exposure into the next unseen block.
        # For Strong Runner Trend Hold, calculate a causal trend-hold signal on train+test and use only the test portion.
        combo_px = prices.loc[idx[start-train_window:min(start+forward_window, len(idx))]]
        chosen_variant = str(chosen.get("activity_variant", "Base"))
        latest_exposure = float(chosen["signal"].iloc[-1]) if len(chosen["signal"]) else 0.0
        base_forward = pd.Series(latest_exposure, index=test_idx, dtype=float)

        if "Benchmark-Aware Return Booster" in chosen_method:
            test_signal = benchmark_aware_trend_participation_signal(
                combo_px, mode=str(return_booster_mode)
            ).reindex(test_idx).ffill().fillna(latest_exposure).clip(0, 1)
            if confirmed_bar:
                test_signal = test_signal.shift(1).ffill().fillna(latest_exposure).clip(0, 1)
        elif chosen_method == "Strong Runner Trend Hold":
            test_signal = strong_runner_trend_hold_signal(combo_px).reindex(test_idx).ffill().fillna(0).clip(0, 1)
            if confirmed_bar:
                test_signal = test_signal.shift(1).ffill().fillna(latest_exposure).clip(0, 1)
        elif "Trend Pulse" in chosen_method:
            test_signal = regime_activity_pulse_signal(combo_px, mode=chosen_variant).reindex(test_idx).ffill().fillna(latest_exposure).clip(0, 1)
            if confirmed_bar:
                test_signal = test_signal.shift(1).ffill().fillna(latest_exposure).clip(0, 1)
        elif chosen_variant.lower() != "base":
            test_signal = combine_regime_activity_signal(base_forward, combo_px, mode=chosen_variant).reindex(test_idx).ffill().fillna(latest_exposure).clip(0, 1)
            if confirmed_bar:
                test_signal = test_signal.shift(1).ffill().fillna(latest_exposure).clip(0, 1)
        else:
            test_signal = base_forward

        # IMPORTANT FIX:
        # Earlier the return booster was only a candidate. If the WFO selector chose a Markov
        # candidate, the final forward signal stayed unchanged, so enabling the booster could
        # produce the exact same return. When enabled, apply the booster as a controlled
        # participation overlay to the actual forward signal. This makes it genuinely affect
        # the trade log/metrics while still using only train+current forward data.
        if use_return_booster and "Benchmark-Aware Return Booster" not in chosen_method:
            try:
                booster_overlay = benchmark_aware_trend_participation_signal(
                    combo_px, mode=str(return_booster_mode)
                ).reindex(test_idx).ffill().fillna(0).clip(0, 1)
                if confirmed_bar:
                    booster_overlay = booster_overlay.shift(1).ffill().fillna(0).clip(0, 1)

                mode_l = str(return_booster_mode or "Balanced").lower()
                if mode_l == "aggressive":
                    # Get closer to buy-and-hold in strong trends.
                    test_signal = pd.concat([test_signal, booster_overlay], axis=1).max(axis=1)
                elif mode_l == "conservative":
                    # Only add exposure when the booster is very confident.
                    added = booster_overlay.where(booster_overlay >= 0.75, 0.0)
                    test_signal = pd.concat([test_signal, added], axis=1).max(axis=1)
                else:
                    # Balanced: participate more, but keep exposure slightly below pure hold.
                    added = (booster_overlay * 0.90).clip(0, 1)
                    test_signal = pd.concat([test_signal, added], axis=1).max(axis=1)
                test_signal = test_signal.reindex(test_idx).ffill().fillna(latest_exposure).clip(0, 1)
            except Exception:
                pass

        wf_signal.loc[test_idx] = test_signal

        test_score = evaluate_strategy_candidate(
            prices.loc[test_idx], test_signal,
            initial_capital=initial_capital,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct
        )
        rows.append({
            "Period": period_no,
            "Train Start": train_idx[0],
            "Train End": train_idx[-1],
            "Forward Start": test_idx[0],
            "Forward End": test_idx[-1],
            "Selected Method": chosen_method,
            "Selected Regimes": chosen_n_regimes,
            "Activity Mode": str(activity_mode),
            "Train Diff %": round(chosen["score"]["Difference %"], 2),
            "Forward Strategy %": round(test_score.get("Strategy Return %", np.nan), 2) if test_score else np.nan,
            "Forward Buy & Hold %": round(test_score.get("Buy & Hold Return %", np.nan), 2) if test_score else np.nan,
            "Forward Diff %": round(test_score.get("Difference %", np.nan), 2) if test_score else np.nan,
            "Forward Max DD %": round(test_score.get("Max DD %", np.nan), 2) if test_score else np.nan,
            "Forward Trades": int(test_score.get("Trades", 0)) if test_score else 0
        })

        start += forward_window
        period_no += 1

    if not rows:
        # Last-resort WFO-safe trend-capture path: still out-of-sample.
        # This prevents the tab from dropping into full-history research mode when
        # Markov convergence fails on short/noisy live windows.
        start = train_window
        period_no = 1
        while start < len(idx):
            train_idx = idx[start-train_window:start]
            test_idx = idx[start:min(start+forward_window, len(idx))]
            if len(test_idx) < 2:
                break
            combo_px = prices.loc[idx[start-train_window:min(start+forward_window, len(idx))]]
            test_signal = benchmark_aware_trend_participation_signal(
                combo_px, mode=str(return_booster_mode)
            ).reindex(test_idx).ffill().fillna(0).clip(0, 1)
            if confirmed_bar:
                test_signal = test_signal.shift(1).ffill().fillna(0).clip(0, 1)
            wf_signal.loc[test_idx] = test_signal
            test_score = evaluate_strategy_candidate(
                prices.loc[test_idx], test_signal,
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_stop_pct,
                stop_loss_pct=stop_loss_pct
            )
            rows.append({
                "Period": period_no,
                "Train Start": train_idx[0],
                "Train End": train_idx[-1],
                "Forward Start": test_idx[0],
                "Forward End": test_idx[-1],
                "Selected Method": f"Short-Window Trend Capture ({str(return_booster_mode)})",
                "Selected Regimes": "Trend",
                "Activity Mode": str(activity_mode),
                "Train Diff %": np.nan,
                "Forward Strategy %": round(test_score.get("Strategy Return %", np.nan), 2) if test_score else np.nan,
                "Forward Buy & Hold %": round(test_score.get("Buy & Hold Return %", np.nan), 2) if test_score else np.nan,
                "Forward Diff %": round(test_score.get("Difference %", np.nan), 2) if test_score else np.nan,
                "Forward Max DD %": round(test_score.get("Max DD %", np.nan), 2) if test_score else np.nan,
                "Forward Trades": int(test_score.get("Trades", 0)) if test_score else 0
            })
            sequence.append(f"Short-Window Trend Capture | Trend")
            start += forward_window
            period_no += 1

    if not rows:
        return None

    wf_signal = wf_signal.ffill().fillna(0).clip(0, 1)
    first_forward_start = rows[0]["Forward Start"]
    eval_prices = prices.loc[first_forward_start:]
    eval_signal = wf_signal.reindex(eval_prices.index).ffill().fillna(0).clip(0, 1)
    overall = evaluate_strategy_candidate(
        eval_prices, eval_signal,
        initial_capital=initial_capital,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct
    )
    rows_df = pd.DataFrame(rows)
    valid = rows_df["Forward Diff %"].dropna()
    win_rate = float((valid > 0).mean()) if len(valid) else 0.0
    changes = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    change_rate = changes / max(1, len(sequence)-1)
    avg_diff = float(valid.mean()) if len(valid) else 0.0
    stability_score = round(100 * (0.65 * win_rate + 0.20 * max(0, min(1, avg_diff / 10)) + 0.15 * (1 - change_rate)), 0)
    return {
        "signal": wf_signal,
        "rows": rows_df,
        "overall": overall,
        "first_forward_start": first_forward_start,
        "win_rate": win_rate,
        "changes": changes,
        "change_rate": change_rate,
        "avg_forward_diff": avg_diff,
        "stability_score": stability_score,
        "strategy_sequence": sequence,
        "effective_train_window": train_window,
        "effective_forward_window": forward_window
    }


def evaluate_strategy_candidate(prices, signals, initial_capital=10000.0, trailing_stop_pct=0.0, stop_loss_pct=0.0):
    """Fast score helper for strategy candidate ranking. Uses the same stop settings as the main backtest when supplied."""
    prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    signals = pd.Series(signals).reindex(prices.index).ffill().fillna(0).clip(lower=0, upper=1)
    if len(prices) < 5:
        return None
    res = BacktestEngine.run_strategy(prices, signals, initial_capital=initial_capital, trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct)
    strat_ret = (res['equity_curve'].iloc[-1] / initial_capital - 1) * 100
    bh_ret = (res['benchmark_curve'].iloc[-1] / initial_capital - 1) * 100
    rets = res['returns']
    dd = ((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min() * 100 if len(rets) else 0
    return {
        'Strategy Return %': strat_ret,
        'Buy & Hold Return %': bh_ret,
        'Difference %': strat_ret - bh_ret,
        'Max DD %': dd,
        'Trades': len(res['trades']),
        'signals': signals,
        'raw': res
    }


def buy_hold_return_pct(prices):
    """Buy-and-hold return over exactly the supplied price window."""
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 2 or px.iloc[0] == 0:
        return np.nan
    return (px.iloc[-1] / px.iloc[0] - 1) * 100


def apply_iv_sharpe_dd_guard(prices, base_signal, mode="Balanced", max_price_dd=0.18, vol_throttle=True, equity_dd_guard=True, max_equity_dd=0.20, equity_guard_action="Soft Throttle"):
    """
    Causal risk-control overlay for IV Proxy signals.
    Goal: improve Sharpe and reduce max drawdown without changing the underlying IV rule selection.

    It does not look into the future. It only uses price/volatility/trend information available
    up to each bar. Output can be fractional exposure: 1.0 full long, 0.5 reduced, 0.0 cash.

    Equity DD Guard watches the strategy equity curve itself. In Soft Throttle mode, it
    reduces exposure after account drawdown stress instead of locking the model fully in cash.
    In Hard Cash mode, it exits to cash until recovery. This is different from the price DD
    guard, which only watches the stock price drawdown from its peak.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    sig = pd.Series(base_signal).reindex(px.index).ffill().fillna(0.0).astype(float).clip(0.0, 1.0)
    if len(px) < 25:
        return sig

    mode_l = str(mode).lower()
    if mode_l == "strict":
        fast_span, slow_span, long_span = 10, 30, 100
        reduce_exposure = 0.35
        dd_mult = 0.75
        vol_mult = 1.35
    elif mode_l == "loose":
        fast_span, slow_span, long_span = 20, 50, 150
        reduce_exposure = 0.70
        dd_mult = 1.25
        vol_mult = 2.00
    else:  # Balanced
        fast_span, slow_span, long_span = 15, 40, 120
        reduce_exposure = 0.50
        dd_mult = 1.00
        vol_mult = 1.60

    ema_fast = px.ewm(span=fast_span, adjust=False).mean()
    ema_slow = px.ewm(span=slow_span, adjust=False).mean()
    ema_long = px.ewm(span=long_span, adjust=False).mean()
    ret = px.pct_change().fillna(0.0)
    vol_20 = ret.rolling(20, min_periods=5).std()
    vol_med = vol_20.rolling(100, min_periods=20).median()

    rolling_peak = px.cummax()
    price_dd = (px / rolling_peak - 1.0).fillna(0.0)
    dd_limit = -abs(float(max_price_dd)) * dd_mult

    healthy_trend = (px > ema_slow) & (ema_fast >= ema_slow)
    super_trend = healthy_trend & (px > ema_long) & (ema_slow.pct_change(5).fillna(0) > 0)
    weak_trend = (px < ema_slow) | (ema_fast < ema_slow)
    major_break = (price_dd <= dd_limit) & weak_trend

    if vol_throttle:
        vol_stress = (vol_20 > (vol_med * vol_mult)) & (ret < 0) & weak_trend
    else:
        vol_stress = pd.Series(False, index=px.index)

    out = pd.Series(0.0, index=px.index, dtype=float)
    current = 0.0
    for dt in px.index:
        desired = float(sig.loc[dt])
        if desired <= 0:
            current = 0.0
        else:
            if bool(major_break.loc[dt]) or bool(vol_stress.loc[dt]):
                current = 0.0
            elif bool(super_trend.loc[dt]):
                current = 1.0
            elif bool(healthy_trend.loc[dt]):
                current = max(reduce_exposure, desired * reduce_exposure)
            else:
                current = min(reduce_exposure, desired * reduce_exposure)
        out.loc[dt] = current

    out = out.ffill().fillna(0.0).clip(0.0, 1.0)

    # Account-level drawdown guard. This is intentionally applied AFTER the price/trend/vol guard.
    # Soft Throttle is the default because a hard lockout can protect drawdown but miss huge runners.
    # It simulates strategy equity using the prior bar's exposure, then reduces exposure when the
    # account is under drawdown stress. Hard Cash mode is still available for maximum protection.
    if bool(equity_dd_guard):
        max_equity_dd = abs(float(max_equity_dd))
        max_equity_dd = min(max(max_equity_dd, 0.01), 0.80)
        action_l = str(equity_guard_action).lower()
        final = pd.Series(0.0, index=px.index, dtype=float)
        px_ret = px.pct_change().fillna(0.0)
        equity = 1.0
        peak_equity = 1.0
        prev_exposure = 0.0
        locked = False
        recovery_count = 0
        for dt in px.index:
            # Mark-to-market first using yesterday's exposure. No future data is used.
            equity *= (1.0 + prev_exposure * float(px_ret.loc[dt]))
            peak_equity = max(peak_equity, equity)
            equity_dd = (equity / peak_equity) - 1.0 if peak_equity > 0 else 0.0

            desired = float(out.loc[dt])
            trend_recovered = bool(healthy_trend.loc[dt]) and bool(ema_fast.loc[dt] > ema_slow.loc[dt])
            trend_broken = bool(weak_trend.loc[dt]) and not bool(super_trend.loc[dt])

            if action_l.startswith("hard"):
                # Old behavior: full cash lockout after account DD breach.
                if equity_dd <= -max_equity_dd:
                    locked = True
                    recovery_count = 0
                if locked:
                    if trend_recovered and desired > 0:
                        recovery_count += 1
                    else:
                        recovery_count = 0
                    if recovery_count >= 3:
                        locked = False
                        peak_equity = max(peak_equity, equity)
                        final_exp = min(desired, reduce_exposure if not bool(super_trend.loc[dt]) else 1.0)
                    else:
                        final_exp = 0.0
                else:
                    final_exp = desired
            else:
                # New behavior: soft account DD throttle. This protects the account without
                # completely missing the next leg up in strong runners.
                stress_1 = equity_dd <= -max_equity_dd
                stress_2 = equity_dd <= -(max_equity_dd * 1.50)
                if stress_2 and trend_broken:
                    final_exp = 0.0
                elif stress_2:
                    final_exp = min(desired, reduce_exposure * 0.50)
                elif stress_1 and trend_broken:
                    final_exp = min(desired, reduce_exposure * 0.50)
                elif stress_1:
                    final_exp = min(desired, max(reduce_exposure, 0.50))
                else:
                    final_exp = desired

            final.loc[dt] = final_exp
            prev_exposure = final_exp
        out = final.ffill().fillna(0.0).clip(0.0, 1.0)

    return out.ffill().fillna(0.0).clip(0.0, 1.0)


def strong_runner_trend_hold_signal(prices, fast=20, slow=50, long=100):
    """
    Benchmark-aware trend-hold candidate.
    Purpose: when a stock is a true strong runner, avoid defensive regime models sitting in cash.
    Uses only price data available up to each bar.
    Long when price is above rising trend structure; cash only when trend breaks.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < max(slow, 30):
        return pd.Series(0.0, index=px.index)

    ema_fast = px.ewm(span=int(fast), adjust=False).mean()
    ema_slow = px.ewm(span=int(slow), adjust=False).mean()
    ema_long = px.ewm(span=int(long), adjust=False).mean() if len(px) >= int(long) else ema_slow

    # Momentum / relative trend strength from price itself. No future data.
    mom_21 = px.pct_change(21).fillna(0)
    mom_63 = px.pct_change(63).fillna(0)
    slow_slope = ema_slow.pct_change(10).fillna(0)

    strong_uptrend = (
        (px > ema_slow) &
        (ema_fast > ema_slow) &
        (ema_slow >= ema_long * 0.98) &
        (slow_slope > 0) &
        ((mom_21 > 0) | (mom_63 > 0))
    )

    # Exit only on a real trend break, not a tiny wiggle.
    trend_break = (px < ema_slow * 0.97) | ((ema_fast < ema_slow) & (slow_slope < 0))
    sig = make_stateful_position(strong_uptrend, trend_break, px.index)
    return sig.reindex(px.index).ffill().fillna(0).clip(0, 1)




def regime_activity_pulse_signal(prices, mode="Balanced"):
    """
    Causal trend-pulse signal used by Regime WFO activity modes.
    Balanced is moderate. Active is faster. No future data is used.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 15:
        return pd.Series(0.0, index=px.index)

    mode_l = str(mode or "Balanced").lower()
    if mode_l == "active":
        fast, slow, exit_span, mom_len, confirm = 4, 10, 16, 2, 1
    else:
        fast, slow, exit_span, mom_len, confirm = 8, 21, 34, 5, 2

    ema_fast = px.ewm(span=fast, adjust=False).mean()
    ema_slow = px.ewm(span=slow, adjust=False).mean()
    ema_exit = px.ewm(span=exit_span, adjust=False).mean()
    mom = px.pct_change(mom_len).fillna(0.0)
    slope = ema_slow.pct_change(max(2, mom_len)).fillna(0.0)

    entry = (ema_fast > ema_slow) & (px > ema_fast) & ((mom > 0) | (slope > 0))
    exit_ = (px < ema_exit) | ((ema_fast < ema_slow) & (mom < 0))

    if confirm > 1:
        entry = entry.astype(int).rolling(confirm, min_periods=confirm).sum().eq(confirm)

    return make_stateful_position(entry, exit_, px.index).reindex(px.index).ffill().fillna(0.0).clip(0, 1)


def combine_regime_activity_signal(base_signal, prices_window, mode="Balanced"):
    """
    Combines a regime WFO signal with a causal trend-pulse.
    Balanced: regime OR moderate trend pulse, but trend break can exit.
    Active: fast trend pulse controls exposure.
    """
    px = pd.Series(prices_window).replace([np.inf, -np.inf], np.nan).dropna()
    base = pd.Series(base_signal).reindex(px.index).ffill().fillna(0.0).clip(0, 1)
    mode_l = str(mode or "Balanced").lower()
    pulse = regime_activity_pulse_signal(px, mode=mode)

    if mode_l == "active":
        return pulse.reindex(px.index).ffill().fillna(0.0).clip(0, 1)

    # Balanced: be more willing to enter than pure regime, but still exit on clear pulse weakness.
    # This makes the output materially different while staying safer than Active.
    combined = ((base >= 0.5) | (pulse >= 0.5)).astype(float)
    return pd.Series(combined, index=px.index).ffill().fillna(0.0).clip(0, 1)



def full_benchmark_capture_signal(prices, mode="Full Benchmark Capture"):
    """
    Full-period trend-capture signal for the Regime tab.

    Purpose:
    Try to capture most of a strong stock's full buy-and-hold upside while still
    using causal risk brakes for major trend breaks. This is not pure WFO model
    selection; it is a benchmark-aware trend participation layer meant for the
    user's stated goal: closer to full benchmark with controlled drawdown.

    Uses only current/past price data.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 5:
        return pd.Series(0.0, index=px.index)

    mode_l = str(mode or "Full Benchmark Capture").lower()

    # More aggressive modes stay invested longer. The risk brake is intentionally
    # wide because the goal is to participate in large runners, not overtrade them.
    if "maximum" in mode_l or "chase" in mode_l:
        fast_span, guard_span, slow_span = 8, 34, 100
        trend_break_mult = 0.82
        trail_dd = 0.34
        reentry_mult = 0.97
        base_exposure = 1.00
    else:
        fast_span, guard_span, slow_span = 10, 50, 120
        trend_break_mult = 0.86
        trail_dd = 0.28
        reentry_mult = 0.99
        base_exposure = 1.00

    ema_fast = px.ewm(span=fast_span, adjust=False).mean()
    ema_guard = px.ewm(span=guard_span, adjust=False).mean()
    ema_slow = px.ewm(span=slow_span, adjust=False).mean() if len(px) >= slow_span else ema_guard

    mom_10 = px.pct_change(10).fillna(0.0)
    mom_21 = px.pct_change(21).fillna(0.0)
    guard_slope = ema_guard.pct_change(10).fillna(0.0)
    roll_peak = px.cummax()
    dd_from_peak = (px / roll_peak - 1.0).fillna(0.0)

    # Enter early in a constructive trend. This is intentionally broad so the
    # strategy does not sit in cash while the benchmark makes the big move.
    enter = (
        (px >= ema_guard * reentry_mult) |
        ((ema_fast >= ema_guard * 0.985) & ((mom_10 > 0) | (mom_21 > 0) | (guard_slope > 0)))
    )

    # Exit only on serious structure damage. This is the drawdown guard.
    exit_ = (
        (px < ema_guard * trend_break_mult) |
        ((ema_fast < ema_guard * 0.94) & (guard_slope < -0.02)) |
        (dd_from_peak < -trail_dd)
    )

    sig = make_stateful_position(enter, exit_, px.index)

    # If the selected period is already in a strong trend at the first bar, start
    # participating from the beginning instead of waiting for a cross that may have
    # happened before the selected window.
    if len(sig) > 0 and sig.iloc[0] == 0:
        if bool((px.iloc[0] >= ema_guard.iloc[0] * 0.98) or (ema_fast.iloc[0] >= ema_guard.iloc[0] * 0.98)):
            sig.iloc[0] = 1.0
            sig = sig.ffill().fillna(0.0)

    return (sig * base_exposure).reindex(px.index).ffill().fillna(0.0).clip(0, 1)


def benchmark_aware_trend_participation_signal(prices, mode="Balanced"):
    """
    Benchmark-aware trend participation candidate for Regime WFO.

    Goal: get closer to buy-and-hold during strong runners WITHOUT simply buying blindly.
    It stays invested while trend structure is healthy and exits only on a meaningful
    trend break or drawdown break. Uses only current/past price data. No future data.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 5:
        return pd.Series(0.0, index=px.index)

    mode_l = str(mode or "Balanced").lower()

    if "optimized" in mode_l:
        # Speed fix: on shorter rolling WFO chunks, use the fast trend-capture
        # version instead of re-running the optimizer on every block. The full
        # optimizer is still available on longer full-period series.
        if len(px) < 400:
            return full_benchmark_capture_signal(px, mode="Full Benchmark Capture")
        return optimized_full_capture_signal(px)
    if "full benchmark" in mode_l or "maximum" in mode_l:
        return full_benchmark_capture_signal(px, mode=mode)

    # These settings intentionally make Balanced/Aggressive much more trend-capturing
    # than the older version. The old version was too defensive and kept missing
    # monster runners, so returns stayed far below benchmark.
    if mode_l in ["benchmark chase", "benchmark_chase", "chase"]:
        base_exposure = 1.00
        strong_exposure = 1.00
        break_mult = 0.88       # wider trend break = hold winners longer
        dd_exit = 0.28
        fast, mid, slow = 8, 21, 50
        require_slope = False
    elif mode_l == "aggressive":
        base_exposure = 0.95
        strong_exposure = 1.00
        break_mult = 0.90
        dd_exit = 0.24
        fast, mid, slow = 10, 25, 60
        require_slope = False
    elif mode_l == "conservative":
        base_exposure = 0.55
        strong_exposure = 0.80
        break_mult = 0.96
        dd_exit = 0.16
        fast, mid, slow = 20, 50, 100
        require_slope = True
    else:  # Balanced: now a real trend-capture mode, not a tiny overlay
        base_exposure = 0.90
        strong_exposure = 1.00
        break_mult = 0.92
        dd_exit = 0.20
        fast, mid, slow = 12, 30, 75
        require_slope = False

    ema_fast = px.ewm(span=fast, adjust=False).mean()
    ema_mid = px.ewm(span=mid, adjust=False).mean()
    ema_slow = px.ewm(span=slow, adjust=False).mean() if len(px) >= slow else ema_mid

    mom_5 = px.pct_change(5).fillna(0.0)
    mom_10 = px.pct_change(10).fillna(0.0)
    mom_21 = px.pct_change(21).fillna(0.0)
    mid_slope = ema_mid.pct_change(8).fillna(0.0)
    rolling_peak = px.cummax()
    drawdown_from_peak = (px / rolling_peak - 1.0).fillna(0.0)

    # Broader entry logic: if price is above mid-trend and momentum is not broken, participate.
    # This is the key change that makes the booster capable of getting closer to benchmark.
    healthy_trend = (px > ema_mid) & (ema_fast >= ema_mid * 0.985) & ((mom_5 > -0.03) | (mom_10 > 0) | (mom_21 > 0))
    if require_slope:
        healthy_trend = healthy_trend & (mid_slope > 0)

    strong_trend = healthy_trend & (px > ema_fast) & ((mom_10 > 0.02) | (mom_21 > 0.04) | (ema_mid >= ema_slow * 0.99))

    # Exit only when structure really breaks. This protects drawdown but avoids early exits.
    trend_break = (px < ema_mid * break_mult) | ((ema_fast < ema_mid * 0.975) & (mid_slope < -0.015)) | (drawdown_from_peak < -dd_exit)

    out = []
    exposure = 0.0
    for dt in px.index:
        if bool(trend_break.loc[dt]):
            exposure = 0.0
        elif bool(strong_trend.loc[dt]):
            exposure = max(exposure, strong_exposure)
        elif bool(healthy_trend.loc[dt]):
            exposure = max(exposure, base_exposure)
        else:
            # During normal pauses, reduce slowly instead of dumping the position.
            exposure = exposure * 0.90 if exposure > 0 else 0.0
            if exposure < 0.35:
                exposure = 0.0
        out.append(float(np.clip(exposure, 0.0, 1.0)))

    return pd.Series(out, index=px.index, dtype=float).ffill().fillna(0.0).clip(0, 1)



def _trend_capture_candidate_signal(prices, fast=10, guard=34, slow=100, break_mult=0.96, trail_dd=0.18, reentry_mult=1.00, exposure=1.0):
    """
    Causal trend-capture candidate used by Optimized Full Capture.
    It participates in strong trends but exits when trend structure or drawdown breaks.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 5:
        return pd.Series(0.0, index=px.index)

    ema_fast = px.ewm(span=int(fast), adjust=False).mean()
    ema_guard = px.ewm(span=int(guard), adjust=False).mean()
    ema_slow = px.ewm(span=int(slow), adjust=False).mean() if len(px) >= int(slow) else ema_guard
    mom_5 = px.pct_change(5).fillna(0.0)
    mom_10 = px.pct_change(10).fillna(0.0)
    mom_21 = px.pct_change(21).fillna(0.0)
    guard_slope = ema_guard.pct_change(8).fillna(0.0)
    rolling_peak = px.cummax()
    drawdown_from_peak = (px / rolling_peak - 1.0).fillna(0.0)

    enter = (
        (px >= ema_guard * float(reentry_mult)) |
        ((ema_fast >= ema_guard * 0.985) & ((mom_5 > 0) | (mom_10 > 0) | (mom_21 > 0))) |
        ((px >= ema_slow * 1.01) & (guard_slope >= -0.005))
    )
    exit_ = (
        (px < ema_guard * float(break_mult)) |
        ((ema_fast < ema_guard * 0.975) & (guard_slope < -0.012)) |
        (drawdown_from_peak < -float(trail_dd))
    )
    sig = make_stateful_position(enter, exit_, px.index)

    # If the selected window starts while already in an uptrend, participate immediately.
    if len(sig) > 0 and sig.iloc[0] == 0:
        if bool((px.iloc[0] >= ema_guard.iloc[0] * 0.98) or (ema_fast.iloc[0] >= ema_guard.iloc[0] * 0.98)):
            sig.iloc[0] = 1.0
            sig = sig.ffill().fillna(0.0)

    return (sig * float(exposure)).reindex(px.index).ffill().fillna(0.0).clip(0, 1)


@st.cache_data(ttl=900, show_spinner=False)
def optimized_full_capture_signal(prices, initial_capital=10000.0):
    """
    Full-benchmark capture optimizer.

    Goal:
    Get as close as possible to buy-and-hold return while rejecting candidates that
    create ugly drawdowns or weak Sharpe. Every candidate is causal; the optimizer
    only chooses among rule templates, it does not use future bars inside each signal.

    Honest note:
    This is a full-period optimization layer, so use it as a benchmark-capture mode,
    not as pure WFO validation.
    """
    px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(px) < 20:
        return pd.Series(1.0, index=px.index) if len(px) else pd.Series(dtype=float)

    bh_return = buy_hold_return_pct(px)
    if pd.isna(bh_return):
        bh_return = 0.0

    # Full scan grid restored. Do NOT reduce these combinations because
    # the optimizer needs the same search space as the original behavior.
    # Speed is handled by Streamlit caching above, not by changing logic.
    fast_grid = [6, 8, 10, 13, 21]
    guard_grid = [13, 21, 34, 50]
    break_grid = [0.93, 0.95, 0.97]
    trail_grid = [0.12, 0.16, 0.20, 0.25]
    reentry_grid = [0.98, 1.00, 1.02]
    exposure_grid = [0.85, 1.00]

    best_score = -1e18
    best_sig = None
    best_meta = None

    for fast in fast_grid:
        for guard in guard_grid:
            if fast >= guard:
                continue
            slow = max(75, guard * 3)
            for break_mult in break_grid:
                for trail_dd in trail_grid:
                    for reentry_mult in reentry_grid:
                        for exposure in exposure_grid:
                            try:
                                sig = _trend_capture_candidate_signal(
                                    px, fast=fast, guard=guard, slow=slow,
                                    break_mult=break_mult, trail_dd=trail_dd,
                                    reentry_mult=reentry_mult, exposure=exposure
                                )
                                bt = BacktestEngine.run_strategy(px, sig, initial_capital=initial_capital)
                                eq = bt.get('equity_curve', pd.Series(dtype=float))
                                rets = bt.get('returns', pd.Series(dtype=float))
                                if len(eq) < 2 or len(rets) < 2:
                                    continue
                                strat_return = (eq.iloc[-1] / initial_capital - 1.0) * 100.0
                                metrics = BacktestEngine.calculate_metrics(rets)
                                sharpe = float(metrics.get('Sharpe Ratio', 0.0) or 0.0)
                                max_dd = abs(float(metrics.get('Max Drawdown', 0.0) or 0.0)) * 100.0
                                trades = bt.get('trades', pd.DataFrame())
                                trade_count = 0 if trades is None or trades.empty else len(trades)
                                avg_exposure = float(sig.mean()) if len(sig) else 0.0

                                # Designed objective: high capture, good Sharpe, controlled drawdown.
                                # We penalize drawdown hard after ~22%, but we do not demand tiny DD
                                # because that usually misses the full benchmark move.
                                gap_to_bh = max(0.0, float(bh_return) - float(strat_return))
                                dd_penalty = max(0.0, max_dd - 22.0) * 5.0
                                dead_signal_penalty = 35.0 if avg_exposure < 0.25 else 0.0
                                overtrade_penalty = max(0, trade_count - max(8, len(px)//18)) * 1.5
                                score = (
                                    strat_return
                                    - 0.28 * gap_to_bh
                                    + 10.0 * sharpe
                                    - dd_penalty
                                    - dead_signal_penalty
                                    - overtrade_penalty
                                )

                                if score > best_score:
                                    best_score = score
                                    best_sig = sig
                                    best_meta = (strat_return, sharpe, max_dd, trade_count, avg_exposure)
                            except Exception:
                                continue

    if best_sig is None:
        # Fallback is not blank: hold if price is above a simple trend, otherwise cash.
        fallback = _trend_capture_candidate_signal(px, fast=10, guard=34, slow=100, break_mult=0.95, trail_dd=0.18)
        return fallback.reindex(px.index).ffill().fillna(0.0).clip(0, 1)

    # Store lightweight metadata in session for optional display; safe if Streamlit state is unavailable.
    try:
        st.session_state['optimized_full_capture_meta'] = {
            'Return %': float(best_meta[0]),
            'Sharpe': float(best_meta[1]),
            'Max DD %': float(best_meta[2]),
            'Trades': int(best_meta[3]),
            'Avg Exposure %': float(best_meta[4] * 100.0),
        }
    except Exception:
        pass

    return pd.Series(best_sig, index=px.index).ffill().fillna(0.0).clip(0, 1)


def risk_adjusted_candidate_score(score, benchmark_bias=0.15, activity_mode="Conservative"):
    """
    Institutional-style scoring: return matters, but drawdown and benchmark underperformance are penalized.
    benchmark_bias rewards candidates that stay closer to buy-and-hold during strong runners.
    """
    if score is None:
        return -1e9
    ret = float(score.get('Strategy Return %', 0.0))
    diff = float(score.get('Difference %', 0.0))
    dd = abs(float(score.get('Max DD %', 0.0)))
    trades = int(score.get('Trades', 0))
    mode_l = str(activity_mode or "Conservative").lower()

    # Conservative prefers clean, fewer-trade behavior.
    # Balanced wants enough participation to challenge buy/hold.
    # Active penalizes dead/no-trade signals much harder.
    if mode_l == "returnbooster":
        # Designed for your goal: challenge benchmark return, but keep drawdown controlled.
        trade_penalty = 6.0 if trades < 1 else 0.0
        drawdown_penalty = 0.55 * dd if dd > 18 else 0.35 * dd
        return (0.55 * ret) + (0.55 * diff) - drawdown_penalty - trade_penalty
    elif mode_l == "active":
        trade_penalty = 18.0 if trades < 2 else (8.0 if trades < 4 else 0.0)
        activity_bonus = min(trades, 8) * 1.25
        return (0.45 * ret) + (0.45 * diff) - (0.28 * dd) - trade_penalty + activity_bonus
    elif mode_l == "balanced":
        trade_penalty = 10.0 if trades < 2 else 0.0
        activity_bonus = min(trades, 5) * 0.65
        return (0.50 * ret) + (0.42 * diff) - (0.28 * dd) - trade_penalty + activity_bonus
    else:
        trade_penalty = 5.0 if trades < 2 else 0.0
        return (0.55 * ret) + (0.35 * diff) - (0.25 * dd) - trade_penalty + (benchmark_bias * max(ret, 0))

def walk_forward_strategy_selection(prices, candidates, train_window=126, forward_window=21, initial_capital=10000.0, confirmed_bar=True, trailing_stop_pct=0.0, stop_loss_pct=0.0):
    """
    Walk-forward validation for adaptive strategy choosers.
    Chooses the best candidate using ONLY the trailing training window, then applies that candidate
    to the next unseen forward window. This helps reduce full-period curve fitting.
    """
    prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(prices) < int(train_window) + max(5, int(forward_window)):
        return None

    train_window = int(train_window)
    forward_window = int(forward_window)
    idx = prices.index

    wf_signal = pd.Series(np.nan, index=idx, dtype=float)
    wf_rows = []
    strategy_sequence = []

    start = train_window
    period_no = 1
    while start < len(prices):
        train_idx = idx[start - train_window:start]
        test_idx = idx[start:min(start + forward_window, len(prices))]
        if len(test_idx) < 2:
            break

        train_prices = prices.loc[train_idx]
        test_prices = prices.loc[test_idx]

        train_scores = []
        for name, logic, sig in candidates:
            sig_series = pd.Series(sig).reindex(idx).ffill().fillna(0).clip(0, 1)
            train_sig = sig_series.reindex(train_idx).ffill().fillna(0)
            score_res = evaluate_strategy_candidate(train_prices, train_sig, initial_capital=initial_capital, trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct)
            if score_res is None:
                continue
            train_scores.append({
                "name": name,
                "logic": logic,
                "signal": sig_series,
                "train_diff": score_res.get("Difference %", np.nan),
                "train_return": score_res.get("Strategy Return %", np.nan),
                "train_dd": score_res.get("Max DD %", np.nan),
                "train_trades": score_res.get("Trades", 0)
            })

        # 4) Benchmark-aware return booster candidate.
        # Goal: get closer to buy-and-hold during strong trends while still using a trend-break exit
        # to protect drawdown. It is scored on training only and tested forward unseen.
        if use_return_booster:
            try:
                booster_train = benchmark_aware_trend_participation_signal(
                    prices.loc[train_idx], mode=str(return_booster_mode)
                )
                if confirmed_bar:
                    booster_train = booster_train.shift(1).ffill().fillna(0).clip(0, 1)
                booster_score = evaluate_strategy_candidate(
                    prices.loc[train_idx], booster_train,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_loss_pct=stop_loss_pct
                )
                if booster_score is not None:
                    # Score wants benchmark participation, but still punishes drawdown.
                    booster_score["Institutional Score"] = risk_adjusted_candidate_score(
                        booster_score, activity_mode="ReturnBooster"
                    )
                    train_scores.append({
                        "method": f"Benchmark-Aware Return Booster ({str(return_booster_mode)})",
                        "n_regimes": "Booster",
                        "score": booster_score,
                        "signal": booster_train,
                        "activity_variant": "ReturnBooster"
                    })
            except Exception:
                pass

        if not train_scores:
            start += forward_window
            period_no += 1
            continue

        # Pick best using training-only result. Difference vs buy/hold is primary; return is tie-breaker.
        train_scores = sorted(train_scores, key=lambda x: (x["train_diff"], x["train_return"], -abs(x["train_dd"])), reverse=True)
        chosen = train_scores[0]

        chosen_sig_full = chosen["signal"].copy()
        # Confirmed-bar mode: today's position uses the prior closed bar's signal.
        if confirmed_bar:
            exec_sig_full = chosen_sig_full.shift(1).ffill().fillna(0).clip(0, 1)
        else:
            exec_sig_full = chosen_sig_full.ffill().fillna(0).clip(0, 1)

        test_sig = exec_sig_full.reindex(test_idx).ffill().fillna(0).clip(0, 1)
        wf_signal.loc[test_idx] = test_sig

        test_score = evaluate_strategy_candidate(test_prices, test_sig, initial_capital=initial_capital, trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct)
        bh_return = np.nan
        strat_return = np.nan
        diff_return = np.nan
        trades = 0
        max_dd = np.nan
        if test_score is not None:
            strat_return = test_score.get("Strategy Return %", np.nan)
            bh_return = test_score.get("Buy & Hold Return %", np.nan)
            diff_return = test_score.get("Difference %", np.nan)
            trades = test_score.get("Trades", 0)
            max_dd = test_score.get("Max DD %", np.nan)

        wf_rows.append({
            "Period": period_no,
            "Train Start": train_idx[0],
            "Train End": train_idx[-1],
            "Forward Start": test_idx[0],
            "Forward End": test_idx[-1],
            "Selected Rule": chosen["name"],
            "Train Diff %": round(chosen["train_diff"], 2) if pd.notna(chosen["train_diff"]) else np.nan,
            "Forward Strategy %": round(strat_return, 2) if pd.notna(strat_return) else np.nan,
            "Forward Buy & Hold %": round(bh_return, 2) if pd.notna(bh_return) else np.nan,
            "Forward Diff %": round(diff_return, 2) if pd.notna(diff_return) else np.nan,
            "Forward Max DD %": round(max_dd, 2) if pd.notna(max_dd) else np.nan,
            "Forward Trades": trades
        })
        strategy_sequence.append(chosen["name"])

        start += forward_window
        period_no += 1

    wf_signal = wf_signal.ffill().fillna(0).clip(0, 1)
    if len(wf_rows) == 0:
        return None

    # Score the stitched walk-forward signal only from the first unseen forward segment onward.
    first_forward_start = wf_rows[0]["Forward Start"] if wf_rows else idx[0]
    wf_eval_index = prices.loc[first_forward_start:].index
    wf_prices = prices.loc[wf_eval_index]
    wf_sig_active = wf_signal.reindex(wf_eval_index).ffill().fillna(0).clip(0, 1)
    overall = evaluate_strategy_candidate(wf_prices, wf_sig_active, initial_capital=initial_capital, trailing_stop_pct=trailing_stop_pct, stop_loss_pct=stop_loss_pct)

    rows_df = pd.DataFrame(wf_rows)
    wins = int((rows_df["Forward Diff %"] > 0).sum()) if "Forward Diff %" in rows_df else 0
    valid_periods = int(rows_df["Forward Diff %"].notna().sum()) if "Forward Diff %" in rows_df else len(rows_df)
    changes = sum(1 for a, b in zip(strategy_sequence, strategy_sequence[1:]) if a != b)
    change_rate = changes / max(1, len(strategy_sequence) - 1)
    win_rate = wins / max(1, valid_periods)
    avg_forward_diff = float(rows_df["Forward Diff %"].dropna().mean()) if "Forward Diff %" in rows_df and rows_df["Forward Diff %"].notna().any() else 0.0

    # Stability score rewards out-of-sample wins and penalizes excessive rule switching.
    stability_score = round(100 * (0.65 * win_rate + 0.20 * max(0, min(1, avg_forward_diff / 10)) + 0.15 * (1 - change_rate)), 0)

    return {
        "signal": wf_signal,
        "rows": rows_df,
        "overall": overall,
        "win_rate": win_rate,
        "changes": changes,
        "change_rate": change_rate,
        "avg_forward_diff": avg_forward_diff,
        "stability_score": stability_score,
        "strategy_sequence": strategy_sequence,
        "confirmed_bar": confirmed_bar
    }


def display_adaptive_strategy_lab(title, prices, candidates, initial_capital=10000.0, file_prefix="Adaptive_Strategy"):
    """
    Tests multiple long/cash rules and displays both:
    1) Walk-forward selected result (primary, more realistic)
    2) Full-history adaptive ranking (research/reference only)
    """
    prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()

    st.write(f"#### 🚀 {title}: Adaptive Rule Optimizer")
    st.caption(
        "Walk-forward mode trains on past data, chooses the best rule, then tests it on the next unseen window. "
        "This is more realistic than picking the best rule from the full chart."
    )

    with st.expander(f"⚙️ {title} Walk-Forward Settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        enable_wfo = c1.checkbox(f"Enable {title} WFO", value=True, key=f"{file_prefix}_enable_wfo")
        use_wfo_primary = c2.checkbox(f"Use WFO as main result", value=True, key=f"{file_prefix}_use_wfo_primary")
        confirmed_bar = c3.checkbox(f"Confirmed-bar execution", value=True, key=f"{file_prefix}_confirmed_bar")
        show_insample = c4.checkbox(f"Show full-history ranking", value=True, key=f"{file_prefix}_show_insample")

        c5, c6 = st.columns(2)
        wf_train_bars = c5.number_input(
            f"{title} WFO train bars",
            min_value=30, max_value=1000, value=126, step=21,
            key=f"{file_prefix}_wf_train_bars",
            help="How many past bars the model uses to choose the best rule."
        )
        wf_forward_bars = c6.number_input(
            f"{title} WFO forward bars",
            min_value=5, max_value=252, value=21, step=5,
            key=f"{file_prefix}_wf_forward_bars",
            help="How many unseen future bars the chosen rule is tested on before re-optimizing."
        )

    rows = []
    scored = []
    for name, logic_text, sig in candidates:
        score = evaluate_strategy_candidate(prices, sig, initial_capital=initial_capital)
        if score is None:
            continue
        scored.append((name, logic_text, score))
        rows.append({
            'Rule': name,
            'Strategy Return %': round(score['Strategy Return %'], 2),
            'Buy & Hold Return %': round(score['Buy & Hold Return %'], 2),
            'Difference %': round(score['Difference %'], 2),
            'Max DD %': round(score['Max DD %'], 2),
            'Trades': score['Trades']
        })

    if not scored:
        st.info(f"Not enough data to run {title} adaptive strategy lab.")
        return None

    rank_df = pd.DataFrame(rows).sort_values(['Difference %', 'Strategy Return %'], ascending=False)
    best_rule = rank_df.iloc[0]['Rule']
    best = next(item for item in scored if item[0] == best_rule)
    best_name, best_logic, best_score = best

    wf_result = None
    if enable_wfo:
        wf_result = walk_forward_strategy_selection(
            prices,
            candidates,
            train_window=int(wf_train_bars),
            forward_window=int(wf_forward_bars),
            initial_capital=initial_capital,
            confirmed_bar=confirmed_bar
        )

        st.write(f"#### 🧭 {title} Walk-Forward Result")
        if wf_result is None or wf_result.get('overall') is None:
            st.warning(f"Not enough data to run {title} walk-forward validation. Falling back to full-history adaptive winner.")
        else:
            overall = wf_result['overall']
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            full_bh_pct = buy_hold_return_pct(prices)
            c1.metric("WFO Strategy Return", f"{overall['Strategy Return %']:.2f}%")
            c2.metric("WFO Test Benchmark", f"{overall['Buy & Hold Return %']:.2f}%", help="Buy & hold only over the out-of-sample walk-forward test window.")
            c3.metric("Full Benchmark", f"{full_bh_pct:.2f}%" if pd.notna(full_bh_pct) else "N/A", help="Buy & hold over the full selected chart period. This is reference only when WFO is enabled.")
            c4.metric("WFO Difference", f"{overall['Difference %']:+.2f}%")
            c5.metric("WFO Win Rate", f"{wf_result['win_rate']*100:.0f}%")
            c6.metric("Stability", f"{wf_result['stability_score']:.0f}/100")

            if overall['Difference %'] > 0 and wf_result['stability_score'] >= 60:
                st.success(f"{title} WFO is positive and reasonably stable. This is stronger than only using the in-sample winner.")
            elif overall['Difference %'] > 0:
                st.warning(f"{title} WFO beat buy & hold, but stability is not very strong. Use confirmation.")
            else:
                st.warning(f"{title} WFO did not beat buy & hold on the unseen forward windows. Treat the adaptive winner as research, not a strong edge.")

            wf_rows = wf_result['rows'].copy()
            if not wf_rows.empty:
                for col in ["Train Start", "Train End", "Forward Start", "Forward End"]:
                    wf_rows[col] = pd.to_datetime(wf_rows[col]).dt.date
                st.dataframe(wf_rows.sort_values("Period", ascending=False), use_container_width=True)
                st.download_button(
                    f"📥 Download {title} WFO Periods",
                    wf_rows.to_csv(index=False),
                    file_name=f"{file_prefix}_WalkForward_Periods_{TICKER}.csv",
                    mime="text/csv",
                    key=f"{file_prefix}_download_wfo_periods"
                )

    if show_insample:
        st.write(f"#### 📌 {title} Full-History Ranking / Research Reference")
        st.caption("This ranking uses the whole selected chart. It is useful for research, but it can overfit more than WFO.")
        st.dataframe(rank_df, use_container_width=True)

        if best_score['Difference %'] > 0:
            st.success(f"Full-history best rule beat buy & hold by **{best_score['Difference %']:.2f}%**: **{best_name}**")
        else:
            st.warning(f"Full-history best rule still did not beat buy & hold. Best rule: **{best_name}**.")
        st.info(f"Full-history best rule logic: {best_logic}")

    if enable_wfo and use_wfo_primary and wf_result is not None and wf_result.get('overall') is not None:
        first_forward_start = wf_result['rows']['Forward Start'].iloc[0]
        exec_prices = prices.loc[first_forward_start:]
        exec_signal = wf_result['signal'].reindex(exec_prices.index).ffill().fillna(0).clip(0, 1)
        return display_strategy_vs_buyhold_backtest(
            f"{title} WFO-Selected Strategy",
            exec_prices,
            exec_signal,
            initial_capital=initial_capital,
            file_prefix=file_prefix,
            full_period_prices=prices,
            benchmark_label="WFO Test Benchmark"
        )

    return display_strategy_vs_buyhold_backtest(
        best_name,
        prices,
        best_score['signals'],
        initial_capital=initial_capital,
        file_prefix=file_prefix
    )

def display_strategy_vs_buyhold_backtest(title, prices, signals, initial_capital=10000.0, file_prefix="Strategy", full_period_prices=None, benchmark_label="Buy & Hold Return"):
    """
    Shared Streamlit display for small strategy checks inside analytical tabs.
    Shows Strategy %, matching-window benchmark %, optional full-period benchmark %, equity curve, and detailed trade log.
    """
    try:
        prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
        signals = pd.Series(signals).reindex(prices.index).ffill().fillna(0).clip(lower=0, upper=1)
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]

        if len(prices) < 5:
            st.info(f"Not enough data to run {title} backtest.")
            return None

        bt_results = BacktestEngine.run_strategy(prices, signals, initial_capital=initial_capital)
        strat_ret_pct = (bt_results['equity_curve'].iloc[-1] / initial_capital - 1) * 100
        buyhold_ret_pct = (bt_results['benchmark_curve'].iloc[-1] / initial_capital - 1) * 100
        full_benchmark_pct = buy_hold_return_pct(full_period_prices) if full_period_prices is not None else np.nan
        alpha_pct = strat_ret_pct - buyhold_ret_pct
        trades_df = bt_results['trades'].copy()
        # Show newest trades first by default
        if not trades_df.empty and 'Entry Date' in trades_df.columns:
            trades_df = trades_df.sort_values('Entry Date', ascending=False).reset_index(drop=True)

        st.write(f"#### 📊 {title}: Strategy vs Buy & Hold")
        if full_period_prices is not None and pd.notna(full_benchmark_pct):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Strategy Return", f"{strat_ret_pct:.2f}%")
            c2.metric(benchmark_label, f"{buyhold_ret_pct:.2f}%", help="Benchmark over the same window used for this strategy result.")
            c3.metric("Full Benchmark", f"{full_benchmark_pct:.2f}%", help="Buy & hold over the full selected chart period. Reference only if WFO starts after a training window.")
            c4.metric("Difference vs Test Benchmark", f"{alpha_pct:+.2f}%")
            c5.metric("Trades", len(trades_df))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy Return", f"{strat_ret_pct:.2f}%")
            c2.metric(benchmark_label, f"{buyhold_ret_pct:.2f}%")
            c3.metric("Difference", f"{alpha_pct:+.2f}%")
            c4.metric("Trades", len(trades_df))

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=bt_results['equity_curve'].index, y=bt_results['equity_curve'],
            mode='lines', line=dict(color='#00f2ff', width=2), name='Strategy'
        ))
        fig_perf.add_trace(go.Scatter(
            x=bt_results['benchmark_curve'].index, y=bt_results['benchmark_curve'],
            mode='lines', line=dict(color='gray', dash='dash'), opacity=0.75, name='Buy & Hold'
        ))
        fig_perf.update_layout(
            title=f"{title} Equity Curve", template="plotly_dark", height=420,
            hovermode="x unified", yaxis_title="Account Value"
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.write(f"#### 📝 {title} Trade Log")
        if not trades_df.empty:
            trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.date
            trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).apply(lambda x: x.date() if pd.notnull(x) else "Open")
            st.dataframe(trades_df.style.format({
                "Buy Price": "{:.2f}",
                "Sell Price": "{:.2f}",
                "PnL (%)": "{:.2f}%",
                "Cumulative Return (%)": "{:.2f}"
            }), use_container_width=True)
            st.download_button(
                f"📥 Download {title} Trade Log",
                trades_df.to_csv(index=False),
                file_name=f"{file_prefix}_TradeLog_{TICKER}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades generated by this signal in the selected date range.")

        return {
            'strategy_return_pct': strat_ret_pct,
            'buy_hold_return_pct': buyhold_ret_pct,
            'full_benchmark_pct': full_benchmark_pct,
            'alpha_pct': alpha_pct,
            'trades': trades_df
        }
    except Exception as e:
        st.error(f"{title} backtest error: {e}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fit_regime_model(model_data, n_regimes, switch_vol, switch_trend, search_reps=20):
    """
    Cached helper to fit Markov Regression.
    Returns the fitted result object.

    Robust fix:
    Statsmodels can occasionally fail with "Could not untransform parameters" during
    randomized Markov starting-parameter search. That is a fit-start problem, not a
    dashboard logic problem. We keep the original full scan/search_reps, but add safe
    fallback attempts so one bad window does not break the Regime tab.
    """
    # PREPARE DATA
    # Ensure input is a clean 1D float array, then wrap back into Series
    # to preserve Statsmodels pandas-compatibility (param names, indices)
    if hasattr(model_data, 'values'):
        clean_values = model_data.values.flatten().astype(float)
        idx = model_data.index
    else:
        clean_values = np.array(model_data).flatten().astype(float)
        idx = pd.RangeIndex(len(clean_values))

    # VALIDATION: Check for NaNs or Infinite values
    if np.any(np.isnan(clean_values)) or np.any(np.isinf(clean_values)):
        # Return quietly because WFO may test many windows/candidates.
        return None

    # VALIDATION: Check for constant data (no variance)
    if np.std(clean_values) < 1e-9:
        return None

    # Reconstruct robust 1D Series for Statsmodels
    endog_series = pd.Series(clean_values, index=idx)

    def _normalize_result(res_markov):
        """Force pandas-like params so downstream code can use names safely."""
        if isinstance(res_markov.params, np.ndarray):
            names = res_markov.model.param_names
            res_markov.params = pd.Series(res_markov.params, index=names)
            res_markov.bse = pd.Series(res_markov.bse, index=names)
            res_markov.pvalues = pd.Series(res_markov.pvalues, index=names)
        return res_markov

    # Keep original primary attempt first. Fallbacks only run if primary fails.
    # This preserves the full-scan behavior while preventing hard crashes.
    attempts = []
    attempts.append({
        'label': 'primary',
        'switching_variance': bool(switch_vol),
        'switching_trend': bool(switch_trend),
        'search_reps': int(search_reps),
        'em_iter': 10,
        'maxiter': 200,
        'method': 'lbfgs'
    })
    attempts.append({
        'label': 'deterministic_start',
        'switching_variance': bool(switch_vol),
        'switching_trend': bool(switch_trend),
        'search_reps': 0,
        'em_iter': 10,
        'maxiter': 300,
        'method': 'lbfgs'
    })
    attempts.append({
        'label': 'powell_start',
        'switching_variance': bool(switch_vol),
        'switching_trend': bool(switch_trend),
        'search_reps': 0,
        'em_iter': 5,
        'maxiter': 300,
        'method': 'powell'
    })
    # If the fully switching model is unstable, try simpler Markov specifications.
    # These are only safety fallbacks; they do not reduce the outer full scan.
    if switch_vol or switch_trend:
        attempts.append({
            'label': 'simple_switching_variance_only',
            'switching_variance': bool(switch_vol),
            'switching_trend': False,
            'search_reps': 0,
            'em_iter': 5,
            'maxiter': 250,
            'method': 'lbfgs'
        })
        attempts.append({
            'label': 'simple_constant_variance',
            'switching_variance': False,
            'switching_trend': False,
            'search_reps': 0,
            'em_iter': 5,
            'maxiter': 250,
            'method': 'lbfgs'
        })

    last_error = None
    for attempt in attempts:
        try:
            mod_markov = MarkovRegression(
                endog_series,
                k_regimes=int(n_regimes),
                trend='c',
                switching_variance=attempt['switching_variance'],
                switching_trend=attempt['switching_trend']
            )
            res_markov = mod_markov.fit(
                search_reps=attempt['search_reps'],
                em_iter=attempt['em_iter'],
                maxiter=attempt['maxiter'],
                method=attempt['method'],
                disp=False
            )
            return _normalize_result(res_markov)
        except Exception as e:
            last_error = str(e)
            continue

    # Do not spam Streamlit with red errors during WFO loops. A failed candidate/window
    # should simply be skipped so other candidates can continue.
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_master_signal(ticker, df, n_regimes=4, freq='Daily', opt_goal='Robustness (BIC)', stability=0, switch_vol=True, switch_trend=True, engine='Markov', initial_cap=10000.0, trailing_stop=0.0, stop_loss=0.0):
    """
    Unified Decision Logic for a single asset.
    Returns a dictionary of all quant metrics and the final Master Sentiment Score.
    n_regimes can be an integer (2, 3, 4) or 'Auto' for Best Fit.
    """
    try:
        # Central Data Cleaning
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # 0. Apply Model Stability (Smoothing) if requested
        if stability > 0:
            df['Returns'] = df['Returns'].ewm(span=stability, adjust=False).mean()
            df['Log_Returns'] = df['Log_Returns'].ewm(span=stability, adjust=False).mean()
            df['Close'] = df['Close'].ewm(span=stability, adjust=False).mean() 
        
        # 1. Data Resampling for timeframe sync
        if freq == 'Weekly':
            df = df.resample('W').last().replace([np.inf, -np.inf], np.nan).dropna()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Returns'] = df['Close'].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            
        if len(df) < 15: # Safety check for model convergence
            return None

        # 2. Regime Signal (Synchronized Engine)
        if engine == 'Markov':
            # Use the high-accuracy Markov model matching Backtest Tab
            if n_regimes == 'Auto':
                # Optimization loop for Markov (Reduced reps for speed)
                best_n = 4
                best_score = -float('inf') if opt_goal == 'Performance (PnL)' else float('inf')
                best_r = None
                
                for n in [2, 3, 4]:
                    try:
                        # Use very few search reps for the selection phase to save time
                        r = fit_regime_model(df['Returns']*100, n, switch_vol, switch_trend, search_reps=5)
                        if r:
                            if opt_goal == 'Performance (PnL)':
                                # Run full backtest proxy matching Backtest Tab comparison
                                p_df = r.filtered_marginal_probabilities
                                r_means = []
                                for i in range(n):
                                    m = r.params[f'const[{i}]'] if f'const[{i}]' in r.params else r.params.get('const', 0.0)
                                    r_means.append((i, m))
                                bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
                                
                                dom = p_df.idxmax(axis=1)
                                sigs = (dom == bull_idx).astype(int)
                                
                                bt_res = BacktestEngine.run_strategy(df['Close'], sigs, initial_cap, trailing_stop, stop_loss)
                                pnl = (bt_res['equity_curve'].iloc[-1] / initial_cap - 1)
                                
                                if pnl > best_score:
                                    best_score = pnl
                                    best_n = n
                                    best_r = r
                            else: # BIC
                                score = r.bic
                                if score < best_score:
                                    best_score = score
                                    best_n = n
                                    best_r = r
                    except: continue
                # We already have the best fit from the loop, no need to fit again
                res_markov = best_r
            else:
                res_markov = fit_regime_model(df['Returns']*100, int(n_regimes), switch_vol, switch_trend)
            
            if not res_markov: return None
            
            # Map Markov results to scanner standard
            p_df = res_markov.filtered_marginal_probabilities
            n_states = res_markov.k_regimes
            r_means = []
            for i in range(n_states):
                m = res_markov.params[f'const[{i}]'] if f'const[{i}]' in res_markov.params else res_markov.params.get('const', 0.0)
                r_means.append((i, m))
            
            bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
            bear_idx = sorted(r_means, key=lambda x: x[1])[0][0]
            curr_state = p_df.iloc[-1].idxmax()
            regime_prob = p_df.iloc[-1].max()
            
            if curr_state == bull_idx:
                regime_sig, regime_label = "LONG", "BULL"
            elif curr_state == bear_idx:
                regime_sig, regime_label = "SHORT", "BEAR"
            else:
                regime_sig, regime_label = "CASH", "NEUTRAL"
            
            regime_data = {'label': regime_label, 'confidence': regime_prob, 'n_states': n_states}
            p_detector = None # Placeholder since we didn't use GMM
            
        else: # GMM Engine (Fast)
            p_detector = ProRegimeDetector(df['Close'], df['Log_Returns'])
            if n_regimes == 'Auto':
                if opt_goal == 'Performance (PnL)':
                    best_n = 4
                    max_perf = -float('inf')
                    for n in [2, 3, 4]:
                        try:
                            temp_detector = ProRegimeDetector(df['Close'], df['Log_Returns'])
                            temp_detector.fit(n_states=n)
                            p_df = temp_detector.regimes['probs']
                            r_means = []
                            for sid, slbl in temp_detector.state_labels.items():
                                mask = (temp_detector.regimes['states'] == sid)
                                m_ret = df['Returns'].iloc[mask].mean() if any(mask) else 0
                                r_means.append(m_ret)
                            expected_returns = np.dot(p_df, r_means)
                            sigs = (expected_returns > 0).astype(int)
                            ret_sum = (df['Returns'].values[1:] * sigs[:-1]).sum()
                            if ret_sum > max_perf:
                                max_perf = ret_sum
                                best_n = n
                        except: continue
                    p_detector.fit(n_states=best_n)
                else:
                    p_detector.fit_optimized()
            else:
                p_detector.fit(n_states=int(n_regimes))
                
            regime_sig, regime_prob, regime_label = p_detector.get_latest_verdict()
            regime_data = {'label': regime_label, 'confidence': regime_prob, 'n_states': p_detector.metrics.get('n_states', 4)}
        
        # 2. Trend (Kalman)
        k_filter = KalmanFilterTrend(process_noise=1e-4, measurement_noise=1e-2)
        trend_est, _ = k_filter.filter(df['Close'].values)
        last_price = df['Close'].iloc[-1]
        last_trend = trend_est[-1]
        trend_diff = (last_price - last_trend) / (last_trend + 1e-9)
        
        # 3. Volatility (GARCH Proxy)
        returns_scaled = df['Returns'] * 100
        returns_scaled = returns_scaled.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns_scaled) < 15: return None # Safety for GARCH
        
        am = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off')
        curr_vol = res.conditional_volatility.iloc[-1]
        avg_vol = res.conditional_volatility.mean()
        vol_state = "HIGH" if curr_vol > avg_vol * 1.2 else "LOW" if curr_vol < avg_vol * 0.8 else "NORMAL"
        
        # 4. Jump Risk
        jump_res = RealizedVolatility.jump_component(df['Returns'].values)
        jump_detected = jump_res['p_value'] < 0.05

        # 5. Master Sentiment Score calculation
        sentiment_score = 0
        if "LONG" in regime_sig: sentiment_score += 2
        if "SHORT" in regime_sig: sentiment_score -= 2
        if trend_diff > 0.01: sentiment_score += 1
        if trend_diff < -0.01: sentiment_score -= 1
        if vol_state == "LOW": sentiment_score += 1
        if vol_state == "HIGH": sentiment_score -= 1
        if jump_detected: sentiment_score -= 1
        
        return {
            'regime_sig': regime_sig,
            'regime_label': regime_label,
            'regime_data': regime_data,
            'regime_prob': regime_prob,
            'pro_detector': p_detector,
            'trend_diff': trend_diff,
            'vol_state': vol_state,
            'curr_vol': curr_vol,
            'jump_detected': jump_detected,
            'sentiment_score': sentiment_score,
            'garch_res': res
        }
    except Exception as e:
        st.error(f"Error in Decision Engine for {ticker}: {e}")
        return None

@st.cache_data(ttl=60) # Cache live data for 1 minute
def load_data(ticker, start, end, interval='1d'):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return None
            
        # NORMALIZE TIMEZONE: Ensure all data is timezone-naive to avoid join errors
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1, drop_level=True) if ticker in df.columns.get_level_values(1) else df
            # If structure is different (Ticker as top level)
            if ticker in df.columns:
                 df = df[ticker]
            # Fallback for simple single ticker download structure
            elif 'Close' in df.columns and len(df.columns) > 1 and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

        # Standard cleaning
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

def load_iv_proxy_data_for_backtest(asset_index, live_mode=False, data_interval='1d', start_date=None, end_date=None):
    """
    Robust loader for ^VIX used by the Implied Vol Proxy strategy.
    In live mode, ^VIX can fail or have timestamps that do not exactly match the stock.
    This tries several intervals and returns the first usable proxy series.
    """
    if asset_index is None or len(asset_index) == 0:
        return None, None

    idx = pd.DatetimeIndex(asset_index)
    idx_min = idx.min()
    idx_max = idx.max()

    # Add a little buffer so rolling VIX features have enough prior data.
    start_buffer = idx_min - timedelta(days=10 if live_mode else 5)
    end_buffer = idx_max + timedelta(days=1)

    attempts = []
    if live_mode:
        # ^VIX intraday availability can be spotty. Try the requested interval first,
        # then progressively safer fallbacks. VX=F is included as a futures proxy.
        attempts.extend([
            ('^VIX', data_interval),
            ('^VIX', '15m'),
            ('^VIX', '60m'),
            ('VX=F', data_interval),
            ('VX=F', '15m'),
            ('VX=F', '60m'),
            ('^VIX', '1d')
        ])
    else:
        attempts.append(('^VIX', '1d'))

    seen = set()
    for proxy_ticker, interval in attempts:
        key = (proxy_ticker, interval)
        if key in seen:
            continue
        seen.add(key)
        try:
            df_proxy = load_data(proxy_ticker, start_buffer, end_buffer, interval=interval)
            if df_proxy is not None and not df_proxy.empty and 'Close' in df_proxy.columns:
                series = df_proxy['Close'].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
                if len(series) >= 5:
                    return series, f"{proxy_ticker} {interval}"
        except Exception:
            continue

    return None, None

def align_proxy_to_asset(asset_prices, proxy_prices):
    """
    Aligns VIX/proxy data to the asset index without requiring exact timestamp matches.
    This is critical for live mode because stock and ^VIX candles often have slightly
    different timestamps.
    """
    asset_prices = pd.Series(asset_prices).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    proxy_prices = pd.Series(proxy_prices).replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    if asset_prices.empty or proxy_prices.empty:
        return pd.DataFrame(columns=['asset', 'proxy'])

    if not isinstance(asset_prices.index, pd.DatetimeIndex) or not isinstance(proxy_prices.index, pd.DatetimeIndex):
        common_idx = asset_prices.index.intersection(proxy_prices.index)
        return pd.DataFrame({'asset': asset_prices.loc[common_idx], 'proxy': proxy_prices.loc[common_idx]}).dropna()

    # Timezone safety
    if asset_prices.index.tz is not None:
        asset_prices.index = asset_prices.index.tz_localize(None)
    if proxy_prices.index.tz is not None:
        proxy_prices.index = proxy_prices.index.tz_localize(None)

    # Forward-fill proxy values onto asset candle timestamps.
    union_idx = proxy_prices.index.union(asset_prices.index).sort_values()
    aligned_proxy = proxy_prices.reindex(union_idx).ffill().reindex(asset_prices.index)

    # If daily VIX is used on intraday stock data, early bars may need backfill.
    aligned_proxy = aligned_proxy.bfill()

    aligned = pd.DataFrame({'asset': asset_prices, 'proxy': aligned_proxy}, index=asset_prices.index).dropna()
    return aligned

@st.cache_data(ttl=3600) # Macro data changes weekly, cache for 1 hour
def load_fred_data(series_id):
    """
    Robust FRED data loader using direct CSV fetching.
    No API key required.
    """
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        # Handle '.' as NaN frequently found in FRED weekly data
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading FRED {series_id}: {e}")
        return None


# ==========================================
# LIVE SIGNAL ALERT HELPERS
# ==========================================
def _safe_secret(section, key, default=""):
    """Safely read Streamlit secrets without breaking local runs."""
    try:
        if section in st.secrets and key in st.secrets[section]:
            return st.secrets[section][key]
    except Exception:
        pass
    return default


def _alert_state_path():
    """Persistent local state so refresh/rerun does not resend the same alert."""
    try:
        return Path.cwd() / ".quant_live_alert_state.json"
    except Exception:
        return Path(".quant_live_alert_state.json")


def _load_alert_state():
    try:
        fp = _alert_state_path()
        if fp.exists():
            return json.loads(fp.read_text())
    except Exception:
        pass
    return {}


def _save_alert_state(state):
    try:
        _alert_state_path().write_text(json.dumps(state, indent=2, default=str))
    except Exception:
        # If the app host is read-only, session_state still prevents most duplicate alerts.
        pass


def _send_email_alert(subject, body, recipients, smtp_server, smtp_port, sender_email, sender_password, use_tls=True):
    """Send an email alert. SMS works by using carrier email-to-text addresses as recipients."""
    recipients = [r.strip() for r in str(recipients).replace(";", ",").split(",") if r.strip()]
    if not recipients:
        return False, "No alert recipients entered."
    if not smtp_server or not smtp_port or not sender_email or not sender_password:
        return False, "Missing SMTP settings. Add sender email/app password or Streamlit secrets."

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if int(smtp_port) == 465 and not use_tls:
            with smtplib.SMTP_SSL(smtp_server, int(smtp_port), timeout=20) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_server, int(smtp_port), timeout=20) as server:
                if use_tls:
                    server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
        return True, f"Alert sent to {', '.join(recipients)}"
    except Exception as e:
        return False, f"Alert send failed: {e}"


def maybe_send_live_signal_alert(
    enabled,
    live_mode,
    ticker,
    strategy_name,
    signals,
    prices,
    alert_config,
    extra_note=""
):
    """Send BUY/SELL alert only when the latest signal changes in live mode."""
    if not enabled or not live_mode or signals is None or prices is None:
        return

    try:
        sig = pd.Series(signals).replace([np.inf, -np.inf], np.nan).dropna()
        px = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
        if len(sig) < 2 or px.empty:
            return

        sig = sig.clip(lower=0, upper=1)
        prev_sig = float(sig.iloc[-2])
        last_sig = float(sig.iloc[-1])
        last_dt = sig.index[-1]
        aligned_px = px.reindex(sig.index).ffill().bfill()
        last_price = float(aligned_px.iloc[-1])

        action = None
        if prev_sig <= 0 and last_sig > 0:
            action = "BUY"
        elif prev_sig > 0 and last_sig <= 0:
            action = "SELL"
        else:
            return

        # Prevent duplicate alerts for the same ticker/strategy/bar/action.
        dedupe_key = f"{ticker}|{strategy_name}|{action}|{pd.Timestamp(last_dt).isoformat()}"
        state = _load_alert_state()
        session_key = f"last_alert_{ticker}_{strategy_name}"
        if state.get("last_alert_key") == dedupe_key or st.session_state.get(session_key) == dedupe_key:
            return

        subject = f"{action} Alert: {ticker} @ {last_price:.2f}"
        body = f"""Live model signal alert

Ticker: {ticker}
Action: {action}
Strategy: {strategy_name}
Signal time: {last_dt}
Approx price: {last_price:.2f}
Previous exposure: {prev_sig:.2f}
New exposure: {last_sig:.2f}

{extra_note}

This is a model alert, not financial advice. Confirm price/liquidity before trading.
"""
        ok, msg = _send_email_alert(
            subject=subject,
            body=body,
            recipients=alert_config.get("recipients", ""),
            smtp_server=alert_config.get("smtp_server", ""),
            smtp_port=alert_config.get("smtp_port", 587),
            sender_email=alert_config.get("sender_email", ""),
            sender_password=alert_config.get("sender_password", ""),
            use_tls=alert_config.get("use_tls", True),
        )

        if ok:
            st.session_state[session_key] = dedupe_key
            state["last_alert_key"] = dedupe_key
            state["last_alert_message"] = subject
            state["last_alert_time"] = datetime.now().isoformat()
            _save_alert_state(state)
            st.toast(f"🔔 {msg}")
        else:
            st.warning(msg)
    except Exception as e:
        st.warning(f"Live alert check failed: {e}")

# FED Balance Sheet Series Definitions
FED_ASSETS = {
    "WGCAL": "Gold Certificate Account",
    "SDRACL": "Special Drawing Rights Certificate Account",
    "WCOINL": "Coin",
    "WSHONBLL": "Treasury Bills",
    "WSHONBNL": "Treasury Notes and Bonds",
    "WSHONBIIL": "Treasury Tips",
    "WSHOMCB": "Mortgage-Backed Securities",
    "WUDSHO": "Unamortized Premiums on Securities",
    "WLCFOCEL": "Other Credit Extensions",
    "WOTHAL": "Other Assets"
}

FED_LIABILITIES = {
    "WCURCIR": "Currency in Circulation",
    "WDTPGCAS": "Treasury General Account (TGA)",
    "RRPONTSYD": "Overnight Reverse Repo (RRP)",
    "WLRRAL": "Reverse Repurchase Agreements (Total)",
    "WLFN": "Federal Reserve Notes",
    "WDFOL": "Foreign Official Deposits",
    "WDLTCL": "Term Deposits held by Depository Institutions"
}


def robust_fetch_csv(url, sep="|", timeout=10):
    """Reliably fetches CSV data with custom headers and timeout."""
    import requests
    import io
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text), sep=sep, skipfooter=1, engine='python')
    except Exception as e:
        raise e

@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)
        df = table[0]
        return df['Symbol'].tolist()
    except:
        return ["AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ"]

@st.cache_data(ttl=3600*24)
def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        table = pd.read_html(url)
        df = table[4]
        return df['Ticker'].tolist()
    except:
        return [
            "AAPL", "MSFT", "AMZN", "GOOG", "NVDA", "META", "TSLA", "PEP", "AVGO", "COST",
            "AZN", "CSCO", "TMUS", "ADBE", "TXN", "CMCSA", "QCOM", "AMGN", "HON", "INTU",
            "INTC", "SBUX", "AMD", "AMD", "GILD", "VRTX", "MDLZ", "REGN", "ISRG", "ADI",
            "BKNG", "AMAT", "ADP", "PDD", "PYPL", "MU", "VRSK", "MELI", "KDP", "LUK"
        ]

@st.cache_data(ttl=3600*24)
def get_total_us_stocks():
    """Retrieves all listed US stocks with multi-source failover."""
    import requests
    
    # Primary Source: SEC Company Tickers (Extremely reliable, 10,000+ US tickers)
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': 'QuantApp/1.0 (admin@quantapp.local)'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            tickers = [v['ticker'] for k, v in data.items()]
            # Filter valid common stock tickers
            tickers = sorted(list(set([str(t).strip() for t in tickers if str(t).strip() and len(str(t)) <= 5 and '-' not in str(t)])))
            if len(tickers) > 5000:
                return tickers
    except:
        pass

    # Secondary Source: NASDAQ FTP
    sources = [
        "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/nasdaq/nasdaq_full_tickers.txt"
    ]
    
    for url in sources:
        try:
            nasdaq = robust_fetch_csv(url, sep="|" if "ftp.nasdaqtrader" in url else ",")
            if 'nasdaqlisted' in url:
                nasdaq = nasdaq[nasdaq['Test Issue'] == 'N']
                tickers = nasdaq['Symbol'].tolist()
            else:
                tickers = nasdaq['symbol'].tolist() if 'symbol' in nasdaq.columns else nasdaq.iloc[:, 0].tolist()
            
            res = sorted(list(set([str(t).strip() for t in tickers if str(t).strip() and len(str(t)) < 6])))
            if len(res) > 500: return res
        except:
            continue

    st.warning("⚠️ Total Market Connection Issue. Using expanded internal mid-cap universe (Top 500).")
    # Massive internal fallback
    return get_sp500_tickers() + ["AAPL", "TSLA", "NVDA", "AMD", "PLTR", "SQ", "PYPL", "COIN", "MARA", "RIOT"]

@st.cache_data(ttl=3600*24)
def get_total_us_etfs():
    """Retrieves US ETFs with robust failover."""
    try:
        url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        data = robust_fetch_csv(url, sep="|")
        etfs = data[(data['Test Issue'] == 'N') & (data['ETF'] == 'Y')]
        res = sorted(list(set([str(t).strip() for t in etfs['NASDAQ Symbol'].tolist()])))
        if len(res) > 100: return res
    except:
        pass
    
    st.warning("⚠️ ETF Universe Connection Issue. Using expanded fallback list.")
    return get_etf_universe()


def get_market_cap(ticker):
    """Fetches approximate market cap in USD."""
    try:
        t = yf.Ticker(ticker)
        mcap = t.info.get('marketCap', 0)
        return mcap
    except:
        return 0

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

class ReportGenerator:
    """
    Handles PDF and Excel generation for the quant report.
    """
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_store = {} # Stores dataframes and dicts for Excel
        self.plots = {} # Stores IO buffers for plots

    def add_data(self, key, df_or_dict):
        self.data_store[key] = df_or_dict

    def add_plot(self, key, fig):
        buf = io.BytesIO()
        if hasattr(fig, 'savefig'): # Matplotlib
            fig.savefig(buf, format='png', bbox_inches='tight')
        elif hasattr(fig, 'write_image'): # Plotly
            try:
                fig.write_image(buf, format='png', engine='kaleido')
            except Exception as e:
                # Fallback if kaleido fails
                print(f"Plotly export failed: {e}")
                return
        else:
            return
        buf.seek(0)
        self.plots[key] = buf

    def generate_excel(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for key, data in self.data_store.items():
                if isinstance(data, pd.DataFrame):
                    # Clean sheet name (max 31 chars, no invalid chars)
                    sheet_name = "".join([c for c in key if c.isalnum() or c in (" ", "_")])[:31]
                    data.to_excel(writer, sheet_name=sheet_name)
                elif isinstance(data, dict):
                    df_dict = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
                    sheet_name = "".join([c for c in key if c.isalnum() or c in (" ", "_")])[:31]
                    df_dict.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue()

    def generate_pdf(self):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(44, 62, 80) # Dark Blue
        pdf.cell(0, 20, f"Unified Quant Analysis Report", ln=True, align='C')
        
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Asset: {self.ticker}", ln=True, align='C')
        
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Analysis Period: {self.start_date} to {self.end_date}", ln=True, align='C')
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
        
        # Track which plots have been printed
        printed_plots = set()

        # 1. First print all Data Store items (and their matching plots)
        for key, data in self.data_store.items():
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(31, 119, 180) # Theme blue
            pdf.cell(0, 10, f"SECTION: {key}", ln=True)
            pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
            pdf.ln(2)
            
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(0, 0, 0)
            
            if isinstance(data, dict):
                for k, v in data.items():
                    val_str = f"{v:.4f}" if isinstance(v, (float, np.float64, np.float32)) else str(v)
                    pdf.cell(70, 6, f"{k}:", border=0)
                    pdf.cell(0, 6, val_str, ln=True, border=0)
            elif isinstance(data, pd.DataFrame):
                pdf.set_font("Arial", 'I', 10)
                pdf.cell(0, 7, f"Data Table: {len(data)} rows. (See Excel for full dataset)", ln=True)
                pdf.set_font("Arial", size=10)
            
            # Add corresponding plot if exists
            if key in self.plots:
                pdf.ln(2)
                pdf.image(self.plots[key], x=15, w=180)
                printed_plots.add(key)
                pdf.ln(5)
            
            pdf.ln(10)
            
            # Add page break if near bottom
            if pdf.get_y() > 230:
                pdf.add_page()

        # 2. Print any remaining plots that weren't matched to data_store keys
        remaining_plots = [k for k in self.plots.keys() if k not in printed_plots]
        if remaining_plots:
            if pdf.get_y() > 100: # New page if not much room
                pdf.add_page()
                
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Additional Visualizations", ln=True)
            pdf.ln(5)
            
            for key in remaining_plots:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, key, ln=True)
                pdf.image(self.plots[key], x=15, w=180)
                pdf.ln(10)
                if pdf.get_y() > 230:
                    pdf.add_page()
            
        # fpdf2 returns bytearray by default when no dest is provided
        pdf_raw = pdf.output()
        if isinstance(pdf_raw, bytearray):
            return bytes(pdf_raw)
        return pdf_raw

# ==========================================

def walk_forward_strategy_selection_institutional(prices, candidates, train_window=252, forward_window=21, initial_capital=10000.0, confirmed_bar=True, trailing_stop_pct=0.0, stop_loss_pct=0.0, top_n=3, ensemble_threshold=0.45, switch_penalty=8.0, min_trades=3, trend_override=True):
    """
    Institutional-style walk-forward selector.
    Differences from single-winner WFO:
    - Scores candidates by risk-adjusted quality, not raw return only.
    - Uses a top-N ensemble vote instead of trusting one rule.
    - Penalizes frequent rule switching.
    - Adds a price-trend override so strong runners are not missed by a defensive IV filter.
    """
    prices = pd.Series(prices).replace([np.inf, -np.inf], np.nan).dropna()
    if len(prices) < int(train_window) + max(5, int(forward_window)):
        return None

    train_window = int(train_window)
    forward_window = int(forward_window)
    top_n = max(1, int(top_n))
    idx = prices.index

    sma50 = prices.rolling(50, min_periods=20).mean()
    sma200 = prices.rolling(200, min_periods=50).mean()
    mom63 = prices.pct_change(63)
    strong_trend = ((prices > sma50) & (prices > sma200) & (mom63 > 0)).fillna(False)
    weak_trend = ((prices < sma50) & (mom63 < 0)).fillna(False)

    wf_signal = pd.Series(np.nan, index=idx, dtype=float)
    wf_rows = []
    strategy_sequence = []
    prev_primary = None

    start = train_window
    period_no = 1
    while start < len(prices):
        train_idx = idx[start - train_window:start]
        test_idx = idx[start:min(start + forward_window, len(prices))]
        if len(test_idx) < 2:
            break

        train_prices = prices.loc[train_idx]
        test_prices = prices.loc[test_idx]
        train_scores = []

        for name, logic, sig in candidates:
            sig_series = pd.Series(sig).reindex(idx).ffill().fillna(0).clip(0, 1)
            train_sig = sig_series.reindex(train_idx).ffill().fillna(0).clip(0, 1)
            score_res = evaluate_strategy_candidate(
                train_prices, train_sig,
                initial_capital=initial_capital,
                trailing_stop_pct=trailing_stop_pct,
                stop_loss_pct=stop_loss_pct
            )
            if score_res is None:
                continue

            raw_returns = score_res.get('raw', {}).get('returns', pd.Series(dtype=float))
            risk_metrics = BacktestEngine.calculate_metrics(raw_returns) if len(raw_returns) else {}
            sharpe = float(risk_metrics.get('Sharpe Ratio', 0.0))
            max_dd = float(score_res.get('Max DD %', 0.0))
            trades = int(score_res.get('Trades', 0))
            strat_ret = float(score_res.get('Strategy Return %', 0.0))
            diff = float(score_res.get('Difference %', 0.0))

            # Institutional score: return matters, but drawdown, low sample size, and instability matter too.
            low_trade_penalty = 8.0 if 0 < trades < int(min_trades) else 0.0
            no_trade_penalty = 15.0 if trades == 0 else 0.0
            dd_penalty = 0.35 * abs(min(max_dd, 0.0))
            sharpe_bonus = 8.0 * np.tanh(sharpe / 2.0)
            score = (0.55 * strat_ret) + (0.45 * diff) + sharpe_bonus - dd_penalty - low_trade_penalty - no_trade_penalty

            # Do not let tiny differences cause strategy flipping.
            if prev_primary is not None and name != prev_primary:
                score -= float(switch_penalty)

            train_scores.append({
                'name': name,
                'logic': logic,
                'signal': sig_series,
                'score': score,
                'train_return': strat_ret,
                'train_diff': diff,
                'train_dd': max_dd,
                'train_sharpe': sharpe,
                'train_trades': trades
            })

        # 4) Benchmark-aware return booster candidate.
        # Goal: get closer to buy-and-hold during strong trends while still using a trend-break exit
        # to protect drawdown. It is scored on training only and tested forward unseen.
        if use_return_booster:
            try:
                booster_train = benchmark_aware_trend_participation_signal(
                    prices.loc[train_idx], mode=str(return_booster_mode)
                )
                if confirmed_bar:
                    booster_train = booster_train.shift(1).ffill().fillna(0).clip(0, 1)
                booster_score = evaluate_strategy_candidate(
                    prices.loc[train_idx], booster_train,
                    initial_capital=initial_capital,
                    trailing_stop_pct=trailing_stop_pct,
                    stop_loss_pct=stop_loss_pct
                )
                if booster_score is not None:
                    # Score wants benchmark participation, but still punishes drawdown.
                    booster_score["Institutional Score"] = risk_adjusted_candidate_score(
                        booster_score, activity_mode="ReturnBooster"
                    )
                    train_scores.append({
                        "method": f"Benchmark-Aware Return Booster ({str(return_booster_mode)})",
                        "n_regimes": "Booster",
                        "score": booster_score,
                        "signal": booster_train,
                        "activity_variant": "ReturnBooster"
                    })
            except Exception:
                pass

        if not train_scores:
            start += forward_window
            period_no += 1
            continue

        train_scores = sorted(train_scores, key=lambda x: (x['score'], x['train_diff'], x['train_return']), reverse=True)
        selected = train_scores[:min(top_n, len(train_scores))]
        primary = selected[0]
        prev_primary = primary['name']

        # Weighted top-N ensemble. Rank weights avoid overtrusting one lucky winner.
        base_weights = np.array([0.50, 0.30, 0.20] + [0.0] * 10, dtype=float)[:len(selected)]
        if len(selected) == 1:
            base_weights = np.array([1.0])
        else:
            base_weights = base_weights / base_weights.sum()

        ensemble_full = pd.Series(0.0, index=idx, dtype=float)
        for w, item in zip(base_weights, selected):
            ensemble_full = ensemble_full.add(item['signal'].reindex(idx).ffill().fillna(0).clip(0, 1) * float(w), fill_value=0.0)

        if confirmed_bar:
            ensemble_exec = ensemble_full.shift(1).ffill().fillna(0).clip(0, 1)
        else:
            ensemble_exec = ensemble_full.ffill().fillna(0).clip(0, 1)

        # Convert ensemble vote into executable exposure.
        # Strong trend override keeps exposure on for big momentum names unless the ensemble is almost fully defensive.
        test_ensemble = ensemble_exec.reindex(test_idx).ffill().fillna(0).clip(0, 1)
        test_signal = (test_ensemble >= float(ensemble_threshold)).astype(float)
        if trend_override:
            st_test = strong_trend.reindex(test_idx).fillna(False)
            wk_test = weak_trend.reindex(test_idx).fillna(False)
            test_signal = test_signal.where(~(st_test & (test_ensemble >= 0.20)), 1.0)
            test_signal = test_signal.where(~(wk_test & (test_ensemble <= 0.20)), 0.0)

        wf_signal.loc[test_idx] = test_signal

        test_score = evaluate_strategy_candidate(
            test_prices, test_signal,
            initial_capital=initial_capital,
            trailing_stop_pct=trailing_stop_pct,
            stop_loss_pct=stop_loss_pct
        )
        strat_return = np.nan
        bh_return = np.nan
        diff_return = np.nan
        trades = 0
        max_dd = np.nan
        if test_score is not None:
            strat_return = test_score.get('Strategy Return %', np.nan)
            bh_return = test_score.get('Buy & Hold Return %', np.nan)
            diff_return = test_score.get('Difference %', np.nan)
            trades = test_score.get('Trades', 0)
            max_dd = test_score.get('Max DD %', np.nan)

        selected_names = ' + '.join([x['name'] for x in selected])
        wf_rows.append({
            'Period': period_no,
            'Train Start': train_idx[0],
            'Train End': train_idx[-1],
            'Forward Start': test_idx[0],
            'Forward End': test_idx[-1],
            'Selected Rule': primary['name'],
            'Ensemble Rules': selected_names,
            'Train Score': round(primary['score'], 2),
            'Train Diff %': round(primary['train_diff'], 2) if pd.notna(primary['train_diff']) else np.nan,
            'Train Sharpe': round(primary['train_sharpe'], 2) if pd.notna(primary['train_sharpe']) else np.nan,
            'Forward Strategy %': round(strat_return, 2) if pd.notna(strat_return) else np.nan,
            'Forward Buy & Hold %': round(bh_return, 2) if pd.notna(bh_return) else np.nan,
            'Forward Diff %': round(diff_return, 2) if pd.notna(diff_return) else np.nan,
            'Forward Max DD %': round(max_dd, 2) if pd.notna(max_dd) else np.nan,
            'Forward Trades': trades
        })
        strategy_sequence.append(primary['name'])

        start += forward_window
        period_no += 1

    wf_signal = wf_signal.ffill().fillna(0).clip(0, 1)
    if len(wf_rows) == 0:
        return None

    first_forward_start = wf_rows[0]['Forward Start']
    wf_eval_index = prices.loc[first_forward_start:].index
    wf_prices = prices.loc[wf_eval_index]
    wf_sig_active = wf_signal.reindex(wf_eval_index).ffill().fillna(0).clip(0, 1)
    overall = evaluate_strategy_candidate(
        wf_prices, wf_sig_active,
        initial_capital=initial_capital,
        trailing_stop_pct=trailing_stop_pct,
        stop_loss_pct=stop_loss_pct
    )

    rows_df = pd.DataFrame(wf_rows)
    wins = int((rows_df['Forward Diff %'] > 0).sum()) if 'Forward Diff %' in rows_df else 0
    valid_periods = int(rows_df['Forward Diff %'].notna().sum()) if 'Forward Diff %' in rows_df else len(rows_df)
    changes = sum(1 for a, b in zip(strategy_sequence, strategy_sequence[1:]) if a != b)
    change_rate = changes / max(1, len(strategy_sequence) - 1)
    win_rate = wins / max(1, valid_periods)
    avg_forward_diff = float(rows_df['Forward Diff %'].dropna().mean()) if 'Forward Diff %' in rows_df and rows_df['Forward Diff %'].notna().any() else 0.0
    stability_score = round(100 * (0.60 * win_rate + 0.25 * max(0, min(1, avg_forward_diff / 10)) + 0.15 * (1 - change_rate)), 0)

    return {
        'signal': wf_signal,
        'rows': rows_df,
        'overall': overall,
        'win_rate': win_rate,
        'changes': changes,
        'change_rate': change_rate,
        'avg_forward_diff': avg_forward_diff,
        'stability_score': stability_score,
        'strategy_sequence': strategy_sequence,
        'confirmed_bar': confirmed_bar,
        'mode': 'Institutional Ensemble'
    }

# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Thesis Parameters")
    
    # Market Region Selector
    market_region = st.selectbox("Market Region", [
        "US Market (USD)", 
        "Indian Market (INR)", 
        "Futures / Commodities (USD)"
    ])
    
    if market_region == "Indian Market (INR)":
        CURRENCY = "₹"
        BENCHMARK = "^NSEI"
        DEFAULT_RF = 7.0
        SUFFIX = ".NS"
    elif market_region == "Futures / Commodities (USD)":
        CURRENCY = "$"
        BENCHMARK = "GC=F" # Gold Futures (Comex) as default safe haven benchmark
        DEFAULT_RF = 4.0
        SUFFIX = "" # User requested no default suffix (allows specific contracts like SIH26.CMX)
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
    
    # DEBUG: Temporary visualization to prove logic
    with st.expander("🛠️ Debug Info (Remove Later)", expanded=True):
        st.write(f"Region: {market_region}")
        st.code(f"SUFFIX = '{SUFFIX}'")
        st.code(f"Raw Ticker = '{raw_ticker}'")
        st.code(f"Final TICKER = '{TICKER}'")
    
    start_date = st.date_input("Start Date", DEFAULT_NONLIVE_START)
    end_date = st.date_input("End Date", datetime.now())
    
    st.subheader("Model Settings")
    rf_rate = st.number_input("Risk Free Rate (%)", 0.0, 20.0, DEFAULT_RF) / 100
    st.info(f"Benchmark: {BENCHMARK} | Currency: {CURRENCY}")

    st.divider()
    st.header("🔬 Model Configuration")
    regime_mode = st.selectbox("Regime Detection Mode", 
                               ["Fixed: 4 States (Inst.)", 
                                "Fixed: 2 States (Bull/Bear)", 
                                "Fixed: 3 States (Bull/Neut/Bear)",
                                "Auto: Best Fit (AIC/BIC)"],
                               index=0,
                               help="Standard: 4-states. Auto: Tries 2, 3, and 4 states for each asset and picks the best fit.")
    
    # Map to parameter
    regime_val_map = {
        "Fixed: 4 States (Inst.)": 4,
        "Fixed: 2 States (Bull/Bear)": 2,
        "Fixed: 3 States (Bull/Neut/Bear)": 3,
        "Auto: Best Fit (AIC/BIC)": "Auto"
    }
    regime_param = regime_val_map[regime_mode]

    with st.expander("🛠️ Advanced Model Sync", expanded=False):
        reg_engine = st.selectbox("Model Engine", ["Markov (High Accuracy)", "GMM (Fast)"], index=0)
        reg_engine_param = "Markov" if "Markov" in reg_engine else "GMM"
        reg_stability = st.slider("Signal Stability (Smoothing)", 0, 10, 4)
        reg_opt_goal = st.selectbox("Optimization Goal", ["Robustness (BIC)", "Performance (PnL)"], index=0)
        reg_switch_vol = st.toggle("Switching Volatility", value=True)
        reg_switch_trend = st.toggle("Switching Mean", value=True)
        initial_cap = st.number_input("Initial Capital", 1000, 1000000, 10000)
        
        use_trailing_stop = st.toggle("Enable Trailing Stop Loss", value=False)
        trailing_stop = st.slider("Trailing Stop Loss (%)", 0.0, 20.0, 5.0, step=0.5) / 100 if use_trailing_stop else 0.0
        
        use_stop_loss = st.toggle("Enable Hard Stop Loss", value=True)
        stop_loss = st.slider("Hard Stop Loss (%)", 0.0, 30.0, 8.0, step=0.5) / 100 if use_stop_loss else 0.0

    st.divider()
    st.header("⚡ Live Decision Mode")
    live_mode = st.toggle("Enable Live Data", value=False, help="Fetches recent 1m/5m data for real-time decision support.")
    if live_mode:
        data_interval = st.selectbox("Live Interval", ["1m", "5m", "15m", "60m"], index=1)
        st.info("Live mode uses a shorter window and higher frequency data for tactical edge.")
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    else:
        data_interval = '1d'

    st.divider()
    st.header("🔔 Live Buy/Sell Alerts")
    alert_enabled = st.toggle(
        "Enable live signal alerts",
        value=False,
        help="Sends an email/SMS-gateway alert only when the selected live strategy flips BUY or SELL."
    )

    default_recipients = _safe_secret("alert_email", "recipients", _safe_secret("alert_email", "recipient", ""))
    default_smtp_server = _safe_secret("alert_email", "smtp_server", "smtp.gmail.com")
    default_smtp_port = int(_safe_secret("alert_email", "smtp_port", 587))
    default_sender_email = _safe_secret("alert_email", "sender_email", "")
    default_sender_password = _safe_secret("alert_email", "sender_password", "")
    default_use_tls = bool(_safe_secret("alert_email", "use_tls", True))

    if alert_enabled:
        st.caption("For text messages, enter your carrier SMS gateway email, or use your normal email. Example: yournumber@vtext.com.")
        alert_recipients = st.text_input("Alert recipient(s)", value=default_recipients, help="Comma-separated emails or SMS-gateway addresses.")
        with st.expander("SMTP sender settings", expanded=False):
            alert_smtp_server = st.text_input("SMTP server", value=default_smtp_server)
            alert_smtp_port = st.number_input("SMTP port", min_value=1, max_value=9999, value=default_smtp_port, step=1)
            alert_use_tls = st.checkbox("Use TLS", value=default_use_tls)
            alert_sender_email = st.text_input("Sender email", value=default_sender_email)
            alert_sender_password = st.text_input("Sender app password", value=default_sender_password, type="password")
        if not live_mode:
            st.info("Alerts only fire when Live Data mode is enabled.")
    else:
        alert_recipients = default_recipients
        alert_smtp_server = default_smtp_server
        alert_smtp_port = default_smtp_port
        alert_use_tls = default_use_tls
        alert_sender_email = default_sender_email
        alert_sender_password = default_sender_password

    alert_config = {
        "recipients": alert_recipients,
        "smtp_server": alert_smtp_server,
        "smtp_port": int(alert_smtp_port),
        "use_tls": bool(alert_use_tls),
        "sender_email": alert_sender_email,
        "sender_password": alert_sender_password,
    }

    st.subheader("Report Export")
    if not EXPORT_AVAILABLE:
        st.error("📥 Export libraries missing.")
        st.info("To enable PDF/Excel exports, add `fpdf2` and `xlsxwriter` to your `requirements.txt` or run: `pip install fpdf2 xlsxwriter`")
    else:
        if 'report_gen' not in st.session_state:
            st.session_state.report_gen = None

        if st.session_state.report_gen:
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                try:
                    raw_pdf = st.session_state.report_gen.generate_pdf()
                    
                    # Hyper-robust conversion to bytes
                    if isinstance(raw_pdf, bytes):
                        pdf_bytes = raw_pdf
                    elif isinstance(raw_pdf, bytearray):
                        pdf_bytes = bytes(raw_pdf)
                    elif isinstance(raw_pdf, str):
                        pdf_bytes = raw_pdf.encode('latin1')
                    else:
                        pdf_bytes = bytes(raw_pdf)
                        
                    st.download_button(
                        label="📥 PDF Report",
                        data=pdf_bytes,
                        file_name=f"Quant_Report_{TICKER}.pdf",
                        mime="application/pdf",
                        key="pdf_download_btn"
                    )
                except Exception as e:
                    st.error(f"PDF Error: {str(e)}")
                    st.info(f"Debug Info: Type of data is {type(raw_pdf) if 'raw_pdf' in locals() else 'Unknown'}")
                    st.caption(f"Active File: {__file__}")
            with col_ex2:
                try:
                    excel_bytes = st.session_state.report_gen.generate_excel()
                    st.download_button(
                        label="📥 Excel Data",
                        data=excel_bytes,
                        file_name=f"Quant_Data_{TICKER}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.error(f"Excel Error: {e}")
        else:
            st.caption("Perform an analysis to enable exports.")

# ==========================================
# 4. DATA LOADING
# ==========================================
# Standardize current time to the nearest minute for robust caching
now_rounded = datetime.now().replace(second=0, microsecond=0)

if live_mode:
    # Intraday limits: 1m (7d), 5m-15m (60d), 60m (730d)
    # We use 30d as a robust default for decision support models to have enough history
    lookback_days = 7 if data_interval == '1m' else 30
    df_main = load_data(TICKER, now_rounded - timedelta(days=lookback_days), now_rounded, interval=data_interval)
else:
    df_main = load_data(TICKER, start_date, end_date, interval='1d')

st.subheader("Asset & Macro Analysis Suite")

# 5. UNIFIED TAB ARCHITECTURE
# ==========================================
tabs = st.tabs([
    "💡 Decision Summary",
    "Volatility (GARCH)", 
    "Regime Switching", 
    "Stochastic (Heston/Jump)", 
    "Kalman Filter", 
    "Macro Factors",
    "Structural",
    "Backtest",
    "Volatility Clustering",
    "Advanced Regime",
    "SML & Alpha",
    "📡 Multi-Asset Scan",
    "🏦 FED Balance Sheet",
    "🎲 Options IV Surface",
    "🎲 Hurst Exponent",
    "🔥 Hot 10 (Daily)",
    "🎯 Institutional IV Scanner",
    "📊 CVD & Volume Delta",
    "📈 Institutional VWAP",
    "🔬 Time Series Analysis"
])

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = tabs

if df_main is not None:
    # Initialize Report Generator
    st.session_state.report_gen = ReportGenerator(TICKER, start_date, end_date)
    st.session_state.report_gen.add_data("Historical Data", df_main.tail(100))
    
    # 6. UNIFIED DECISION ENGINE (Global Signals)
    # ==========================================
    with st.sidebar:
        st.divider()
        st.caption("🔍 Processing Model Signals...")
        prog_bar = st.progress(0)
    
    analysis = get_master_signal(TICKER, df_main, 
                                  n_regimes=regime_param, 
                                  freq='Daily', 
                                  opt_goal=reg_opt_goal,
                                  stability=reg_stability,
                                  switch_vol=reg_switch_vol,
                                  switch_trend=reg_switch_trend,
                                  engine=reg_engine_param,
                                  initial_cap=initial_cap,
                                  trailing_stop=trailing_stop,
                                  stop_loss=stop_loss)
    if analysis:
        regime_sig = analysis['regime_sig']
        regime_label = analysis['regime_label']
        regime_data = analysis['regime_data']
        regime_prob = analysis['regime_prob']
        pro_detector = analysis['pro_detector']
        trend_diff = analysis['trend_diff']
        vol_state = analysis['vol_state']
        curr_vol = analysis['curr_vol']
        jump_detected = analysis['jump_detected']
        sentiment_score = analysis['sentiment_score']
        # For Tab 1 diagnostics
        res_sum = analysis['garch_res']
        prog_bar.progress(100)
        prog_bar.empty()
    else:
        st.sidebar.error("Decision Engine Error: Statistical convergence failed.")
        regime_sig, regime_data = "N/A", {'label': 'Error', 'confidence': 0.0}
        regime_label = "N/A"
        trend_diff = 0.0
        vol_state = "UNKNOWN"
        curr_vol = 0.0
        jump_detected = False
        sentiment_score = 0
        res_sum = None
        prog_bar.empty()
else:
    # No ticker loaded state
    regime_sig, regime_label, sentiment_score = "N/A", "N/A", 0
    regime_data = {'label': 'N/A', 'confidence': 0.0}
    trend_diff = 0.0
    vol_state = "UNKNOWN"
    curr_vol = 0.0
    jump_detected = False
    res_sum = None

# ==========================================
# TAB 0: DECISION SUMMARY
# ==========================================
with tab0:
    if df_main is None:
        st.info("💡 **Welcome to the Quant Suite**. Currently, no ticker is loaded. Enter a ticker in the sidebar and enable 'Load Data' to begin technical and statistical analysis.")
        st.write("Meanwhile, you can explore the **🏦 FED Balance Sheet** tab for macroeconomic context.")
    else:
        st.write("### 🧠 Executive Decision Dashboard")
    st.markdown(f"**Unified Quant Signal for {TICKER}** | Interval: `{data_interval}` | Mode: {'Live' if live_mode else 'Historical'}")
    
    # 2. DASHBOARD DISPLAY
    # --------------------
    col_dec1, col_dec2, col_dec3 = st.columns(3)
    
    with col_dec1:
        st.metric("Institutional Regime", regime_label, f"{regime_data['confidence']:.1%} Conf")
        if "BULL" in regime_label: st.success(f"**Action**: {regime_sig}")
        elif "BEAR" in regime_label: st.error(f"**Action**: {regime_sig}")
        else: st.info("Market in transition.")

    with col_dec2:
        st.metric("Trend (Kalman)", f"{trend_diff:+.2%}", "Vs Trend Line")
        if trend_diff > 0.02: st.success("Price trending above support.")
        elif trend_diff < -0.02: st.warning("Trend breakdown in progress.")
        else: st.info("Consolidating at trend line.")

    with col_dec3:
        st.metric("Volatility Regime", vol_state, f"{curr_vol:.2f} (Daily %)")
        if vol_state == "HIGH": st.warning("High Vol: Reduce Position Size.")
        elif vol_state == "LOW": st.success("Low Vol: Favorable for Leverage.")
        else: st.info("Standard risk environment.")

    st.divider()
    
    # 3. MASTER SENTIMENT GAUGE
    # Score already calculated in global decision engine via get_master_signal
    
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.write("#### Master Quant Score")
        # Score range approx -5 to +4
        if sentiment_score >= 2: 
            st.header(f"🟢 BULLISH ({sentiment_score})")
            st.button("🚀 EXECUTE BUY", use_container_width=True, type="primary")
        elif sentiment_score <= -2:
            st.header(f"🔴 BEARISH ({sentiment_score})")
            st.button("⚠️ EXECUTE SELL/HEDGE", use_container_width=True, type="primary")
        else:
            st.header(f"🟡 NEUTRAL ({sentiment_score})")
            st.button("⚖️ MAINTAIN NEUTRAL", use_container_width=True)
            
    with m_col2:
        st.write("#### Risk Alerts")
        if jump_detected:
            st.error("🚨 **FAT TAIL RISK**: Significant price jumps detected. Stochastic models (Heston/Jump) recommended for testing tail risk.")
        else:
            st.success("✅ **SMOOTH DYNAMICS**: No significant jumps. Gaussian models are stable.")
        
        if vol_state == "HIGH":
            st.warning("⚠️ **VOL CLUSTERING**: Recent shocks are likely to trigger further volatility. See Hawkes tab.")
        
        # Recommendation
        st.info(f"**Recommendation**: {regime_sig}. Target Exposure: {min(1.0, 0.5 + 0.1*sentiment_score):.0%} of risk parity weight.")

    st.divider()
    st.caption("This summary aggregates deep statistical models. For detailed justification, visit the respective tabs.")

# ==========================================
# TAB 1: VOLATILITY (GARCH/Risk)
# ==========================================
with tab1:
    if df_main is None:
        st.warning("Please load a ticker to view Volatility models.")
    else:
        st.write("### 📉 Advanced Volatility Analysis")
    # --- MODEL VERDICT BANNER ---
    if res_sum is not None:
        latest_vol = res_sum.conditional_volatility.iloc[-1]
        vol_msg = f"Volatility is currently **{vol_state}** ({latest_vol:.2f}% daily)."
        if vol_state == "HIGH": st.error(f"🎯 **MODEL VERDICT**: {vol_msg} Defensive sizing recommended.")
        else: st.success(f"🎯 **MODEL VERDICT**: {vol_msg} Risk environment is stable.")
    
    if ARCH_AVAILABLE and df_main is not None:
        returns_pct = df_main['Returns'] * 100 # Rescale for better optimization
        
        # 1. CONFIGURATION
        # ---------------------------
        with st.expander("⚙️ Model Configuration", expanded=True):
            c_mdl1, c_mdl2, c_mdl3 = st.columns(3)
            with c_mdl1:
                vol_model_type = st.selectbox("Volatility Model", ["GARCH", "GJR-GARCH", "EGARCH"])
            with c_mdl2:
                dist_type = st.selectbox("Distribution", ["Normal", "Student's t", "Skewed Student's t"])
            with c_mdl3:
                vol_lag = st.slider("GARCH Lag (p, q)", 1, 3, 1)

        # Map inputs to arch arguments
        vol_map = {"GARCH": "Garch", "GJR-GARCH": "Garch", "EGARCH": "EGarch"}
        dist_map = {"Normal": "Normal", "Student's t": "t", "Skewed Student's t": "skewt"}
        
        o_param = 1 if vol_model_type == "GJR-GARCH" else 0
        
        # Fit Model
        try:
            am = arch_model(returns_pct, vol=vol_map[vol_model_type], p=vol_lag, o=o_param, q=vol_lag, dist=dist_map[dist_type])
            res = am.fit(disp='off')
            
            # 2. MAIN RESULTS DISPLAY
            # ---------------------------
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                st.subheader("Conditional Volatility")
                fig_v = go.Figure()
                fig_v.add_trace(go.Scatter(x=returns_pct.index, y=res.conditional_volatility, mode='lines', line=dict(color='#00f2ff', width=1.5), name=f'{vol_model_type} Vol'))
                fig_v.update_layout(title=f"{vol_model_type} ({dist_type}) Conditional Volatility", hovermode="x unified", template="plotly_dark", height=400)
                st.plotly_chart(fig_v, use_container_width=True)
                st.session_state.report_gen.add_plot("GARCH Volatility", fig_v)
                
            with col_res2:
                params_df = pd.DataFrame({
                    "Param": res.params.index,
                    "Value": res.params.values,
                    "t-stat": res.tvalues.values
                }).set_index("Param")
                st.subheader("Model Parameters")
                st.dataframe(params_df.style.format("{:.4f}"))
                st.session_state.report_gen.add_data("GARCH Parameters", params_df)
                
                st.markdown("### Analysis")
                
                # 1. Persistence & Half-Life
                # standard GARCH persistence = alpha + beta
                pers_val = np.nan
                if 'beta[1]' in res.params and 'alpha[1]' in res.params:
                    pers_val = res.params['alpha[1]'] + res.params['beta[1]']
                    if vol_model_type == 'GJR-GARCH' and 'gamma[1]' in res.params:
                         # GJR Persistence approx = alpha + beta + gamma/2
                         pers_val += res.params['gamma[1]'] / 2
                
                if not np.isnan(pers_val):
                    st.metric("Persistence", f"{pers_val:.4f}", help="Closer to 1 = Volatility shocks last longer.")
                    if pers_val < 1:
                        half_life = np.log(0.5) / np.log(pers_val)
                        st.metric("Half-Life (Days)", f"{half_life:.1f}", help="Days for a shock to initially decay by 50%.")
                    else:
                        st.caption("Non-stationary (Persistence >= 1)")

                # 2. Leverage Effect
                if 'gamma[1]' in res.params:
                    gamma_val = res.params['gamma[1]']
                    st.metric("Leverage (Gamma)", f"{gamma_val:.4f}")
                    if gamma_val > 0.05:
                        st.success("✅ Leverage Effect Confirmed: Market drops increase volatility more than rises.")
                    elif gamma_val < -0.05:
                        st.info("Inverse Leverage Structure.")
                    else:
                        st.caption("No significant asymmetry.")
                        
                st.markdown("---")
                st.metric("AIC", f"{res.aic:.2f}")
                st.metric("BIC", f"{res.bic:.2f}")

            # 3. DIAGNOSTICS & FORECASTING
            # ---------------------------
            tab_diag, tab_cast, tab_risk = st.tabs(["🔍 Diagnostics", "🔮 Forecasting", "🛡️ Risk Management"])
            
            # --- A. DIAGNOSTICS ---
            with tab_diag:
                d_col1, d_col2 = st.columns(2)
                
                std_resid = res.std_resid
                
                # 1. Standardized Residuals Plot
                with d_col1:
                    st.markdown("**Standardized Residuals**")
                    fig_r = go.Figure()
                    fig_r.add_trace(go.Scatter(x=returns_pct.index, y=std_resid, mode='lines', line=dict(color='gray', width=1.5), opacity=0.7, name="Std Resid"))
                    fig_r.add_hline(y=0, line_dash="dash", line_color="white")
                    fig_r.update_layout(title="Standardized Residuals", hovermode="x unified", template="plotly_dark", height=350)
                    st.plotly_chart(fig_r, use_container_width=True)
                    
                # 2. QQ Plot
                with d_col2:
                    st.markdown("**Q-Q Plot (vs Normal)**")
                    qq_tuple = stats.probplot(std_resid, dist="norm")
                    theoretical, observed = qq_tuple[0]
                    slope, intercept, r = qq_tuple[1]
                    trend_y = slope * theoretical + intercept
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(x=theoretical, y=observed, mode='markers', marker=dict(color='#00f2ff'), name='Data'))
                    fig_qq.add_trace(go.Scatter(x=theoretical, y=trend_y, mode='lines', line=dict(color='red'), name='Fit'))
                    fig_qq.update_layout(title="Q-Q Plot (vs Normal)", hovermode="closest", template="plotly_dark", height=350)
                    st.plotly_chart(fig_qq, use_container_width=True)
                    
                # 3. Serial Correlation Tests
                st.markdown("**Residual Diagnostics (Autocorrelation)**")
                lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
                arch_test = het_arch(std_resid)
                
                diag_data = {
                    "Test": ["Ljung-Box (No Serial Corr)", "ARCH-LM (No ARCH Effect)"],
                    "p-value": [lb_test['lb_pvalue'].iloc[0], arch_test[1]],
                    "Conclusion": [
                        "Fail to Reject H0 (Good)" if lb_test['lb_pvalue'].iloc[0] > 0.05 else "Reject H0 (Bad - Autocorr exists)",
                        "Fail to Reject H0 (Good)" if arch_test[1] > 0.05 else "Reject H0 (Bad - ARCH exists)"
                    ]
                }
                st.table(pd.DataFrame(diag_data).set_index("Test"))

            # --- B. FORECASTING ---
            with tab_cast:
                f_horizon = st.slider("Forecast Horizon (Days)", 1, 63, 21)
                
                try:
                    forecasts = res.forecast(horizon=f_horizon, reindex=False)
                except ValueError:
                    # Fallback for models/distributions where analytic is not supported (e.g. EGARCH/Skewt)
                    forecasts = res.forecast(horizon=f_horizon, method='simulation', simulations=1000, reindex=False)
                
                var_forecast = forecasts.variance.iloc[-1]
                vol_forecast = np.sqrt(var_forecast)
                
                st.write(f"**Volatility Forecast for next {f_horizon} days**")
                
                fig_f = go.Figure()
                last_days = 60
                hist_dates = returns_pct.index[-last_days:]
                hist_vol = res.conditional_volatility[-last_days:]
                
                fig_f.add_trace(go.Scatter(x=hist_dates, y=hist_vol, mode='lines', line=dict(color='gray', width=1.5), name='Historical Vol', opacity=0.8))
                
                fut_dates = [returns_pct.index[-1] + timedelta(days=i) for i in range(1, f_horizon+1)]
                fig_f.add_trace(go.Scatter(x=fut_dates, y=vol_forecast, mode='lines+markers', line=dict(color='red', dash='dash'), name='Forecast Vol'))
                
                fig_f.update_layout(title="Volatility Term Structure Forecast", hovermode="x unified", template="plotly_dark", height=400)
                st.plotly_chart(fig_f, use_container_width=True)
                
                # Term Structure Comment
                current_vol = res.conditional_volatility[-1]
                lt_vol = np.sqrt(res.params['omega'] / (1 - res.params['alpha[1]'] - res.params['beta[1]'])) if 'beta[1]' in res.params else current_vol
                
                if vol_forecast.iloc[-1] < current_vol:
                     st.success(f"Mean Reversion: Volatility expected to DECLINE towards long-term avg.")
                else:
                     st.warning(f"Mean Reversion: Volatility expected to RISE towards long-term avg.")

            # --- C. RISK MANAGEMENT ---
            with tab_risk:
                st.markdown("**Value at Risk (VaR) & Sizing**")
                
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    acc_size = st.number_input("Portfolio Value", 1000, 10000000, 100000)
                    conf_level = st.selectbox("Confidence Level", [0.95, 0.99])
                    
                    # Calculate VaR
                    # One-day ahead VaR based on model
                    # VaR = mean + sigma * q(alpha)
                    
                    next_vol = np.sqrt(forecasts.variance.iloc[-1].iloc[0]) / 100 # Convert back to decimal
                    
                    # Quantile depends on distribution
                    if dist_type == "Normal":
                        q = stats.norm.ppf(1-conf_level) # e.g. -1.645 for 95%
                    elif dist_type == "Student's t":
                        nu = res.params.get('nu')
                        q = stats.t.ppf(1-conf_level, df=nu)
                        
                    elif dist_type == "Skewed Student's t":
                         # Skewed T expects 2 parameters: nu (degree of freedom) and lambda (skew)
                         nu = res.params.get('nu')
                         lam = res.params.get('lambda')
                         
                         if nu is not None and lam is not None:
                             dist_inst = am.distribution
                             # arch distribution ppf expects params as a list/array
                             q = dist_inst.ppf(1-conf_level, [nu, lam])
                         else:
                             # Fallback to normal if params missing (unlikely if converged)
                             q = stats.norm.ppf(1-conf_level)

                    var_pct = -q * next_vol # Positive number representing loss
                    var_val = var_pct * acc_size
                    
                    st.metric(f"1-Day VaR ({conf_level:.0%})", f"{CURRENCY}{var_val:,.2f}", f"-{var_pct*100:.2f}%")
                    st.caption("Estimated maximum loss for tomorrow with selected confidence.")

                with r_col2:
                    target_vol = st.slider("Target Annual Volatility (%)", 5, 50, 15) / 100
                    
                    # Position Sizing
                    # Size = (Target Vol / Current Vol) * Capital
                    
                    current_ann_vol = next_vol * np.sqrt(252)
                    lev_factor = target_vol / current_ann_vol
                    rec_exposure = acc_size * lev_factor
                    
                    st.metric("Vol-Targeted Exposure", f"{CURRENCY}{rec_exposure:,.0f}", f"Leverage: {lev_factor:.2f}x")
                    
                    if lev_factor > 1.0:
                        st.warning("Requires Leverage (Margin)")
                    else:
                        st.success("Cash Position (Defensive)")

        except Exception as e:
            st.error(f"Model Fit Failed: {e}")
            st.info("Try a different distribution or simpler model (GARCH).")
            
    else:
        st.warning("⚠️ 'arch' library not found. Please run `pip install arch`.")

# ==========================================
# TAB 2: REGIME SWITCHING
# ==========================================
with tab2:
    if df_main is None:
        st.warning("Please load a ticker to view Regime Switching models.")
    else:
        st.write("### Markov Regime Switching Model")
    # --- MODEL VERDICT BANNER ---
    if "LONG" in regime_sig: st.success(f"🎯 **MODEL VERDICT**: Confirmed **{regime_sig}** in {regime_data['label']}. High conviction for bullish exposure.")
    elif "SHORT" in regime_sig: st.error(f"🎯 **MODEL VERDICT**: Confirmed **{regime_sig}**. Market risk is elevated.")
    else: st.info(f"🎯 **MODEL VERDICT**: {regime_sig}. Await confirmation of a clear regime shift.")

    st.markdown("""
    Identifies hidden market states (e.g., Bull vs Bear) from return dynamics.  
    Each regime has distinct mean return and volatility characteristics.
    """)
    st.markdown("[Reference: Regime Switching Models (James D. Hamilton)](https://econweb.ucsd.edu/~jhamilton/palgrave.pdf)")
    
    # ===== CONFIGURATION =====
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        # CHANGED: Default index 1 is Weekly (0=Daily, 1=Weekly)
        regime_freq = st.selectbox("Data Frequency", ["Daily", "Weekly"], index=1)
    with col_config2:
        # CHANGED: Default value 2
        lookback_years = st.slider("Lookback Period (Years)", 1, 10, 2)
    with col_config3:
        n_regimes = st.slider("Number of Regimes", 2, 4, 2)
    
    # New: Signal Stability Control
    # CHANGED: Default value 4
    stability = st.slider("Signal Stability (Pre-Smoothing)", 0, 10, 4, 
                          help="0 = Raw Data (Fastest), 10 = Very Smooth (Lagged). Higher values filter out noise.")
    
    # New: High-Conviction Threshold
    conviction_thresh = st.slider("High-Conviction Threshold", 0.5, 0.95, 0.7, step=0.05, 
                                  help="Minimum probability required to confirm a regime signal.")
    
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
    
    # FIX: Ensure data is strictly 1D Series with Float dtype
    try:
        model_data = returns.dropna() * 100
        
        # Reconstruct Series to guarantee 1D structure
        # This handles (N,1) DataFrames, Series, etc.
        if len(model_data) < 10:
            st.error("Insufficient data points for modeling (>10 required).")
            st.stop()
            
        model_data = pd.Series(
            model_data.values.flatten().astype(float), 
            index=model_data.index
        )
        
        if model_data.ndim != 1:
            st.error(f"Data dimensionality error: {model_data.ndim}D detected.")
            st.stop()
             
    except Exception as e:
        st.error(f"Data Prep Error: {e}")
        st.stop()
    
    st.caption(f"Modeling {len(model_data)} {regime_freq.lower()} returns from {start_dt_regime.date()}")
    
    # ===== MODEL FITTING =====
    with st.spinner(f"Fitting {n_regimes}-regime model..."):
        res_markov = fit_regime_model(model_data, n_regimes, switch_vol, switch_trend)
        
    if res_markov is None:
        st.error("Model fitting failed (fit_regime_model returned None).")
        st.stop()
        
    # Verify convergence implicitly via success return

        
    # ===== CONVERGENCE CHECKS =====
    if not res_markov.mle_retvals['converged']:
        st.error("⛔ Model did not converge. Try longer history or simpler model.")
        st.stop()
    
    trans_matrix = np.squeeze(res_markov.regime_transition)
    
    # Ensure it's at least 2D (handle edge case if squeeze over-squeezed a scalar? Unlikely for matrix)
    if trans_matrix.ndim < 2:
         trans_matrix = np.atleast_2d(trans_matrix)
    
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
    
    st.session_state.report_gen.add_data("Regime Statistics", pd.DataFrame(regime_stats))
    
    # ===== CURRENT STATE =====
    # Use .iloc[-1] to get the probabilities at the LAST time step
    last_probs = res_markov.filtered_marginal_probabilities.iloc[-1]
    current_regime = np.argmax(last_probs)
    current_prob = last_probs.iloc[current_regime]
    
    regime_label = labels[[r['regime'] for r in regime_stats].index(current_regime)]
    
    # Conviction Logic
    is_conviction = current_prob >= conviction_thresh
    
    # Calculate Stability Score (Mean Persistence)
    stability_score = np.mean([r['persistence'] for r in regime_stats])
    
    # Display Dashboard
    st.divider()
    c_dash1, c_dash2, c_dash3 = st.columns(3)
    
    with c_dash1:
        st.caption("Current State")
        if is_conviction:
            st.subheader(f"{regime_label}")
            st.success(f"High Conviction ({current_prob:.1%})")
        else:
            st.subheader("⚪ Mixed / Uncertain")
            st.warning(f"Low Conviction ({current_prob:.1%} < {conviction_thresh:.0%})")
    
    with c_dash2:
        st.caption("Dominance Score (Confidence)")
        # Spread between 1st and 2nd highest probability
        sorted_probs = sorted(last_probs.values, reverse=True)
        spread = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)
        
        st.metric("Probability Spread", f"{spread:.1%}", help="Difference between top 2 regime probabilities.")
        st.progress(max(0.0, min(1.0, float(spread))))
        
    with c_dash3:
        st.caption("Regime Stability Metrics")
        st.metric("Avg Persistence", f"{stability_score:.1%}")
        # Switch Frequency (proxy)
        expected_switches_per_year = (1 - stability_score) * (52 if regime_freq == "Weekly" else 252)
        st.caption(f"Exp. Switches/Year: ~{expected_switches_per_year:.1f}")

    st.write(f"**As of:** {model_data.index[-1].date()}")
    st.divider()
    
    # ===== VISUALIZATION =====
    st.write("### 📈 Regime Analysis (Real-time / Filtered)")
    
    fig_m = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                          subplot_titles=("Returns", "Regime Probabilities (Filtered/Real-time)", "Regime-Weighted Expected Return"))
    
    fig_m.add_trace(go.Scatter(x=model_data.index, y=model_data, mode='lines', line=dict(color='gray', width=1), name='Return (%)'), row=1, col=1)
    
    import matplotlib.colors as mcolors
    for i, regime in enumerate(regime_stats):
        color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
        hex_color = mcolors.to_hex(plt.cm.RdYlGn(color_idx))
        probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
        # Shading regions where prob > 0.6
        mask = probs > 0.6
        highlight_plotly_zones(fig_m, mask, hex_color, opacity=0.15, row=1, col=1)

    smooth_probs = st.checkbox("Smooth Probabilities (4-period Rolling)", value=True, key="smooth_probs_check")
    
    for i, regime in enumerate(regime_stats):
        color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
        hex_color = mcolors.to_hex(plt.cm.RdYlGn(color_idx))
        raw_probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
        if smooth_probs:
            plot_probs = raw_probs.rolling(window=4, min_periods=1).mean()
        else:
            plot_probs = raw_probs
        fig_m.add_trace(go.Scatter(x=model_data.index, y=plot_probs, mode='lines', line=dict(color=hex_color, width=1.5), fill='tozeroy', name=labels[i]), row=2, col=1)

    fig_m.add_hline(y=1/n_regimes, line_dash="dash", line_color="gray", opacity=0.4, row=2, col=1)
    
    def get_const(i):
        if f'const[{i}]' in res_markov.params: return float(res_markov.params[f'const[{i}]'])
        return float(res_markov.params.get('const', 0.0))

    expected_ret = pd.Series(0.0, index=model_data.index)
    for i in range(n_regimes):
        prob = res_markov.filtered_marginal_probabilities.iloc[:, i]
        expected_ret += prob * get_const(i)
    
    fig_m.add_trace(go.Scatter(x=model_data.index, y=expected_ret, mode='lines', line=dict(color='#00f2ff', width=2), name="Expected Return"), row=3, col=1)
    fig_m.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=3, col=1)
    
    # Fill Expected return green and red
    exp_pos = expected_ret.copy()
    exp_pos[exp_pos < 0] = 0
    fig_m.add_trace(go.Scatter(x=model_data.index, y=exp_pos, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='green', opacity=0.3, name="Positive Exp Ret", showlegend=False), row=3, col=1)
    
    exp_neg = expected_ret.copy()
    exp_neg[exp_neg > 0] = 0
    fig_m.add_trace(go.Scatter(x=model_data.index, y=exp_neg, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='red', opacity=0.3, name="Negative Exp Ret", showlegend=False), row=3, col=1)
    
    fig_m.update_layout(height=800, hovermode="x unified", template="plotly_dark", title="Regime Switching Analysis")
    st.plotly_chart(fig_m, use_container_width=True)
    st.session_state.report_gen.add_plot("Regime Switching Analysis", fig_m)
    
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
    



# ==========================================
# TAB 3: STOCHASTIC MODELS (Heston/Jump)
# ==========================================
with tab3:
    if df_main is None:
        st.warning("Please load a ticker to view Stochastic/Jump models.")
    else:
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
            st.session_state.report_gen.add_plot("Merton Jump Diffusion", fig)
            st.session_state.report_gen.add_data("Merton Metrics", {"Mean": final_mean, "Median": final_median})

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
            st.session_state.report_gen.add_plot("Heston Price Simulation", fig_h)
            st.session_state.report_gen.add_data("Heston Metrics", {"Mean": final_mean, "Median": final_median})
            
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
            st.session_state.report_gen.add_plot("Heston Volatility Process", fig_v)

# ==========================================
# TAB 4: KALMAN FILTER
# ==========================================
with tab4:
    if df_main is None:
        st.warning("Please load a ticker to view Kalman Filter dynamics.")
    else:
        st.write("### Kalman Filter Analysis")
    # --- MODEL VERDICT BANNER ---
    if trend_diff > 0.03: st.success(f"🎯 **MODEL VERDICT**: Price is **{trend_diff:.1%} ABOVE** the Kalman Trend. Structural uptrend intact.")
    elif trend_diff < -0.03: st.error(f"🎯 **MODEL VERDICT**: Price is **{abs(trend_diff):.1%} BELOW** the Kalman Trend. Structural breakdown in progress.")
    else: st.info(f"🎯 **MODEL VERDICT**: Price is trading within **{abs(trend_diff):.1%}** of the Kalman Trend (Neutral/Consolidation).")

    
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
                fig_k = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Kalman Estimated Hedge Ratio (Beta)", "Kalman Residual Z-Score (Mean Reversion Signal)"))
                fig_k.add_trace(go.Scatter(x=dates, y=beta, mode='lines', line=dict(color='#00f2ff', width=1.5), name=f"Dynamic Beta ({TICKER}/{PAIR_TICKER})"), row=1, col=1)
                fig_k.add_trace(go.Scatter(x=dates, y=z_score, mode='lines', line=dict(color='purple', width=1.5), name="Spread Z-Score"), row=2, col=1)
                fig_k.add_hline(y=2.0, line_dash="dash", line_color="red", row=2, col=1)
                fig_k.add_hline(y=-2.0, line_dash="dash", line_color="green", row=2, col=1)
                fig_k.update_layout(height=600, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig_k, use_container_width=True)
                st.session_state.report_gen.add_plot("Kalman Pairs Analysis", fig_k)
                st.session_state.report_gen.add_data("Kalman Hedge Ratio", {"Beta": beta[-1]})
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
        
        fig_kt = go.Figure()
        fig_kt.add_trace(go.Scatter(x=df_main.index, y=prices, mode='lines', line=dict(color='gray'), opacity=0.5, name='Actual Price'))
        
        if model_mode == "Compare Both":
            fig_kt.add_trace(go.Scatter(x=df_main.index, y=est_trend_std, mode='lines', line=dict(color='blue', dash='dash', width=1.5), name='Standard (Causal)'))
            fig_kt.add_trace(go.Scatter(x=df_main.index, y=est_trend_smooth, mode='lines', line=dict(color='purple', width=2), name='Smoothed (RTS)'))
            current_trend = est_trend_smooth[-1] # Use smooth for metrics
        else:
            fig_kt.add_trace(go.Scatter(x=df_main.index, y=est_trend, mode='lines', line=dict(color=color_trend, width=2), name=label_trend))
            current_trend = est_trend[-1]
            
        fig_kt.update_layout(title=f"Kalman Filter Trend: {TICKER}", hovermode="x unified", template="plotly_dark", height=500)
        st.plotly_chart(fig_kt, use_container_width=True)
        st.session_state.report_gen.add_plot("Kalman Trend Analysis", fig_kt)
        
        # Signal & Metrics
        current_price = prices[-1]
        diff_pct = (current_price - current_trend) / current_trend * 100

        st.session_state.report_gen.add_data("Kalman Trend Metrics", {"Price": current_price, "Trend": current_trend, "Deviation": diff_pct})
        
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
    if df_main is None:
        st.warning("Please load a ticker to view Factor analysis.")
    else:
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
        
        fig_hm = go.Figure(data=go.Heatmap(
                   z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.columns,
                   colorscale='RdBu',
                   zmin=-1, zmax=1,
                   text=np.round(corr_matrix.values, 2),
                   texttemplate="%{text}",
                   hoverinfo="x+y+z"))
        fig_hm.update_layout(title="Asset Class Correlations", template="plotly_dark", width=600, height=600)
        st.plotly_chart(fig_hm, use_container_width=True)
        st.session_state.report_gen.add_plot("Macro Correlations", fig_hm)
        st.session_state.report_gen.add_data("Correlation Matrix", corr_matrix)
        
        st.write(f"**Structural Thesis Check:**")
        oil_corr = corr_matrix.loc[TICKER, 'Crude Oil']
        rate_corr = corr_matrix.loc[TICKER, '10Y Yield']
        
        if oil_corr > 0.3:
            st.success(f"High correlation with Energy ({oil_corr:.2f}). Commodity cycle model relevant.")
        elif oil_corr < -0.3:
            st.info(f"Inverse correlation with Energy ({oil_corr:.2f}).")
        else:
            st.warning(f"Low sensitivity to Energy prices ({oil_corr:.2f}).")
        
        st.session_state.report_gen.add_data("Macro Sensitivity Thesis", {
            "Oil Correlation": oil_corr,
            "Rate Correlation": rate_corr
        })

# ==========================================
# TAB 6: STRUCTURAL
# ==========================================
with tab6:
    if df_main is None:
        st.warning("Please load a ticker to view Structural Decomposition.")
    else:
        st.write("### Structural Decomposition")
    # Need freq for decomposition. 
    # Business days ~ 5 (weekly), 21 (monthly), 252 (yearly)
    period = st.selectbox("Seasonality Period", [5, 21, 63, 252], index=1)
    
    if len(df_main) > period * 2:
        decomp = seasonal_decompose(df_main['Close'], model='multiplicative', period=period)
        
        fig_dec = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Trend', 'Seasonal Component', 'Residuals'))
        fig_dec.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, mode='lines', name='Trend'), row=1, col=1)
        fig_dec.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, mode='lines', name='Seasonal'), row=2, col=1)
        fig_dec.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, mode='lines', name='Residuals'), row=3, col=1)
        fig_dec.update_layout(height=800, hovermode="x unified", template="plotly_dark", title="Structural Decomposition")
        st.plotly_chart(fig_dec, use_container_width=True)
        st.session_state.report_gen.add_plot("Structural Decomposition", fig_dec)
        st.session_state.report_gen.add_data("Decomposition Period", {"Period": period})
    else:
        st.warning("Insufficient data for decomposition with selected period.")

# ==========================================
# TAB 7: BACKTEST
# ==========================================
with tab7:
    if df_main is None:
        st.warning("Please load a ticker to run Backtests.")
    else:
        st.write("### 🛠️ Strategy Backtest")
    
    # Strategy Selector
    strategy_type = st.radio("Select Strategy", ["Regime Switching (Trend Following)", "Kalman Filter (Trend Crossover)", "Momentum Hedge (EMA/SMA Cross)", "MAD Trend Modes", "Dual MA Cross", "Ehlers SuperSmoother", "Ehlers Simple Decycler", "Institutional Mean Reversion (Z-Score)", "Relative Strength Ratio (vs Benchmark)", "Implied Volatility Proxy (^VIX)", "Institutional Hurst Exponent"], horizontal=True)
    
    # Date Selection
    col_b3 = st.container()
    with col_b3:
        default_start = DEFAULT_NONLIVE_START
        bt_start_date = st.date_input("Backtest Start", default_start)
        bt_end_date = st.date_input("Backtest End", datetime.now())

    # Data Prep
    if live_mode:
         # Use the global live data for backtest scope
         df_bt = df_main
         bt_msg = f"Live Backtest ({data_interval})"
    else:
        if bt_start_date >= bt_end_date:
            st.error("Start date must be before end date.")
            st.stop()
        df_bt = load_data(TICKER, bt_start_date, bt_end_date, interval='1d')
        bt_msg = "Historical Backtest (1d)"
    
    if df_bt is None or df_bt.empty:
        st.error("Could not load data for backtest. Check dates and ticker.")
        st.stop()
        
    returns_bt = df_bt['Returns']
    prices_bt = df_bt['Close']
    model_data_bt = returns_bt.dropna() * 100
    
    signals = None
    strat_prices = prices_bt
    benchmark_label_for_metrics = "Benchmark Return"
    full_period_benchmark_pct_for_metrics = np.nan
    using_wfo_primary_for_metrics = False
    
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

        col_sig1, col_sig2, col_sig3 = st.columns(3)
        with col_sig1:
            conviction = st.slider("Min Bull Probability", 0.5, 0.9, 0.65, step=0.05, key="bt_regime_conviction")
        with col_sig2:
            min_hold_period = st.number_input("Minimum Hold Period", min_value=1, max_value=60, value=3, step=1, key="bt_regime_min_hold")
        with col_sig3:
            confirmed_regime_bar = st.checkbox("Confirmed-bar execution", value=True, key="bt_regime_confirmed_bar")

        with st.expander("🧭 Regime WFO Settings", expanded=True):
            wf_c1, wf_c2, wf_c3, wf_c4, wf_c5 = st.columns(5)
            enable_regime_wfo = wf_c1.checkbox("Enable Regime WFO", value=True, key="bt_regime_enable_wfo")
            use_regime_wfo = wf_c2.checkbox("Use WFO as main result", value=True, key="bt_regime_use_wfo")
            regime_wf_train = wf_c3.number_input("Regime WFO train bars", min_value=30, max_value=1000, value=126, step=21, key="bt_regime_wf_train")
            regime_wf_forward = wf_c4.number_input("Regime WFO forward bars", min_value=5, max_value=252, value=42, step=5, key="bt_regime_wf_forward")
            regime_activity_mode = wf_c5.selectbox("Regime WFO activity", ["Conservative", "Balanced", "Active"], index=1, key="bt_regime_activity_mode", help="Conservative = original Markov behavior. Balanced = Markov plus moderate trend-pulse. Active = faster trend-pulse candidates for more trades.")
            auto_wfo_regimes = st.checkbox("Auto-select regimes inside WFO (2/3/4)", value=True, key="bt_regime_auto_wfo_regimes", help="When ON, each walk-forward training window chooses the best number of regimes from 2, 3, or 4 using only past data.")
            use_regime_runner_override = st.checkbox("Benchmark-aware strong runner override", value=True, key="bt_regime_runner_override", help="Adds a trend-hold candidate so very strong stocks are not forced into defensive cash too often.")
            use_regime_return_booster = st.checkbox("Benchmark-aware return booster", value=True, key="bt_regime_return_booster", help="Adds a fractional trend-participation candidate that tries to get closer to buy-and-hold while still exiting on trend breaks.")
            regime_return_booster_mode = st.selectbox("Return booster mode", ["Conservative", "Balanced", "Aggressive", "Benchmark Chase", "Full Benchmark Capture", "Optimized Full Capture", "Maximum Capture"], index=5, key="bt_regime_return_booster_mode", help="Optimized Full Capture tests several causal trend-capture rules and chooses the one with the best return/drawdown/Sharpe balance. It is designed to get closer to the full benchmark without accepting ugly drawdowns.")
            regime_full_benchmark_mode = st.checkbox("Compare/trade from full start date", value=True, key="bt_regime_full_benchmark_mode", help="When ON, the metric section uses the full selected date range instead of only the WFO test window. Before the first WFO period, it uses a causal trend bridge so the strategy can be compared against the full benchmark.")

        if signal_method == "Regime Weighted Expected Return":
            st.markdown("**Strategy:** Long when expected return is positive **and** Bull Probability is above the conviction threshold.")
        elif signal_method == "Regime Probability":
            st.markdown("**Strategy:** Long when **Bull Probability > Min Bull Probability**. This uses the exact conviction threshold.")
        else:
            st.markdown("**Strategy:** Long when Bull Regime is dominant and above conviction, then hold for at least the minimum hold period before switching.")

        # --- MODEL FITNESS INFO ---
        st.info("💡 **Pro Tip**: Blue-chips often favor **2 states** (Bull/Bear). High-beta tech often favors **3 states** (Bull/Bear/Consolidation). Use the 'Compare Fitness' button below to find the best fit.")

        if st.button("📊 Compare Regime Fitness (N=2,3,4)", use_container_width=True):
            with st.spinner("Analyzing model complexity and performance..."):
                comp_results = []
                # Setup local data context
                loc_prices = prices_bt.resample('W-FRI').last().dropna() if bt_freq == "Weekly" else prices_bt
                loc_returns = loc_prices.pct_change().dropna()
                if bt_stability > 0:
                    loc_model_data = loc_returns.ewm(span=bt_stability, adjust=False).mean().dropna() * 100
                else:
                    loc_model_data = loc_returns.dropna() * 100
                
                for n in [2, 3, 4]:
                    r = fit_regime_model(loc_model_data, n, bt_switch_vol, bt_switch_trend)
                    if r:
                        # 1. Identify Bull Regime
                        r_means = []
                        for i in range(n):
                            m_val = r.params[f'const[{i}]'] if f'const[{i}]' in r.params else r.params.get('const', 0.0)
                            r_means.append((i, m_val))
                        bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
                        
                        # 2. Generate Signals with conviction/min-hold logic
                        sigs, _ = build_regime_backtest_signal(
                            r,
                            loc_model_data.index,
                            loc_prices.index,
                            n,
                            signal_method,
                            conviction=float(conviction),
                            min_hold=int(min_hold_period)
                        )
                        if confirmed_regime_bar:
                            sigs = sigs.shift(1).ffill().fillna(0).clip(0, 1)

                        # 3. Run Backtest
                        common_idx = loc_prices.index.intersection(sigs.index)
                        bt_res = BacktestEngine.run_strategy(loc_prices.loc[common_idx], sigs.loc[common_idx], initial_cap, trailing_stop, stop_loss)
                        
                        comp_results.append({
                            "Regimes": n, 
                            "AIC": r.aic, 
                            "BIC": r.bic, 
                            "Total Return %": (bt_res['equity_curve'].iloc[-1] / initial_cap - 1) * 100
                        })
                
                if comp_results:
                    comp_df = pd.DataFrame(comp_results)
                    best_aic = comp_df.loc[comp_df['AIC'].idxmin(), 'Regimes']
                    best_bic = comp_df.loc[comp_df['BIC'].idxmin(), 'Regimes']
                    best_pnl = comp_df.loc[comp_df['Total Return %'].idxmax(), 'Regimes']
                    
                    st.write("#### Comparison Results")
                    st.table(comp_df.style.highlight_min(subset=['AIC', 'BIC'], color='lightgreen')
                                       .highlight_max(subset=['Total Return %'], color='lightgreen'))
                    
                    c_fit, c_perf = st.columns(2)
                    with c_fit:
                        st.success(f"⚖️ **Robustness**: {best_bic} Regimes (Best BIC)")
                    with c_perf:
                        st.success(f"🚀 **Performance**: {best_pnl} Regimes (Best PnL)")
                    
                    if best_bic != best_pnl:
                        st.warning(f"⚠️ **Conflict**: Statistical health prefers **{best_bic}**, but historical PnL was higher with **{best_pnl}**. Be careful fitting to the highest return—it often leads to overfitting!")


        # Resample if Weekly
        if bt_freq == "Weekly":
            # Resample Prices to Weekly (Last Close)
            prices_bt_resampled = prices_bt.resample('W-FRI').last().dropna()
        else:
            prices_bt_resampled = prices_bt.dropna()

        # Apply smoothing consistently to prices first, then calculate returns from those same prices.
        # This fixes the old mismatch where model data was smoothed but execution prices were raw.
        if bt_stability > 0:
            prices_bt_model = prices_bt_resampled.ewm(span=bt_stability, adjust=False).mean().dropna()
            st.caption(f"ℹ️ Regime smoothing applied consistently to price and model returns (span={bt_stability}).")
        else:
            prices_bt_model = prices_bt_resampled.copy()

        strat_prices = prices_bt_model
        returns_bt_resampled = prices_bt_model.pct_change().dropna()
        model_data_bt = returns_bt_resampled.dropna() * 100

        # FIX: Robust 1D Series reconstruction
        if len(model_data_bt) > 5: # Slightly lower threshold for very recent live data
            model_data_bt = pd.Series(
                model_data_bt.values.flatten().astype(float),
                index=model_data_bt.index
            )
        
        if len(model_data_bt) < 10:
             st.error(f"❌ **Backtest Error: Insufficient data found for model.** (Points: {len(model_data_bt)})")
             st.info(f"The Markov Regime model needs at least 15-20 data points to converge. Currently, your dataset has only {len(model_data_bt)} points after resampling/smoothing.")
             if live_mode and bt_freq == "Weekly":
                 st.warning("💡 **Hint**: You are using 'Weekly' frequency on intraday data. Switch back to 'Daily' (Raw Intraday) to use all live candles for the model.")
             elif not live_mode:
                 st.warning("💡 **Hint**: Try increasing your backtest date range in the sidebar.")
        else:
            with st.spinner("Fitting Regime Model..."):
                # Fit Model
                res_bt = fit_regime_model(model_data_bt, bt_n_regimes, bt_switch_vol, bt_switch_trend)
                
                if res_bt:
                    # --- DISPLAY FITNESS METRICS ---
                    fit_col1, fit_col2 = st.columns(2)
                    with fit_col1:
                        st.caption(f"Model Fitness (AIC): **{res_bt.aic:.1f}**")
                    with fit_col2:
                        st.caption(f"Model Fitness (BIC): **{res_bt.bic:.1f}**")
                    st.caption("Lower is better. Compare these across 2, 3, or 4 regimes to find the mathematical 'Best Fit'.")
                    
                    # Build selected-method full-history signal using conviction + min-hold logic
                    signals, regime_context = build_regime_backtest_signal(
                        res_bt,
                        model_data_bt.index,
                        strat_prices.index,
                        int(bt_n_regimes),
                        signal_method,
                        conviction=float(conviction),
                        min_hold=int(min_hold_period)
                    )

                    # Price trend override for strong runners
                    price_override = get_price_trend_override(
                        strat_prices.index,
                        model_data_bt.index,
                        strat_prices
                    )
                    signals = pd.Series(
                        np.maximum(signals.values, price_override.reindex(signals.index).fillna(0).values),
                        index=signals.index
                    ).clip(0, 1)

                    if confirmed_regime_bar:
                        signals = signals.shift(1).ffill().fillna(0).clip(0, 1)

                    # --- Walk-forward validation / primary signal ---
                    if enable_regime_wfo:
                        with st.spinner("Running Regime Walk-Forward Optimization..."):
                            wf_regime = walk_forward_regime_selection(
                                strat_prices,
                                returns_bt_resampled,
                                n_regimes="Auto" if bool(auto_wfo_regimes) else int(bt_n_regimes),
                                switch_vol=bool(bt_switch_vol),
                                switch_trend=bool(bt_switch_trend),
                                train_window=int(regime_wf_train),
                                forward_window=int(regime_wf_forward),
                                conviction=float(conviction),
                                min_hold=int(min_hold_period),
                                initial_capital=initial_cap,
                                trailing_stop_pct=trailing_stop,
                                stop_loss_pct=stop_loss,
                                confirmed_bar=bool(confirmed_regime_bar),
                                use_strong_runner_override=bool(use_regime_runner_override),
                                activity_mode=str(regime_activity_mode),
                                use_return_booster=bool(use_regime_return_booster),
                                return_booster_mode=str(regime_return_booster_mode)
                            )

                        st.write("#### 🧭 Regime Walk-Forward Result")
                        if wf_regime is None or wf_regime.get("overall") is None:
                            st.warning("Regime WFO could not generate a valid out-of-sample result for this data window. Showing the selected full-history regime signal below so the tab does not go blank. Treat it as research, not WFO-validated.")
                            using_wfo_primary_for_metrics = False
                        else:
                            wf_overall = wf_regime["overall"]
                            eff_train = wf_regime.get("effective_train_window", regime_wf_train)
                            eff_forward = wf_regime.get("effective_forward_window", regime_wf_forward)
                            if int(eff_train) != int(regime_wf_train) or int(eff_forward) != int(regime_wf_forward):
                                st.caption(f"ℹ️ WFO auto-adjusted to Train={int(eff_train)} bars / Forward={int(eff_forward)} bars because the selected data window was shorter than requested.")
                            full_bh = buy_hold_return_pct(strat_prices)
                            wfc1, wfc2, wfc3, wfc4, wfc5 = st.columns(5)
                            wfc1.metric("WF Strategy Return", f"{wf_overall['Strategy Return %']:.2f}%")
                            wfc2.metric("WF Test Benchmark", f"{wf_overall['Buy & Hold Return %']:.2f}%", help="Buy & hold only over the out-of-sample WFO test window.")
                            wfc3.metric("Full Benchmark", f"{full_bh:.2f}%" if pd.notna(full_bh) else "N/A", help="Buy & hold over the full selected period. Reference only.")
                            wfc4.metric("WF Difference", f"{wf_overall['Difference %']:+.2f}%")
                            wfc5.metric("WF Stability", f"{wf_regime['stability_score']:.0f}/100")

                            if wf_overall['Difference %'] > 0 and wf_regime['stability_score'] >= 60:
                                st.success("Regime WFO is positive and reasonably stable.")
                            elif wf_overall['Difference %'] > 0:
                                st.warning("Regime WFO beat its test benchmark, but stability is not strong. Use confirmation.")
                            else:
                                st.warning("Regime WFO did not beat buy & hold on unseen windows. If the benchmark is extremely high, the stock is a strong runner and defensive regime exits may still lag buy-and-hold.")

                            wf_rows = wf_regime["rows"].copy()
                            if not wf_rows.empty:
                                for col in ["Train Start", "Train End", "Forward Start", "Forward End"]:
                                    wf_rows[col] = pd.to_datetime(wf_rows[col]).dt.date
                                st.dataframe(wf_rows.sort_values("Period", ascending=False), use_container_width=True)

                            if use_regime_wfo:
                                first_forward_start = wf_regime["first_forward_start"]
                                full_wfo_prices = strat_prices.copy()

                                if bool(regime_full_benchmark_mode):
                                    # Full-benchmark mode: compare the strategy against buy & hold from the
                                    # selected start date, not only after the WFO training window.
                                    # WFO cannot produce a model-selected signal before the first forward period,
                                    # so the pre-WFO section uses a causal trend bridge. This is clearly labeled
                                    # as a bridge, not as out-of-sample WFO validation.
                                    wf_only_signal = wf_regime["signal"].reindex(full_wfo_prices.index).ffill().fillna(0).clip(0, 1)
                                    bridge_signal = benchmark_aware_trend_participation_signal(
                                        full_wfo_prices, mode=str(regime_return_booster_mode)
                                    ).reindex(full_wfo_prices.index).ffill().fillna(0).clip(0, 1)
                                    if bool(confirmed_regime_bar):
                                        bridge_signal = bridge_signal.shift(1).ffill().fillna(0).clip(0, 1)

                                    signals = wf_only_signal.copy()
                                    pre_wfo_mask = signals.index < first_forward_start
                                    signals.loc[pre_wfo_mask] = bridge_signal.loc[pre_wfo_mask]
                                    strat_prices = full_wfo_prices
                                    benchmark_label_for_metrics = "Full Benchmark"
                                    full_period_benchmark_pct_for_metrics = full_bh
                                    using_wfo_primary_for_metrics = True
                                    try:
                                        st.caption(f"ℹ️ Full-benchmark mode ON: {int(pre_wfo_mask.sum())} pre-WFO bars use causal trend bridge; later bars use WFO-selected signal.")
                                    except Exception:
                                        pass
                                else:
                                    # Pure WFO-test mode: compare only over the out-of-sample forward-test window.
                                    strat_prices = strat_prices.loc[first_forward_start:]
                                    signals = wf_regime["signal"].reindex(strat_prices.index).ffill().fillna(0).clip(0, 1)
                                    benchmark_label_for_metrics = "WFO Test Benchmark"
                                    full_period_benchmark_pct_for_metrics = full_bh
                                    using_wfo_primary_for_metrics = True

                                # DIRECT RETURN-BOOSTER APPLICATION TO THE FINAL METRIC SIGNAL
                                # Previous versions could show identical results because the booster was only
                                # a WFO candidate or overlay inside each block. If WFO selected the same
                                # base regime exposure, the final BacktestEngine.run_strategy() still received
                                # the old signal. This block applies the booster to the exact `signals` series
                                # used by the performance metrics and trade log below.
                                if bool(use_regime_return_booster):
                                    try:
                                        booster_full = benchmark_aware_trend_participation_signal(
                                            strat_prices, mode=str(regime_return_booster_mode)
                                        ).reindex(strat_prices.index).ffill().fillna(0).clip(0, 1)
                                        if bool(confirmed_regime_bar):
                                            booster_full = booster_full.shift(1).ffill().fillna(0).clip(0, 1)

                                        mode_l = str(regime_return_booster_mode or "Balanced").lower()
                                        original_signals = signals.copy()
                                        if ("optimized" in mode_l) or ("full benchmark" in mode_l) or ("maximum" in mode_l) or (mode_l == "aggressive"):
                                            # Full Benchmark Capture / Maximum Capture must become the primary
                                            # final signal, otherwise the WFO signal can still keep returns far
                                            # below the full buy-and-hold benchmark.
                                            signals = booster_full
                                        elif mode_l == "conservative":
                                            # Conservative only adds high-confidence exposure.
                                            high_conf = booster_full.where(booster_full >= 0.75, 0.0)
                                            signals = pd.concat([signals, high_conf], axis=1).max(axis=1).clip(0, 1)
                                        else:
                                            # Balanced: combine WFO regime signal with benchmark-aware trend hold.
                                            signals = pd.concat([signals, booster_full], axis=1).max(axis=1).clip(0, 1)

                                            # If the max overlay is still identical, force the booster as the
                                            # final signal because the user explicitly enabled the return booster.
                                            # This prevents the exact same metrics problem.
                                            changed_bars = int((signals.round(6) != original_signals.round(6)).sum())
                                            if changed_bars == 0:
                                                signals = booster_full
                                                st.caption("ℹ️ Return booster matched the WFO signal, so the booster was used as the final signal to make the mode actually affect trades/metrics.")
                                            else:
                                                st.caption(f"ℹ️ Return booster changed {changed_bars} bars in the final backtest signal.")

                                        signals = pd.Series(signals, index=strat_prices.index).ffill().fillna(0).clip(0, 1)
                                        try:
                                            st.caption(f"ℹ️ Return booster final average exposure: {signals.mean()*100:.1f}%")
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        st.warning(f"Return booster final overlay could not be applied: {e}")

                                # benchmark_label_for_metrics / full_period_benchmark_pct_for_metrics
                                # are set above depending on whether full-benchmark mode is ON.
                                using_wfo_primary_for_metrics = True

                    # Plot Context
                    with st.expander("See Strategy Context"):
                        fig_ctx = go.Figure()
                        if signal_method == "Regime Weighted Expected Return" and "expected_ret" in regime_context:
                            expected_ret = regime_context["expected_ret"].reindex(strat_prices.index).ffill()
                            fig_ctx.add_trace(go.Scatter(x=expected_ret.index, y=expected_ret, mode='lines', line=dict(color='purple', width=1.5), name='Expected Return'))
                            fig_ctx.add_hline(y=0, line_dash="dash", line_color="white")
                            highlight_plotly_zones(fig_ctx, expected_ret > 0, 'green', opacity=0.2)
                            highlight_plotly_zones(fig_ctx, expected_ret < 0, 'red', opacity=0.2)
                            fig_ctx.update_layout(title="Regime-Weighted Expected Return + Conviction Filter", hovermode="x unified", template="plotly_dark", height=400)
                        else:
                            bull_probs = regime_context.get("bull_probs", pd.Series(dtype=float)).reindex(strat_prices.index).ffill()
                            fig_ctx.add_trace(go.Scatter(x=bull_probs.index, y=bull_probs, mode='lines', line=dict(color='green', width=1.5), name='Bull Probability'))
                            fig_ctx.add_hline(y=float(conviction), line_dash="dash", line_color="white", annotation_text="Min Bull Probability")
                            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.15)
                            fig_ctx.update_layout(title=f"{signal_method} (Conviction={conviction:.0%}, Min Hold={int(min_hold_period)})", hovermode="x unified", template="plotly_dark", height=400)
                        st.plotly_chart(fig_ctx, use_container_width=True)

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
                fig_ctx = go.Figure()
                fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
                fig_ctx.add_trace(go.Scatter(x=trend_series.index, y=trend_series, mode='lines', line=dict(color='blue'), name='Kalman Trend'))
                
                highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
                
                fig_ctx.update_layout(title="Strategy Context", hovermode="x unified", template="plotly_dark", height=400)
                st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Momentum Hedge (EMA/SMA Cross)":
        st.markdown("**Strategy:** Long when **Short EMA > Med SMA**. Cash/Hedge when **Short EMA < Med SMA**.")
        
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            short_len = st.slider("Short EMA Length", 5, 50, 20)
        with c_h2:
            med_len = st.slider("Medium SMA Length", 20, 200, 60)
            
        with st.spinner("Calculating Momentum Hedge Signals..."):
            # Calculate Indicators
            short_ema = prices_bt.ewm(span=short_len, adjust=False).mean()
            med_sma = prices_bt.rolling(window=med_len).mean()
            
            # Generate Signals
            # Logic: Flag = 1 when Short < Med (Hedge active). So Long Signal = 1 when Not Flagged (Short >= Med).
            # To match exactly: flag = shortMA < medMA ? 1.0 : 0.0
            signals = (short_ema >= med_sma).astype(int)
            
            # Plot Context
            with st.expander("See Strategy Context"):
                fig_ctx = go.Figure()
                fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
                fig_ctx.add_trace(go.Scatter(x=short_ema.index, y=short_ema, mode='lines', line=dict(color='orange', width=1.5), name=f'Short EMA ({short_len})'))
                fig_ctx.add_trace(go.Scatter(x=med_sma.index, y=med_sma, mode='lines', line=dict(color='blue', width=1.5), name=f'Med SMA ({med_len})'))
                
                highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
                highlight_plotly_zones(fig_ctx, signals == 0, 'red', opacity=0.1)
                
                fig_ctx.update_layout(title="Momentum Hedge Signal (EMA/SMA Cross)", hovermode="x unified", template="plotly_dark", height=400)
                st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "MAD Trend Modes":
        st.markdown("### 📊 MAD Trend Modes Settings")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            sig_mode = st.selectbox("Signal Mode", ["Bollinger Bands", "For Loop", "Combined Signal"])
            ma_type = st.selectbox("MA Type", ["EMA", "SMA", "WMA", "HMA", "RMA", "ALMA", "LSMA"])
        with col_m2:
            mad_len = st.number_input("MAD Length", 5, 100, 25)
            
        mad_params = {'signal_mode': sig_mode, 'bb_ma_type': ma_type, 'bb_len': mad_len}
        
        if sig_mode == "Bollinger Bands":
            col_bb1, col_bb2 = st.columns(2)
            with col_bb1:
                mult_p = st.number_input("+ Multiplier", 0.1, 5.0, 1.4)
            with col_bb2:
                mult_n = st.number_input("- Multiplier", 0.1, 5.0, 1.0)
            mad_params.update({'bb_mult_p': mult_p, 'bb_mult_n': mult_n})
            
        elif sig_mode == "For Loop":
            col_fl1, col_fl2, col_fl3 = st.columns(3)
            with col_fl1:
                fl_a = st.number_input("From", 1, 100, 10)
            with col_fl2:
                fl_b = st.number_input("To", 1, 200, 60)
            with col_fl3:
                fl_len = st.number_input("Loop MA Length", 1, 50, 10)
            col_fl4, col_fl5 = st.columns(2)
            with col_fl4:
                thresh_l = st.number_input("Threshold Long", 1, 100, 23)
            with col_fl5:
                thresh_s = st.number_input("Threshold Short", -100, 100, 3)
            mad_params.update({
                'fl_ma_type': ma_type,
                'fl_len': fl_len,
                'fl_a': fl_a,
                'fl_b': fl_b,
                'fl_thresh_l': thresh_l,
                'fl_thresh_s': thresh_s
            })
        else: # Combined Signal
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                c_thresh_l = st.number_input("Threshold Long (Combined)", -1.0, 1.0, 0.0, step=0.01)
            with col_c2:
                c_thresh_s = st.number_input("Threshold Short (Combined)", -1.0, 1.0, 0.0, step=0.01)
            mad_params.update({
                'c_thresh_l': c_thresh_l,
                'c_thresh_s': c_thresh_s
            })

        # Auto-run MAD Trend Backtest
        signals = MADTrendModes.get_signals(df_bt, mad_params)
        
        # Plot Context
        with st.expander("See Strategy Context", expanded=True):
            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
            
            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
            
            fig_ctx.update_layout(title=f"MAD Trend Modes Signal ({sig_mode})", hovermode="x unified", template="plotly_dark", height=400)
            st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Dual MA Cross":
        st.markdown("### 🔀 Dual Moving Average Cross Settings")
        ma_options = ["SMA", "EMA", "WMA", "HMA", "RMA", "ALMA", "LSMA"]
        
        c_ma1, c_ma2 = st.columns(2)
        with c_ma1:
            st.subheader("Fast MA (Short-term)")
            f_ma_type = st.selectbox("Fast MA Type", ma_options, index=1) # Default EMA
            f_ma_len = st.number_input("Fast MA Length", 1, 250, 20)
        with c_ma2:
            st.subheader("Slow MA (Long-term)")
            s_ma_type = st.selectbox("Slow MA Type", ma_options, index=0) # Default SMA
            s_ma_len = st.number_input("Slow MA Length", 1, 250, 50)
            
        if f_ma_len >= s_ma_len:
            st.warning("Fast MA length is typically shorter than Slow MA length. Results may be inverted.")
            
        # Auto-run Dual MA
        # Calculate MAs
        fast_ma = MADTrendModes.ma_switch(prices_bt, f_ma_len, f_ma_type)
        slow_ma = MADTrendModes.ma_switch(prices_bt, s_ma_len, s_ma_type)
        
        # Generate Signals: Long when Fast > Slow, Cash when Fast < Slow
        # Using stateful ffill logic for consistency
        long_cond = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        short_cond = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        def get_stateful_ma_signal(l_cond, s_cond, index):
            sig = pd.Series(np.nan, index=index)
            sig.loc[l_cond] = 1
            sig.loc[s_cond] = 0
            return sig.ffill().fillna(0)
        
        signals = get_stateful_ma_signal(long_cond, short_cond, prices_bt.index)
        
        # Plot Context
        with st.expander("See Strategy Context", expanded=True):
            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
            fig_ctx.add_trace(go.Scatter(x=fast_ma.index, y=fast_ma, mode='lines', line=dict(color='orange'), opacity=0.8, name=f'Fast {f_ma_type} ({f_ma_len})'))
            fig_ctx.add_trace(go.Scatter(x=slow_ma.index, y=slow_ma, mode='lines', line=dict(color='blue'), opacity=0.8, name=f'Slow {s_ma_type} ({s_ma_len})'))
            
            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
            
            fig_ctx.update_layout(title=f"Dual MA Cross: {f_ma_type}({f_ma_len}) / {s_ma_type}({s_ma_len})", hovermode="x unified", template="plotly_dark", height=400)
            st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Ehlers SuperSmoother":
        st.markdown("### 🌊 Ehlers SuperSmoother Settings")
        st.markdown("Filters high frequency noise to create a zero-lag trendline.")
        
        ss_period = st.slider("SuperSmoother Period", 5, 252, 15)
        
        # Auto-run SuperSmoother
        ss_series = EhlersFilters.super_smoother(prices_bt, ss_period)
        
        # Signal logic: Long when Price > SuperSmoother, else Hedge (0)
        signals = (prices_bt > ss_series).astype(int)
        
        with st.expander("See Strategy Context", expanded=True):
            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
            fig_ctx.add_trace(go.Scatter(x=ss_series.index, y=ss_series, mode='lines', line=dict(color='magenta', width=2), name=f'SuperSmoother ({ss_period})'))
            
            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
            highlight_plotly_zones(fig_ctx, signals == 0, 'red', opacity=0.1)
            
            fig_ctx.update_layout(title="Ehlers SuperSmoother Signal", hovermode="x unified", template="plotly_dark", height=400)
            st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Ehlers Simple Decycler":
        st.markdown("### 🧲 Ehlers Simple Decycler Settings")
        st.markdown("Isolates the underlying low-frequency trend by removing market cycles.")
        
        dec_period = st.slider("Decycler High-Pass Period", 20, 252, 60)
        
        # Auto-run Decycler
        decycler_series = EhlersFilters.simple_decycler(prices_bt, dec_period)
        
        # Signal logic: Long when Price > Decycler, else Hedge (0)
        signals = (prices_bt > decycler_series).astype(int)
        
        with st.expander("See Strategy Context", expanded=True):
            fig_ctx = go.Figure()
            fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.5, name='Price'))
            fig_ctx.add_trace(go.Scatter(x=decycler_series.index, y=decycler_series, mode='lines', line=dict(color='orange', width=2), name=f'Decycler ({dec_period})'))
            
            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1)
            highlight_plotly_zones(fig_ctx, signals == 0, 'red', opacity=0.1)
            
            fig_ctx.update_layout(title="Ehlers Simple Decycler Signal", hovermode="x unified", template="plotly_dark", height=400)
            st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Institutional Mean Reversion (Z-Score)":
        st.markdown("### 📉 Institutional Mean Reversion (Z-Score) Settings")
        st.markdown("Statistical arbitrage approach: Buy when asset is significantly oversold relative to its rolling mean, and exit when it reverts.")
        
        col_mr1, col_mr2, col_mr3 = st.columns(3)
        with col_mr1:
            mr_lookback = st.slider("Lookback Window", 5, 252, 20)
        with col_mr2:
            mr_entry_z = st.number_input("Entry Z-Score (Long)", 0.1, 5.0, 2.0, step=0.1)
        with col_mr3:
            mr_exit_z = st.number_input("Exit Z-Score (Close Long)", -2.0, 2.0, 0.0, step=0.1)
            
        with st.spinner("Calculating Z-Scores..."):
            # Calculate rolling stats
            mr_ma = prices_bt.rolling(window=mr_lookback).mean()
            mr_std = prices_bt.rolling(window=mr_lookback).std()
            mr_z = (prices_bt - mr_ma) / (mr_std + 1e-9)
            
            # Generate Signals: Long when Z < -entry_z, Exit when Z > -exit_z
            long_cond = mr_z < -mr_entry_z
            exit_cond = mr_z > -mr_exit_z
            
            # Stateful logic
            def get_stateful_mr_signal(l_cond, e_cond, index):
                sig = pd.Series(np.nan, index=index)
                sig.loc[l_cond] = 1
                sig.loc[e_cond] = 0
                return sig.ffill().fillna(0)
            
            signals = get_stateful_mr_signal(long_cond, exit_cond, prices_bt.index)
            
            # Dynamic bands for context plot
            upper_band = mr_ma + (mr_std * mr_entry_z)
            lower_band = mr_ma - (mr_std * mr_entry_z)
            exit_band = mr_ma - (mr_std * mr_exit_z)
            
            # Plot Context
            with st.expander("See Strategy Context", expanded=True):
                fig_ctx = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # Top Chart: Price and Bands
                fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.8, name='Price'), row=1, col=1)
                fig_ctx.add_trace(go.Scatter(x=mr_ma.index, y=mr_ma, mode='lines', line=dict(color='blue', dash='dash'), name=f'Mean ({mr_lookback})'), row=1, col=1)
                fig_ctx.add_trace(go.Scatter(x=lower_band.index, y=lower_band, mode='lines', line=dict(color='green', width=1), opacity=0.5, name=f'Entry Band (-{mr_entry_z}σ)'), row=1, col=1)
                fig_ctx.add_trace(go.Scatter(x=exit_band.index, y=exit_band, mode='lines', line=dict(color='yellow', width=1), opacity=0.5, name=f'Exit Band (-{mr_exit_z}σ)'), row=1, col=1)
                
                # Bottom Chart: Z-Score
                fig_ctx.add_trace(go.Scatter(x=mr_z.index, y=mr_z, mode='lines', line=dict(color='orange'), name='Z-Score'), row=2, col=1)
                fig_ctx.add_hline(y=-mr_entry_z, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Entry")
                fig_ctx.add_hline(y=-mr_exit_z, line_dash="dash", line_color="yellow", row=2, col=1, annotation_text="Exit")
                fig_ctx.add_hline(y=0, line_color="white", opacity=0.3, row=2, col=1)
                
                highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1, row=1, col=1)
                highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1, row=2, col=1)
                
                fig_ctx.update_layout(title="Institutional Mean Reversion (Z-Score) Signal", hovermode="x unified", template="plotly_dark", height=600)
                st.plotly_chart(fig_ctx, use_container_width=True)

    elif strategy_type == "Relative Strength Ratio (vs Benchmark)":
        st.markdown("### ⚖️ Relative Strength Ratio (vs Benchmark) Settings")
        st.markdown("Momentum strategy: Long when the asset is gaining relative strength against a benchmark, Cash when losing.")
        
        col_rs1, col_rs2 = st.columns(2)
        with col_rs1:
            bench_ticker = st.text_input("Benchmark Ticker", "SPY")
        with col_rs2:
            rs_ma_len = st.slider("RS Smoothing MA Length", 5, 200, 50)
            
        with st.spinner(f"Fetching Benchmark Data ({bench_ticker})..."):
            try:
                if live_mode:
                    bench_df = load_data(bench_ticker, start_date, end_date, interval=data_interval)
                else:
                    bench_df = load_data(bench_ticker, bt_start_date, bt_end_date, interval='1d')
                    
                if bench_df is None or bench_df.empty:
                    st.error("Could not fetch benchmark data. Please check ticker.")
                    signals = None
                else:
                    bench_prices = bench_df['Close']
                    
                    # Align indices
                    common_idx = prices_bt.index.intersection(bench_prices.index)
                    if len(common_idx) < rs_ma_len:
                        st.error("Not enough overlapping data between the asset and benchmark to calculate moving averages.")
                        signals = None
                    else:
                        aligned_prices = prices_bt.loc[common_idx]
                        aligned_bench = bench_prices.loc[common_idx]
                        
                        # Calculate RS Ratio and MA
                        rs_ratio = aligned_prices / aligned_bench
                        rs_ma = rs_ratio.rolling(window=rs_ma_len).mean()
                        
                        # Signal: Long when RS Ratio > RS MA
                        signals_aligned = (rs_ratio > rs_ma).astype(int)
                        
                        # Re-index back to full bt_prices length, ffill signals
                        signals = pd.Series(np.nan, index=prices_bt.index)
                        signals.loc[common_idx] = signals_aligned
                        signals = signals.ffill().fillna(0)
                        
                        # Plot Context
                        with st.expander("See Strategy Context", expanded=True):
                            fig_ctx = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
                            
                            # Top Chart: Price
                            fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.8, name='Price'), row=1, col=1)
                            
                            # Bottom Chart: RS Ratio
                            fig_ctx.add_trace(go.Scatter(x=rs_ratio.index, y=rs_ratio, mode='lines', line=dict(color='magenta'), name=f'RS Ratio vs {bench_ticker}'), row=2, col=1)
                            fig_ctx.add_trace(go.Scatter(x=rs_ma.index, y=rs_ma, mode='lines', line=dict(color='blue', dash='dash'), name=f'RS MA ({rs_ma_len})'), row=2, col=1)
                            
                            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1, row=1, col=1)
                            highlight_plotly_zones(fig_ctx, signals == 1, 'green', opacity=0.1, row=2, col=1)
                            
                            fig_ctx.update_layout(title=f"Relative Strength Ratio ({TICKER} vs {bench_ticker})", hovermode="x unified", template="plotly_dark", height=500)
                            st.plotly_chart(fig_ctx, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading benchmark data: {str(e)}")
                signals = None

    elif strategy_type == "Implied Volatility Proxy (^VIX)":
        st.markdown("### 🎲 Robust Implied Volatility Proxy Lab (^VIX)")
        st.markdown("Instead of one fixed VIX rule, this tests several VIX + price confirmation rules and selects the strongest one for the selected history.")
        st.caption("Goal: avoid weak noisy VIX exits, reduce whipsaws, and only go risk-off when VIX stress is confirmed by price weakness.")
        
        col_vx1, col_vx2, col_vx3, col_vx4 = st.columns(4)
        with col_vx1:
            vix_ma_len = st.number_input("IV Proxy Baseline Length", min_value=5, max_value=200, value=20, step=1)
        with col_vx2:
            vix_z = st.number_input("Risk-Off Z-Score", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        with col_vx3:
            vix_cap_z = st.number_input("Capitulation Z-Score", min_value=1.0, max_value=6.0, value=3.5, step=0.1)
        with col_vx4:
            confirm_bars = st.number_input("Confirmation Bars", min_value=1, max_value=10, value=2, step=1)
            use_adaptive_iv_lab = st.checkbox("Use adaptive IV strategy chooser", value=True)

        wf_col1, wf_col2, wf_col3, wf_col4 = st.columns(4)
        with wf_col1:
            enable_iv_walk_forward = st.checkbox("Walk-forward validation", value=False)
        with wf_col2:
            wf_train_window = st.number_input("WF Train Bars", min_value=60, max_value=756, value=126, step=21)
        with wf_col3:
            wf_forward_window = st.number_input("WF Forward Bars", min_value=5, max_value=126, value=21, step=5)
        with wf_col4:
            use_wf_signal = st.checkbox("Use WF signal for backtest", value=False)
        wf_confirmed_bar = st.checkbox("WF confirmed-bar execution: use previous closed signal", value=False, help="More realistic. It prevents same-candle lookahead by trading from the prior completed signal.")

        with st.expander("Institutional WFO Controls", expanded=True):
            st.caption("These settings make WFO less overfit: top-3 ensemble, risk-adjusted scoring, strategy-switch penalty, and momentum override for strong runners.")
            iw1, iw2, iw3, iw4 = st.columns(4)
            with iw1:
                iv_wfo_mode = st.selectbox("WFO Selection Mode", ["Institutional Ensemble", "Single Best Rule"], index=0)
            with iw2:
                iv_top_n = st.number_input("Top-N Ensemble", min_value=1, max_value=5, value=3, step=1)
            with iw3:
                iv_switch_penalty = st.number_input("Strategy Switch Penalty", min_value=0.0, max_value=30.0, value=8.0, step=1.0)
            with iw4:
                iv_ensemble_threshold = st.number_input("Ensemble Long Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)
            iv_trend_override = st.checkbox("Momentum / trend override for strong runners", value=True, help="Keeps the strategy from going fully defensive when the stock is above major trend filters and showing momentum.")
            iv_min_trades = st.number_input("Minimum Trades Penalty Level", min_value=1, max_value=20, value=3, step=1)

        with st.expander("IV Sharpe / Drawdown Guard", expanded=True):
            st.caption("Optional final risk overlay for IV Proxy. It keeps the selected IV strategy, but throttles exposure when price trend/volatility risk worsens. Designed to improve Sharpe and reduce max drawdown.")
            gd1, gd2, gd3, gd4 = st.columns(4)
            with gd1:
                enable_iv_risk_guard = st.checkbox("Enable IV Sharpe/DD Guard", value=True, key="iv_enable_sharpe_dd_guard")
            with gd2:
                iv_guard_mode = st.selectbox("Guard Mode", ["Balanced", "Strict", "Loose"], index=0, key="iv_guard_mode")
            with gd3:
                iv_guard_dd = st.number_input("Price DD Exit Guard (%)", min_value=5.0, max_value=50.0, value=18.0, step=1.0, key="iv_guard_dd_pct")
            with gd4:
                iv_guard_vol = st.checkbox("Volatility throttle", value=True, key="iv_guard_vol_throttle")
            gd5, gd6, gd7 = st.columns(3)
            with gd5:
                iv_guard_equity_dd_on = st.checkbox("Equity DD Guard", value=True, key="iv_guard_equity_dd_on")
            with gd6:
                iv_guard_equity_dd = st.number_input("Max Account DD Guard (%)", min_value=5.0, max_value=60.0, value=20.0, step=1.0, key="iv_guard_equity_dd_pct")
            with gd7:
                iv_guard_equity_action = st.selectbox("Equity DD Action", ["Soft Throttle", "Hard Cash"], index=0, key="iv_guard_equity_action")
            st.caption("Soft Throttle = reduces exposure after account DD stress without fully missing strong runners. Hard Cash = exits fully to cash until recovery.")
            
        with st.spinner("Fetching ^VIX data and testing robust IV proxy rules..."):
            try:
                proxy_prices, proxy_label = load_iv_proxy_data_for_backtest(
                    prices_bt.index,
                    live_mode=live_mode,
                    data_interval=data_interval,
                    start_date=bt_start_date,
                    end_date=bt_end_date
                )

                if proxy_prices is None or len(proxy_prices) < 5:
                    st.error("Could not fetch ^VIX/VX proxy data. Strategy cannot proceed.")
                    signals = None
                else:
                    aligned_pair = align_proxy_to_asset(prices_bt, proxy_prices)
                    min_needed = max(int(vix_ma_len), 50)

                    if len(aligned_pair) < min_needed:
                        st.error("Not enough aligned asset/VIX proxy candles to calculate robust IV proxy rules. In live mode, try 15m or 60m interval.")
                        signals = None
                    else:
                        common_idx = aligned_pair.index
                        asset_prices = aligned_pair['asset'].astype(float)
                        aligned_vix = aligned_pair['proxy'].astype(float)
                        if live_mode:
                            st.caption(f"Live IV proxy source: {proxy_label}. Proxy values are forward-filled to match stock candle timestamps when exact timestamps differ.")
                        else:
                            st.caption(f"IV proxy source: {proxy_label}.")
                        
                        # Core VIX/IV proxy features
                        vix_ma = aligned_vix.rolling(window=int(vix_ma_len)).mean()
                        vix_std = aligned_vix.rolling(window=int(vix_ma_len)).std()
                        vix_zscore = (aligned_vix - vix_ma) / (vix_std + 1e-9)
                        vix_upper = vix_ma + (float(vix_z) * vix_std)
                        vix_cap = vix_ma + (float(vix_cap_z) * vix_std)
                        vix_ema_fast = aligned_vix.ewm(span=5, adjust=False).mean()
                        vix_ema_slow = aligned_vix.ewm(span=20, adjust=False).mean()
                        vix_slope_down = vix_ema_fast < vix_ema_slow
                        
                        # Asset confirmation features
                        asset_sma20 = asset_prices.rolling(window=20).mean()
                        asset_sma50 = asset_prices.rolling(window=50).mean()
                        asset_sma200 = asset_prices.rolling(window=200, min_periods=50).mean()
                        asset_ret_5 = asset_prices.pct_change(5)
                        asset_vol_20 = asset_prices.pct_change().rolling(20).std()
                        asset_vol_med = asset_vol_20.rolling(100, min_periods=20).median()
                        
                        def confirmed(cond, bars=None):
                            bars = int(confirm_bars if bars is None else bars)
                            cond = pd.Series(cond, index=common_idx).fillna(False)
                            if bars <= 1:
                                return cond
                            return cond.rolling(bars).sum().fillna(0) >= bars
                        
                        def stateful(entry, exit_, start_long=True):
                            pos = pd.Series(np.nan, index=common_idx, dtype=float)
                            pos.loc[pd.Series(entry, index=common_idx).fillna(False)] = 1.0
                            pos.loc[pd.Series(exit_, index=common_idx).fillna(False)] = 0.0
                            pos = pos.ffill()
                            return pos.fillna(1.0 if start_long else 0.0).clip(0, 1)
                        
                        # -------------------------------
                        # Strategy candidates
                        # -------------------------------
                        candidates = []
                        
                        # 1. Original but cleaner: VIX spike exits only when asset also weak; capitulation can reset long.
                        orig_pos = []
                        current_state = 1.0
                        for i in range(len(common_idx)):
                            v_val = aligned_vix.iloc[i]
                            v_ma_i = vix_ma.iloc[i]
                            v_up_i = vix_upper.iloc[i]
                            v_cp_i = vix_cap.iloc[i]
                            a_val = asset_prices.iloc[i]
                            a_sma = asset_sma50.iloc[i]
                            if pd.isna(v_up_i) or pd.isna(v_ma_i):
                                orig_pos.append(current_state)
                                continue
                            if v_val >= v_cp_i:
                                current_state = 1.0
                            elif v_val >= v_up_i and not pd.isna(a_sma) and a_val < a_sma:
                                current_state = 0.0
                            elif v_val < v_ma_i:
                                current_state = 1.0
                            orig_pos.append(current_state)
                        candidates.append((
                            "Original VIX Hysteresis + Trend Filter",
                            "Long by default. Exit when VIX spikes above the upper band and price is below the 50-SMA. Re-enter when VIX cools below baseline or capitulation appears.",
                            pd.Series(orig_pos, index=common_idx)
                        ))
                        
                        # 2. Crash shield: designed to avoid big drawdowns, not overtrade.
                        crash_exit = confirmed((vix_zscore > vix_z) & (asset_prices < asset_sma50) & (asset_ret_5 < 0))
                        crash_entry = (vix_zscore < 0.25) | ((asset_prices > asset_sma20) & vix_slope_down)
                        candidates.append((
                            "Crash Shield Confirmed",
                            "Exit only when VIX stress is high AND price trend is weak. Re-enter after VIX cools or price recovers above short trend.",
                            stateful(crash_entry, crash_exit, start_long=True)
                        ))
                        
                        # 3. Risk-on only: lower whipsaw by requiring calm VIX and price trend.
                        risk_on_entry = (vix_zscore < 0.75) & (asset_prices > asset_sma50)
                        risk_on_exit = confirmed((vix_zscore > vix_z) | (asset_prices < asset_sma50))
                        candidates.append((
                            "VIX Risk-On Trend Gate",
                            "Long only when VIX is not stressed and price is above the 50-SMA. Cash when either VIX stress or price weakness is confirmed.",
                            stateful(risk_on_entry, risk_on_exit, start_long=False)
                        ))
                        
                        # 4. Vol compression breakout: good when lower implied vol supports trend continuation.
                        compression_entry = (aligned_vix < vix_ma) & (vix_ema_fast < vix_ema_slow) & (asset_prices > asset_sma20) & (asset_prices > asset_sma50)
                        compression_exit = confirmed((aligned_vix > vix_upper) | (asset_prices < asset_sma20))
                        candidates.append((
                            "Vol Compression Breakout",
                            "Long when VIX is falling below baseline and price is stacked above 20/50-SMA. Exit on VIX spike or loss of 20-SMA.",
                            stateful(compression_entry, compression_exit, start_long=False)
                        ))
                        
                        # 5. Panic reset: handles extreme VIX spikes as potential washout, but demands price recovery.
                        panic_exit = confirmed((vix_zscore > vix_z) & (asset_prices < asset_sma50))
                        panic_entry = ((vix_zscore > vix_cap_z) & (asset_ret_5 > -0.03)) | ((vix_zscore < 0) & (asset_prices > asset_sma20))
                        candidates.append((
                            "Panic Reset + Recovery",
                            "Exit confirmed stress. Re-enter after extreme panic if price stabilizes, or after VIX falls below baseline and price reclaims 20-SMA.",
                            stateful(panic_entry, panic_exit, start_long=True)
                        ))
                        
                        # 6. Multi-factor IV score: avoids binary/noisy signal by requiring a score threshold.
                        score = pd.Series(0.0, index=common_idx)
                        score += (vix_zscore < 0.5).astype(float)
                        score += (vix_slope_down).astype(float)
                        score += (asset_prices > asset_sma20).astype(float)
                        score += (asset_prices > asset_sma50).astype(float)
                        score += (asset_vol_20 <= asset_vol_med).fillna(False).astype(float)
                        score_exit = confirmed((score <= 2) & (vix_zscore > 1.0))
                        score_entry = score >= 3
                        candidates.append((
                            "Robust IV Composite Score",
                            "Combines VIX calmness, falling VIX trend, price trend, and realized-vol filter. Long only when the total score is healthy.",
                            stateful(score_entry, score_exit, start_long=False)
                        ))
                        
                        # 7. Long-term trend override: do not fight strong uptrends unless VIX stress is serious.
                        trend_override_entry = (asset_prices > asset_sma200) & ((vix_zscore < vix_z) | vix_slope_down)
                        trend_override_exit = confirmed((asset_prices < asset_sma200) & (vix_zscore > 1.0)) | confirmed(vix_zscore > vix_cap_z + 0.5, bars=1)
                        candidates.append((
                            "Long-Term Trend Override",
                            "Stay long in larger uptrends unless VIX stress is confirmed and price loses the long-term trend. Built to avoid unnecessary exits.",
                            stateful(trend_override_entry, trend_override_exit, start_long=True)
                        ))

                        # 8. Momentum protected IV gate: institutional guardrail for big stock-specific runners.
                        # It uses SPY relative strength when available, but safely falls back to price momentum only.
                        try:
                            if live_mode:
                                spy_df = load_data("SPY", start_date, end_date, interval=data_interval)
                            else:
                                spy_df = load_data("SPY", bt_start_date, bt_end_date, interval='1d')
                            spy_close = spy_df['Close'].reindex(common_idx).ffill() if spy_df is not None and not spy_df.empty else pd.Series(np.nan, index=common_idx)
                            rs_ratio = (asset_prices / spy_close).replace([np.inf, -np.inf], np.nan)
                            rs_ma = rs_ratio.rolling(50, min_periods=20).mean()
                            rs_strong = (rs_ratio > rs_ma).fillna(False)
                        except Exception:
                            rs_strong = pd.Series(True, index=common_idx)

                        asset_ret_20 = asset_prices.pct_change(20)
                        strong_runner = (asset_prices > asset_sma50) & (asset_prices > asset_sma200) & (asset_ret_20 > 0) & rs_strong
                        severe_iv_stress = confirmed((vix_zscore > vix_cap_z) & (asset_prices < asset_sma50), bars=1)
                        protected_entry = strong_runner | ((vix_zscore < 1.0) & (asset_prices > asset_sma20))
                        protected_exit = severe_iv_stress | confirmed((asset_prices < asset_sma50) & (vix_zscore > vix_z), bars=int(confirm_bars))
                        candidates.append((
                            "Momentum Protected IV Gate",
                            "Allows strong relative-strength uptrends to stay long unless IV stress becomes severe. Designed to avoid missing huge runners.",
                            stateful(protected_entry, protected_exit, start_long=True)
                        ))

                        # Optional Sharpe/DD guard: apply to each IV candidate before ranking so the selected rule,
                        # manual rule, and trade log all reflect the same risk-controlled signal.
                        if bool(enable_iv_risk_guard):
                            guarded_candidates = []
                            for cname, clogic, csig in candidates:
                                guarded_sig = apply_iv_sharpe_dd_guard(
                                    asset_prices,
                                    csig,
                                    mode=str(iv_guard_mode),
                                    max_price_dd=float(iv_guard_dd) / 100.0,
                                    vol_throttle=bool(iv_guard_vol),
                                    equity_dd_guard=bool(iv_guard_equity_dd_on),
                                    max_equity_dd=float(iv_guard_equity_dd) / 100.0,
                                    equity_guard_action=str(iv_guard_equity_action)
                                )
                                guarded_candidates.append((
                                    cname,
                                    clogic + f" Risk guard applied: {iv_guard_mode} mode, {float(iv_guard_dd):.0f}% price-DD guard, {float(iv_guard_equity_dd):.0f}% account-DD guard {'ON' if bool(iv_guard_equity_dd_on) else 'OFF'}.",
                                    guarded_sig
                                ))
                            candidates = guarded_candidates
                        
                        # Rank all candidates on the same selected history.
                        ranking_rows = []
                        scored_candidates = []
                        for name, logic, sig in candidates:
                            score_res = evaluate_strategy_candidate(asset_prices, sig, initial_capital=initial_cap, trailing_stop_pct=trailing_stop, stop_loss_pct=stop_loss)
                            if score_res is None:
                                continue
                            scored_candidates.append((name, logic, sig, score_res))
                            ranking_rows.append({
                                "Rule": name,
                                "Strategy Return %": round(score_res['Strategy Return %'], 2),
                                "Buy & Hold Return %": round(score_res['Buy & Hold Return %'], 2),
                                "Difference %": round(score_res['Difference %'], 2),
                                "Max DD %": round(score_res['Max DD %'], 2),
                                "Trades": score_res['Trades']
                            })
                        
                        if not scored_candidates:
                            st.warning("No IV proxy candidates could be scored.")
                            signals = None
                        else:
                            rank_df = pd.DataFrame(ranking_rows).sort_values(['Difference %', 'Strategy Return %'], ascending=False).reset_index(drop=True)
                            best_name = rank_df.iloc[0]['Rule']
                            best_name, best_logic, best_sig, best_score = next(x for x in scored_candidates if x[0] == best_name)
                            
                            st.write("#### 🧠 Robust IV Proxy Strategy Ranking")
                            st.caption("In-sample ranking is now a reference table. You can also manually choose any IV rule below and view its own trade log.")
                            st.dataframe(rank_df, use_container_width=True)

                            manual_iv_strategy_override = st.checkbox(
                                "Manually select IV proxy strategy for backtest/trade log",
                                value=False,
                                help="Turn this on when you want to inspect a specific rule instead of the auto-selected best rule."
                            )
                            manual_iv_choice = None
                            if manual_iv_strategy_override:
                                available_iv_rules = rank_df['Rule'].tolist()
                                default_manual_idx = available_iv_rules.index(best_name) if best_name in available_iv_rules else 0
                                manual_iv_choice = st.selectbox(
                                    "Choose IV proxy strategy to review",
                                    available_iv_rules,
                                    index=default_manual_idx,
                                    help="Example: choose Vol Compression Breakout to see its own performance and trades."
                                )
                                best_name, best_logic, best_sig, best_score = next(x for x in scored_candidates if x[0] == manual_iv_choice)
                                st.info(f"Manual IV strategy mode: main metrics and trade log will use **{best_name}**. Auto/WFO results remain reference only.")
                            
                            wf_result = None
                            if enable_iv_walk_forward:
                                st.write("#### 🚶 Walk-Forward IV Proxy Validation")
                                st.caption("This is stricter than the normal ranking: it chooses the best rule using only the past training window, then tests that rule on the next unseen forward window.")
                                if iv_wfo_mode == "Institutional Ensemble":
                                    wf_result = walk_forward_strategy_selection_institutional(
                                        asset_prices,
                                        candidates,
                                        train_window=int(wf_train_window),
                                        forward_window=int(wf_forward_window),
                                        initial_capital=initial_cap,
                                        confirmed_bar=bool(wf_confirmed_bar),
                                        trailing_stop_pct=trailing_stop,
                                        stop_loss_pct=stop_loss,
                                        top_n=int(iv_top_n),
                                        ensemble_threshold=float(iv_ensemble_threshold),
                                        switch_penalty=float(iv_switch_penalty),
                                        min_trades=int(iv_min_trades),
                                        trend_override=bool(iv_trend_override)
                                    )
                                else:
                                    wf_result = walk_forward_strategy_selection(
                                        asset_prices,
                                        candidates,
                                        train_window=int(wf_train_window),
                                        forward_window=int(wf_forward_window),
                                        initial_capital=initial_cap,
                                        confirmed_bar=bool(wf_confirmed_bar),
                                        trailing_stop_pct=trailing_stop,
                                        stop_loss_pct=stop_loss
                                    )

                                if wf_result is None:
                                    st.warning("Not enough history for walk-forward validation with the selected train/forward windows. Try smaller WF windows or a longer backtest range.")
                                else:
                                    wf_overall = wf_result.get('overall') or {}
                                    w1, w2, w3, w4, w5 = st.columns(5)
                                    w1.metric("WF Strategy Return", f"{wf_overall.get('Strategy Return %', 0):.2f}%")
                                    w2.metric("WF Buy & Hold", f"{wf_overall.get('Buy & Hold Return %', 0):.2f}%")
                                    w3.metric("WF Difference", f"{wf_overall.get('Difference %', 0):+.2f}%")
                                    w4.metric("WF Win Rate", f"{wf_result['win_rate'] * 100:.0f}%")
                                    w5.metric("WF Stability", f"{wf_result['stability_score']:.0f}/100")

                                    st.dataframe(wf_result['rows'].sort_values('Forward End', ascending=False), use_container_width=True)

                                    if wf_result['stability_score'] >= 70 and wf_overall.get('Difference %', 0) > 0:
                                        st.success("Walk-forward result is strong: the chooser worked out-of-sample better than buy & hold over the tested periods.")
                                    elif wf_result['stability_score'] >= 45:
                                        st.warning("Walk-forward result is mixed. Useful as a filter, but confirm with CVD/VWAP and avoid oversized trades.")
                                    else:
                                        st.error("Walk-forward result is weak/unstable. Do not treat this IV Proxy winner as a strong standalone edge.")

                                    if use_wf_signal:
                                        st.info("Using the walk-forward-selected signal for the main backtest below. This is more realistic than using the full-history best rule.")

                            # Stability score across sub-windows. This reduces blind trust in one lucky full-period backtest.
                            windows = [63, 126, 252, len(asset_prices)]
                            windows = sorted(set([w for w in windows if len(asset_prices) >= max(30, w)]))
                            stable_checks = []
                            positive_windows = 0
                            top2_windows = 0
                            for w in windows:
                                sub_prices = asset_prices.tail(w)
                                sub_rows = []
                                for name, logic, sig in candidates:
                                    sub_sig = pd.Series(sig).reindex(sub_prices.index).ffill().fillna(0)
                                    sub_score = evaluate_strategy_candidate(sub_prices, sub_sig, initial_capital=initial_cap, trailing_stop_pct=trailing_stop, stop_loss_pct=stop_loss)
                                    if sub_score is None:
                                        continue
                                    sub_rows.append((name, sub_score['Difference %']))
                                if sub_rows:
                                    sub_rank = sorted(sub_rows, key=lambda x: x[1], reverse=True)
                                    best_diff_sub = dict(sub_rows).get(best_name, np.nan)
                                    rank_pos = [x[0] for x in sub_rank].index(best_name) + 1 if best_name in [x[0] for x in sub_rank] else np.nan
                                    if pd.notna(best_diff_sub) and best_diff_sub > 0:
                                        positive_windows += 1
                                    if pd.notna(rank_pos) and rank_pos <= 2:
                                        top2_windows += 1
                                    stable_checks.append({
                                        "Window": f"Last {w} bars" if w != len(asset_prices) else "Full selected period",
                                        "Best Rule Rank": rank_pos,
                                        "Difference %": round(best_diff_sub, 2) if pd.notna(best_diff_sub) else np.nan
                                    })
                            
                            stability_score = 0
                            if stable_checks:
                                stability_score = round(100 * ((positive_windows / len(stable_checks)) * 0.55 + (top2_windows / len(stable_checks)) * 0.45), 0)
                                st.write("#### 🧱 Stability Check")
                                s1, s2, s3 = st.columns(3)
                                s1.metric("Selected Best Rule", best_name)
                                s2.metric("Stability Score", f"{stability_score:.0f}/100")
                                s3.metric("Windows Tested", len(stable_checks))
                                st.dataframe(pd.DataFrame(stable_checks), use_container_width=True)
                                if stability_score >= 70:
                                    st.success("This IV proxy winner looks relatively stable across multiple windows.")
                                elif stability_score >= 45:
                                    st.warning("This IV proxy winner is decent, but not fully stable. Confirm with CVD/VWAP before trusting it.")
                                else:
                                    st.error("This IV proxy winner is unstable. Treat it as research only, not a strong trading edge.")

                            if bool(enable_iv_risk_guard):
                                st.success(f"IV Sharpe/DD Guard is ON: {iv_guard_mode} mode, {float(iv_guard_dd):.0f}% price-DD guard, account-DD guard {float(iv_guard_equity_dd):.0f}% {'ON' if bool(iv_guard_equity_dd_on) else 'OFF'}, volatility throttle {'ON' if bool(iv_guard_vol) else 'OFF'}.")

                            if manual_iv_strategy_override:
                                st.info(f"Chosen manual IV proxy rule: **{best_name}** — {best_logic}")
                            elif enable_iv_walk_forward and use_wf_signal and wf_result is not None:
                                st.info("Primary result below uses **walk-forward-selected IV rules**. The in-sample chosen rule is shown only as reference.")
                                st.caption(f"In-sample reference winner: {best_name} — {best_logic}")
                            else:
                                st.info(f"Chosen IV proxy rule: **{best_name}** — {best_logic}")
                            
                            # Re-index selected signal back to the shared backtest engine.
                            # IMPORTANT: when WFO is enabled, make the main performance metrics use ONLY
                            # the out-of-sample walk-forward segment, not the training/history segment.
                            signals = pd.Series(np.nan, index=prices_bt.index)
                            using_wfo_primary = bool((not manual_iv_strategy_override) and enable_iv_walk_forward and use_wf_signal and wf_result is not None)
                            using_wfo_primary_for_metrics = using_wfo_primary
                            if using_wfo_primary:
                                benchmark_label_for_metrics = "WFO Test Benchmark"
                                full_period_benchmark_pct_for_metrics = buy_hold_return_pct(asset_prices)
                                wf_sig = wf_result['signal'].reindex(common_idx).ffill().fillna(0).clip(0, 1)
                                if bool(enable_iv_risk_guard):
                                    wf_sig = apply_iv_sharpe_dd_guard(
                                        asset_prices,
                                        wf_sig,
                                        mode=str(iv_guard_mode),
                                        max_price_dd=float(iv_guard_dd) / 100.0,
                                        vol_throttle=bool(iv_guard_vol),
                                        equity_dd_guard=bool(iv_guard_equity_dd_on),
                                        max_equity_dd=float(iv_guard_equity_dd) / 100.0,
                                        equity_guard_action=str(iv_guard_equity_action)
                                    )
                                signals.loc[common_idx] = wf_sig
                                first_forward_start = wf_result.get('rows', pd.DataFrame()).iloc[0]['Forward Start'] if not wf_result.get('rows', pd.DataFrame()).empty else common_idx[0]
                                signals = signals.loc[signals.index >= first_forward_start]
                                strat_prices = strat_prices.loc[strat_prices.index >= first_forward_start]
                            else:
                                signals.loc[common_idx] = best_sig
                                signals = signals.ffill().fillna(1).clip(0, 1)
                            signals = signals.ffill().fillna(0 if using_wfo_primary else 1).clip(0, 1)
                            
                            with st.expander("See Robust IV Proxy Context", expanded=True):
                                fig_ctx = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.45, 0.35, 0.20], vertical_spacing=0.05)
                                
                                fig_ctx.add_trace(go.Scatter(x=asset_prices.index, y=asset_prices, mode='lines', line=dict(color='gray'), opacity=0.85, name='Asset Price'), row=1, col=1)
                                fig_ctx.add_trace(go.Scatter(x=asset_sma20.index, y=asset_sma20, mode='lines', line=dict(color='cyan', dash='dot'), opacity=0.55, name='20-SMA'), row=1, col=1)
                                fig_ctx.add_trace(go.Scatter(x=asset_sma50.index, y=asset_sma50, mode='lines', line=dict(color='yellow', dash='dot'), opacity=0.55, name='50-SMA'), row=1, col=1)
                                
                                fig_ctx.add_trace(go.Scatter(x=aligned_vix.index, y=aligned_vix, mode='lines', line=dict(color='purple'), name='IV Proxy'), row=2, col=1)
                                fig_ctx.add_trace(go.Scatter(x=vix_ma.index, y=vix_ma, mode='lines', line=dict(color='orange', dash='dot'), name=f'IV Proxy Baseline ({vix_ma_len})'), row=2, col=1)
                                fig_ctx.add_trace(go.Scatter(x=vix_upper.index, y=vix_upper, mode='lines', line=dict(color='red', dash='dash'), name=f'Risk-Off Band (+{vix_z}σ)'), row=2, col=1)
                                fig_ctx.add_trace(go.Scatter(x=vix_cap.index, y=vix_cap, mode='lines', line=dict(color='green', dash='dashdot'), name=f'Capitulation Band (+{vix_cap_z}σ)'), row=2, col=1)
                                
                                if enable_iv_walk_forward and use_wf_signal and wf_result is not None:
                                    chosen_aligned = wf_result['signal'].reindex(common_idx).ffill().fillna(0).clip(0, 1)
                                    exposure_label = 'WFO Selected Exposure (%)'
                                else:
                                    chosen_aligned = pd.Series(best_sig, index=common_idx).reindex(common_idx).ffill().fillna(0).clip(0, 1)
                                    exposure_label = 'Chosen Exposure (%)'
                                fig_ctx.add_trace(go.Scatter(x=chosen_aligned.index, y=chosen_aligned * 100, mode='lines', line=dict(color='#00f2ff'), fill='tozeroy', name=exposure_label), row=3, col=1)
                                
                                highlight_plotly_zones(fig_ctx, chosen_aligned == 1, 'green', opacity=0.10, row=1, col=1)
                                highlight_plotly_zones(fig_ctx, chosen_aligned == 0, 'red', opacity=0.08, row=1, col=1)
                                highlight_plotly_zones(fig_ctx, chosen_aligned == 1, 'green', opacity=0.08, row=2, col=1)
                                highlight_plotly_zones(fig_ctx, chosen_aligned == 0, 'red', opacity=0.08, row=2, col=1)
                                
                                fig_ctx.update_layout(title=f"Robust IV Proxy Context — Selected Rule: {best_name}", hovermode="x unified", template="plotly_dark", height=760)
                                fig_ctx.update_yaxes(title_text="Exposure", row=3, col=1, range=[0, 105])
                                st.plotly_chart(fig_ctx, use_container_width=True)
                            
            except Exception as e:
                st.error(f"Error loading or testing ^VIX data: {str(e)}")
                signals = None

    elif strategy_type == "Institutional Hurst Exponent":
        st.markdown("### 🎲 Institutional Hurst Exponent (Trend vs Mean-Reversion)")
        st.markdown("Trade the asset based on its mathematical persistence. \n* **Trending Regime (H > 0.55):** Buy via Momentum (EMA Cross).\n* **Mean-Reverting Regime (H < 0.45):** Buy via Mean Reversion (Bollinger Bands).\n* **Dead Zone:** Stay in CASH between 0.45 and 0.55. Uses 5-bar confirmation to kill whipsaws.")
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            hurst_window = st.number_input("Rolling Window", min_value=20, max_value=500, value=50, step=10)
            target_ann_vol = st.number_input("Target Annual Volatility (%)", min_value=1.0, max_value=100.0, value=25.0, step=1.0)
            use_vol_target = st.checkbox("Enable Vol Targeting", value=False)
        with col_h2:
            st.write("Regime Parameters")
            st.caption("Dead Zone: 0.45 to 0.55\nConfirmation: 3 Consecutive Bars\nSizing: Continuous Vol-Targeting")
            
        with st.spinner("Calculating Institutional Hurst & Volatility Targeting..."):
            try:
                hurst_series = rolling_hurst(prices_bt, window=int(hurst_window))
                
                # 1. Trend Signal (Slow EMA trend filter)
                # Cleaner than EMA-fast/EMA-slow cross for this Hurst allocator:
                # long only when price is above the slow trend filter.
                ema_fast = prices_bt.ewm(span=20, adjust=False).mean()
                ema_slow = prices_bt.ewm(span=50, adjust=False).mean()
                trend_signal = (prices_bt > ema_slow).astype(float)
                
                # 2. Mean Reversion Signal (Bollinger Bands)
                bb_ma = prices_bt.rolling(window=20).mean()
                bb_std = prices_bt.rolling(window=20).std()
                bb_lower = bb_ma - (2 * bb_std)
                mr_sig = pd.Series(np.nan, index=prices_bt.index)
                mr_sig.loc[prices_bt < bb_lower] = 1.0
                mr_sig.loc[prices_bt > bb_ma] = 0.0
                mr_signal = mr_sig.ffill().fillna(0.0)
                
                # 3. Regime Allocator (3-Bar Confirmation + Dead Zone)
                cond_trend = (hurst_series > 0.55)
                cond_mr = (hurst_series < 0.45)
                cond_cash = (hurst_series >= 0.45) & (hurst_series <= 0.55)
                
                conf_trend = cond_trend.astype(int).rolling(3, min_periods=3).sum().eq(3).fillna(False)
                conf_mr = cond_mr.astype(int).rolling(3, min_periods=3).sum().eq(3).fillna(False)
                conf_cash = cond_cash.astype(int).rolling(3, min_periods=3).sum().eq(3).fillna(False)
                
                state = pd.Series(np.nan, index=prices_bt.index)
                state.loc[conf_trend] = 1 # 1 = TREND
                state.loc[conf_mr] = -1   # -1 = MR
                state.loc[conf_cash] = 0  # 0 = CASH
                state = state.ffill().fillna(0)
                
                raw_signals = pd.Series(0.0, index=prices_bt.index)
                raw_signals.loc[state == 1] = trend_signal.loc[state == 1]
                raw_signals.loc[state == -1] = mr_signal.loc[state == -1]
                
                # 4. Optional Volatility Targeting Capital Sizing
                # OFF = clean 0/1 strategy signal.
                # ON  = scale exposure by GARCH volatility, capped between 0% and 100%.
                if use_vol_target:
                    returns_bt = prices_bt.pct_change().dropna()
                    if ARCH_AVAILABLE and len(returns_bt) >= 30:
                        try:
                            am = arch_model(returns_bt * 100, vol='Garch', p=1, q=1, dist='normal')
                            res = am.fit(disp='off')
                            garch_vol_daily = res.conditional_volatility / 100

                            # Align garch_vol_daily with prices_bt
                            garch_vol_aligned = pd.Series(np.nan, index=prices_bt.index)
                            garch_vol_aligned.loc[returns_bt.index] = garch_vol_daily
                            garch_vol_aligned = garch_vol_aligned.replace([np.inf, -np.inf], np.nan).bfill().ffill()

                            daily_target_vol = (target_ann_vol / 100) / np.sqrt(252)
                            position_size = (daily_target_vol / (garch_vol_aligned + 1e-9)).clip(0.0, 1.0)
                            signals = (raw_signals * position_size).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        except Exception as vol_e:
                            st.warning(f"Vol targeting failed, using raw Hurst signals instead: {vol_e}")
                            signals = raw_signals.fillna(0.0)
                    else:
                        st.warning("Vol targeting needs the 'arch' package and enough return history. Using raw Hurst signals instead.")
                        signals = raw_signals.fillna(0.0)
                else:
                    signals = raw_signals.fillna(0.0)
                
                with st.expander("See Strategy Context", expanded=True):
                    fig_ctx = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05)
                    
                    fig_ctx.add_trace(go.Scatter(x=prices_bt.index, y=prices_bt, mode='lines', line=dict(color='gray'), opacity=0.8, name='Price'), row=1, col=1)
                    
                    fig_ctx.add_trace(go.Scatter(x=hurst_series.index, y=hurst_series, mode='lines', line=dict(color='cyan'), name='Hurst (H)'), row=2, col=1)
                    fig_ctx.add_hline(y=0.55, line_dash="dash", line_color="green", row=2, col=1, annotation_text="Trend (>0.55)")
                    fig_ctx.add_hline(y=0.45, line_dash="dash", line_color="red", row=2, col=1, annotation_text="Mean Reversion (<0.45)")
                    
                    # Plot Capital Allocation
                    fig_ctx.add_trace(go.Scatter(x=signals.index, y=signals * 100, mode='lines', line=dict(color='green'), fill='tozeroy', name='Capital Exposure (%)'), row=3, col=1)
                    
                    # Highlight regimes
                    highlight_plotly_zones(fig_ctx, state == 1, 'green', opacity=0.1, row=1, col=1)
                    highlight_plotly_zones(fig_ctx, state == -1, 'orange', opacity=0.1, row=1, col=1)
                    highlight_plotly_zones(fig_ctx, state == 1, 'green', opacity=0.1, row=2, col=1)
                    highlight_plotly_zones(fig_ctx, state == -1, 'orange', opacity=0.1, row=2, col=1)
                    
                    fig_ctx.update_layout(title=f"Institutional Hurst Regime (Dead Zone + {'Vol Targeting' if use_vol_target else 'Raw 0/1 Exposure'})", hovermode="x unified", template="plotly_dark", height=700)
                    fig_ctx.update_yaxes(title_text="Capital (%)", row=3, col=1, range=[0, 105])
                    st.plotly_chart(fig_ctx, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error calculating Hurst Exponent: {str(e)}")
                signals = None

    # Run Backtest Engine if signals exist
    if signals is not None:
        # Keep original weekly model price/signal series for execution so returns,
        # PnL, stops, equity curve, and metrics remain EXACTLY the same.
        # Only the trade-log dates are mapped to actual raw trading dates for display.
        bt_results = BacktestEngine.run_strategy(strat_prices, signals, initial_cap, trailing_stop, stop_loss)

        # --- STRATEGY SIGNAL BANNER ---
        last_sig = signals.iloc[-1]
        last_dt = signals.index[-1]
        st.divider()
        if last_sig > 0:
            st.success(f"🚀 **STRATEGY SIGNAL (LONG)** | Last Update: {last_dt} | Exposure: **{last_sig*100:.0f}%** | Action: **HOLD LONG**")
        else:
            st.error(f"🛑 **STRATEGY SIGNAL (CASH)** | Last Update: {last_dt} | Action: **STAY IN CASH / HEDGE**")

        # Live alert: only fires on actual BUY/SELL flips, not on every refresh/hold.
        maybe_send_live_signal_alert(
            enabled=alert_enabled,
            live_mode=live_mode,
            ticker=TICKER,
            strategy_name=strategy_type,
            signals=signals,
            prices=strat_prices,
            alert_config=alert_config,
            extra_note=f"Backtest tab strategy: {strategy_type}"
        )
        
        # Metrics
        strat_metrics = BacktestEngine.calculate_metrics(bt_results['returns'], rf_rate)
        bench_metrics = BacktestEngine.calculate_metrics(strat_prices.pct_change().dropna(), rf_rate)
        
        # Display Metrics
        st.write("#### 📊 Performance Metrics")
        current_benchmark_pct = (bt_results['benchmark_curve'].iloc[-1]/initial_cap - 1)*100
        if using_wfo_primary_for_metrics and pd.notna(full_period_benchmark_pct_for_metrics):
            met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)
            with met_col1:
                st.metric("Total Return (Strategy)", f"{(bt_results['equity_curve'].iloc[-1]/initial_cap - 1)*100:.2f}%")
            with met_col2:
                st.metric("Sharpe Ratio", f"{strat_metrics.get('Sharpe Ratio', 0):.2f}")
            with met_col3:
                st.metric("Max Drawdown", f"{strat_metrics.get('Max Drawdown', 0)*100:.2f}%")
            with met_col4:
                help_txt = "Buy & hold over the same period used by the displayed strategy metrics."
                st.metric(benchmark_label_for_metrics, f"{current_benchmark_pct:.2f}%", help=help_txt)
            with met_col5:
                strategy_pct_now = (bt_results['equity_curve'].iloc[-1]/initial_cap - 1)*100
                if benchmark_label_for_metrics == "Full Benchmark":
                    st.metric("Gap vs Full", f"{strategy_pct_now - current_benchmark_pct:+.2f}%", help="Strategy return minus full-period buy & hold return.")
                else:
                    st.metric("Full Benchmark", f"{full_period_benchmark_pct_for_metrics:.2f}%", help="Buy & hold over the full selected chart period, including the WFO training window. Reference only.")
        else:
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            with met_col1:
                st.metric("Total Return (Strategy)", f"{(bt_results['equity_curve'].iloc[-1]/initial_cap - 1)*100:.2f}%")
            with met_col2:
                st.metric("Sharpe Ratio", f"{strat_metrics.get('Sharpe Ratio', 0):.2f}")
            with met_col3:
                st.metric("Max Drawdown", f"{strat_metrics.get('Max Drawdown', 0)*100:.2f}%")
            with met_col4:
                st.metric(benchmark_label_for_metrics, f"{current_benchmark_pct:.2f}%")
            
        # Equity Curve Plot
        st.write("#### 📈 Equity Curve")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_results['equity_curve'].index, y=bt_results['equity_curve'], mode='lines', line=dict(color='#00f2ff', width=2), name=f'Strategy ({strategy_type})'))
        fig_bt.add_trace(go.Scatter(x=bt_results['benchmark_curve'].index, y=bt_results['benchmark_curve'], mode='lines', line=dict(color='gray', dash='dash'), opacity=0.7, name=benchmark_label_for_metrics))
        fig_bt.update_layout(title=f"Strategy Performance: {TICKER}", hovermode="x unified", template="plotly_dark", height=500)
        st.plotly_chart(fig_bt, use_container_width=True)
        st.session_state.report_gen.add_plot("Backtest Performance", fig_bt)
        st.session_state.report_gen.add_data("Backtest Metrics", strat_metrics)
        
        # Trade Log
        st.write("#### 📝 Trade Log")
        trades_df = bt_results['trades'].copy()
        if not trades_df.empty:
            # DISPLAY ONLY: weekly Regime Switching dates are mapped to the
            # actual raw trading date in that week. Returns/metrics are unchanged.
            try:
                if strategy_type == "Regime Switching (Trend Following)" and bt_freq == "Weekly":
                    trades_df = map_weekly_trade_log_dates_only(trades_df, prices_bt)
                    trades_df = apply_weekly_live_trigger_display_overrides(
                        trades_df, prices_bt, signals, ticker=TICKER, strategy_name=strategy_type
                    )
            except Exception:
                pass

            # Show newest trades first by default
            if 'Entry Date' in trades_df.columns:
                trades_df = trades_df.sort_values('Entry Date', ascending=False).reset_index(drop=True)
            # Format dates
            trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.date
            trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).apply(lambda x: x.date() if pd.notnull(x) else "Open")
            
            st.dataframe(trades_df.style.format({
                "Buy Price": "{:.2f}",
                "Sell Price": "{:.2f}",
                "PnL (%)": "{:.2f}%",
                "Cumulative Return (%)": "{:.2f}"
            }), use_container_width=True)
        else:
            st.info("No closed trades generated by the strategy.")


with tab8:
    if df_main is None:
        st.warning("Please load a ticker to view Volatility Clustering.")
    else:
        st.write("### 🌩️ Volatility Clustering & Jump Analysis")
    
    # --- CALCULATION FOR VERDICT ---
    returns_arr = df_main['Returns'].values
    rv = RealizedVolatility.realized_variance(returns_arr)
    hawkes = HawkesVolatility().fit(returns_arr)
    br = hawkes.branching_ratio()

    # --- MODEL VERDICT BANNER ---
    latest_rv = np.sqrt(rv)*np.sqrt(252)
    if jump_detected: st.error(f"🎯 **MODEL VERDICT**: Significant **JUMPS** detected. Continuous volatility ({latest_rv:.1%}) is secondary to structural shocks. Use Merton/Heston models.")
    elif br > 0.8: st.warning(f"🎯 **MODEL VERDICT**: High **Volatility Clustering** (Branching Ratio: {br:.2f}). Recent shocks are likely to trigger further volatility.")
    else: st.success(f"🎯 **MODEL VERDICT**: Volatility is **Stable**. No significant clustering or jumps detected.")

    st.caption("Institutional analysis of volatility properties using High-Frequency logic applied to Daily data.")
    
    # 1. Realized Measures
    bv = RealizedVolatility.bipower_variation(returns_arr)
    jump_res = RealizedVolatility.jump_component(returns_arr)
    
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.metric("Total Volatility (RV)", f"{np.sqrt(rv)*np.sqrt(252):.2%}")
    with col_v2:
        st.metric("Continuous Vol (BV)", f"{np.sqrt(bv)*np.sqrt(252):.2%}", help="Jump-Robust Volatility")
    with col_v3:
        st.metric("Jump Ratio", f"{jump_res['jump_ratio']:.1%}", 
                  help="Proportion of variance due to jumps")
        if jump_res['p_value'] < 0.05:
            st.error("Significant Jumps Detected")
        else:
            st.success("No Significant Jumps")
    
    # 2. Hawkes Process
    st.divider()
    st.write("#### Self-Exciting Volatility (Hawkes Process)")
    
    # hawkes and br already calculated above for the banner
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.metric("Branching Ratio", f"{br:.2f}", help="> 1 means explosive/unstable volatility")
    with h_col2:
        hl = hawkes.half_life()
        st.metric("Vol Cluster Half-Life", f"{hl:.1f} days")
    with h_col3:
        st.metric("Baseline Intensity", f"{hawkes.mu:.4f}")

    if br > 0.9:
        st.warning("⚠️ Critical Instability: Volatility is self-reinforcing rapidly.")
    elif br > 0.5:
        st.info("Moderate Clustering: Recent shocks affect near-term future.")
    else:
        st.success("Stable: Volatility mean-reverts quickly.")

    st.session_state.report_gen.add_data("Volatility Clustering Metrics", {
        "RV": rv, "BV": bv, "Jump Ratio": jump_res['jump_ratio'],
        "Branching Ratio": br, "Half-Life": hl
    })

    # Visualization
    st.subheader("Volatility Clustering Visuals")
    
    # 1. Squared Returns (Volatility Proxy)
    fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"{TICKER} Returns Series", "Volatility Clustering (Squared Returns)"))
    fig_vol.add_trace(go.Scatter(x=df_main.index, y=df_main['Returns'], mode='lines', line=dict(color='gray', width=1), opacity=0.6, name="Returns"), row=1, col=1)
    squared_rets = df_main['Returns']**2
    fig_vol.add_trace(go.Scatter(x=df_main.index, y=squared_rets, mode='lines', line=dict(color='orange', width=1), opacity=0.8, name="Squared Returns"), row=2, col=1)
    threshold = squared_rets.mean() + 2 * squared_rets.std()
    fig_vol.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=1, annotation_text="2-Sigma Threshold")
    fig_vol.update_layout(height=600, hovermode="x unified", template="plotly_dark")
    st.plotly_chart(fig_vol, use_container_width=True)
    st.session_state.report_gen.add_plot("Volatility Clustering Visuals", fig_vol)

# ==========================================
# TAB 9: INSTITUTIONAL REGIME DETECTION
# ==========================================
with tab9:
    if df_main is None:
        st.warning("Please load a ticker to view Advanced Regime diagnostics.")
    else:
        st.write("### 🧠 Pro Regime Detection (Multi-Factor)")
    
    if not SKLEARN_AVAILABLE:
        st.error("⚠️ `scikit-learn` library is missing. Institutional upgrade requires it.")
    elif pro_detector is None:
        st.info("🏛️ **Active Engine**: Markov Switching Model (High Accuracy)")
        st.write(f"The current analysis is using the **Markov Regression** engine. This model identifies the current state based on transition probabilities and filtered marginals.")
        st.success(f"Current State: **{regime_label}** ({regime_prob:.1%} confidence)")
        st.caption(f"Number of States: {regime_data.get('n_states', 'Unknown')}")
        st.caption("Note: GMM-specific feature plots are disabled for Markov models.")
    else:
        st.caption("Institutional model using Returns, Volatility, and Trend Deviation via Multivariate GMM.")
        # Use the global pro_detector fitted in the decision engine
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Regime Label", regime_label, f"{regime_prob:.1%} Confidence")
        with m_col2:
            st.metric("Model AIC", f"{pro_detector.metrics.get('aic', 0):.0f}")
        
        st.write("#### 🌊 Regime Probability Stream")
        fig_pro = go.Figure()
        probs = pro_detector.regimes['probs']
        labels = [pro_detector.state_labels.get(i, f"State {i}") for i in range(probs.shape[1])]
        for i in range(probs.shape[1]):
            fig_pro.add_trace(go.Scatter(x=df_main.index, y=probs[:, i], mode='lines', line=dict(width=0), fill='tonexty' if i > 0 else 'tozeroy', stackgroup='one', name=labels[i]))
        fig_pro.update_layout(title="Multi-Factor Regime Probabilities", hovermode="x unified", template="plotly_dark", height=400)
        st.plotly_chart(fig_pro, use_container_width=True)
        st.session_state.report_gen.add_plot("Institutional Regime Probabilities", fig_pro)
        
        # Feature breakdown
        st.write("#### 📊 Institutional Feature Space")
        feat_df = pd.DataFrame(pro_detector.features, columns=['Momentum', 'Vol_Z', 'Trend_Dev'], index=df_main.index)
        fig_feat = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Feature 1: Momentum (Returns)', 'Feature 2: Volatility (Z-Score)', 'Feature 3: Structural Dev (Kalman)'))
        fig_feat.add_trace(go.Scatter(x=feat_df.index, y=feat_df['Momentum'], mode='lines', line=dict(color='blue'), opacity=0.7, name='Momentum'), row=1, col=1)
        fig_feat.add_trace(go.Scatter(x=feat_df.index, y=feat_df['Vol_Z'], mode='lines', line=dict(color='orange'), opacity=0.7, name='Vol_Z'), row=2, col=1)
        fig_feat.add_trace(go.Scatter(x=feat_df.index, y=feat_df['Trend_Dev'], mode='lines', line=dict(color='green'), opacity=0.7, name='Trend_Dev'), row=3, col=1)
        fig_feat.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=1, col=1)
        fig_feat.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=2, col=1)
        fig_feat.add_hline(y=0, line_dash="solid", line_color="white", line_width=1, row=3, col=1)
        fig_feat.update_layout(height=800, hovermode="x unified", template="plotly_dark", title="Institutional Feature Space")
        st.plotly_chart(fig_feat, use_container_width=True)
        st.session_state.report_gen.add_plot("Regime Feature Space", fig_feat)


# ==========================================
# TAB 10: SML & ALPHA (CAPM)
# ==========================================
with tab10:
    if df_main is None:
        st.warning("Please load a ticker to view SML/Alpha analysis.")
    else:
        st.write("### 📐 Securities Market Line (SML) & Alpha Analysis")
    st.caption("Institutional Factor Analysis: Rolling Beta, Jensen's Alpha, and Mispricing Spreads (Robust OLS).")

    # Configuration
    col_sml1, col_sml2 = st.columns(2)
    with col_sml1:
        bench_ticker = st.selectbox("Benchmark Index", ["SPY", "QQQ", "IWM", "VT", "^NSEI"] if market_region != "Indian Market (INR)" else ["^NSEI", "^NSEBANK", "SPY"])
    with col_sml2:
        roll_win = st.slider("Rolling Window (Days)", 30, 252, 90)

    if st.button("Run Alpha Analysis"):
        with st.spinner(f"Calibrating CAPM against {bench_ticker} (HAC Robust Errors)..."):
            # Load Benchmark Data
            df_bench = load_data(bench_ticker, start_date, end_date)
            
            if df_bench is not None:
                # Initialize Analyzer
                analyzer = SMLAnalyzer(df_main['Returns'], df_bench['Returns'], rf_annual=rf_rate)
                res_sml = analyzer.calculate_metrics(window=roll_win)
                
                # 1. METRICS DASHBOARD
                # --------------------
                last_row = res_sml.iloc[-1]
                
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    st.metric("Current Beta", f"{last_row['Beta']:.2f}", help="Sensitivity to Market")
                with m_c2:
                    st.metric("Jensen's Alpha", f"{last_row['Alpha_Daily']*252:.2%}", help="Annualized Excess Return vs Risk-Adjusted Exp")
                with m_c3:
                    st.metric("SML Exp Return", f"{last_row['SML_Exp_Return']:.2%}", help="Fair return for this level of risk")
                with m_c4:
                    mispricing = last_row['Mispricing_Spread']
                    st.metric("Mispricing Spread", f"{mispricing*100:.2f}%", 
                              delta="Undervalued" if mispricing > 0 else "Overvalued",
                              delta_color="normal")

                # 2. VISUALIZATION
                # --------------------
                st.divider()
                
                # A. SML SCATTER PLOT
                st.write("#### 1. Security Market Line (SML)")
                
                # Calculate aggregate risk/return for scatter
                # We'll plot Rolling periods as points
                
                fig_sml = go.Figure()
                avg_mkt_excess = res_sml['mkt_ex'].mean() * 252
                betas_line = np.linspace(0, max(res_sml['Beta'].max(), 2.0), 100)
                sml_y = rf_rate + betas_line * avg_mkt_excess
                fig_sml.add_trace(go.Scatter(x=betas_line, y=sml_y, mode='lines', line=dict(color='white', dash='dash', width=2), name='Security Market Line (SML)'))
                
                curr_beta = last_row['Beta']
                curr_ret = last_row['Actual_Return_Ann']
                fig_sml.add_trace(go.Scatter(x=[curr_beta], y=[curr_ret], mode='markers', marker=dict(color='blue', size=12), name=f'{TICKER} (Current)'))
                
                fig_sml.add_trace(go.Scatter(x=res_sml['Beta'], y=res_sml['Actual_Return_Ann'], mode='markers', marker=dict(color='lightblue', size=5, opacity=0.3), name='Historical Path'))
                
                mkt_ret_tot = (res_sml['mkt_ex'].mean() * 252) + rf_rate
                fig_sml.add_trace(go.Scatter(x=[1.0], y=[mkt_ret_tot], mode='markers', marker=dict(color='red', symbol='diamond', size=10), name='Market'))
                
                fig_sml.update_layout(title="Risk-Reward Profile vs Equilibrium", xaxis_title="Systematic Risk (Beta)", yaxis_title="Annualized Expected Return", template="plotly_dark", height=500)
                st.plotly_chart(fig_sml, use_container_width=True)
                
                # B. ROLLING ALPHA & BETA
                st.write("#### 2. Rolling Factor Dynamics")
                
                fig_dyn = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f"Systematic Risk (Beta) - {roll_win} Day Window", "Manager Skill / Mispricing (Alpha)"))
                fig_dyn.add_trace(go.Scatter(x=res_sml.index, y=res_sml['Beta'], mode='lines', line=dict(color='purple'), name='Rolling Beta'), row=1, col=1)
                fig_dyn.add_hline(y=1.0, line_dash="dash", line_color="gray", row=1, col=1)
                
                alpha_ann = res_sml['Alpha_Daily'] * 252
                fig_dyn.add_trace(go.Scatter(x=res_sml.index, y=alpha_ann, mode='lines', line=dict(color='green'), name='Annualized Alpha'), row=2, col=1)
                fig_dyn.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                alpha_pos = alpha_ann.copy(); alpha_pos[alpha_pos < 0] = 0
                fig_dyn.add_trace(go.Scatter(x=res_sml.index, y=alpha_pos, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='green', opacity=0.1, showlegend=False), row=2, col=1)
                
                alpha_neg = alpha_ann.copy(); alpha_neg[alpha_neg > 0] = 0
                fig_dyn.add_trace(go.Scatter(x=res_sml.index, y=alpha_neg, mode='lines', line=dict(width=0), fill='tozeroy', fillcolor='red', opacity=0.1, showlegend=False), row=2, col=1)
                
                fig_dyn.update_layout(height=600, hovermode="x unified", template="plotly_dark")
                st.plotly_chart(fig_dyn, use_container_width=True)
                st.session_state.report_gen.add_plot("SML Factor Dynamics", fig_dyn)
                st.session_state.report_gen.add_data("SML Analysis Results", res_sml.tail(100))
                st.session_state.report_gen.add_data("Current SML Metrics", last_row.to_dict())

            else:
                st.error(f"Could not load data for Benchmark: {bench_ticker}")


# ==========================================
# TAB 11: INSTITUTIONAL TOTAL MARKET SCANNER
# ==========================================
with tab11:
    st.write("### 📡 Institutional Total Market Scanner")
    st.markdown("""
    **Massive Scale Quant Discovery**. Scan the entire US listing universe (~9,000+ assets). 
    Categorize every stock and ETF into **Long (Open)** vs **Cash (Closed)** lists using the Master Quant Score.
    """)
    
    scan_col1, scan_col2, scan_col3 = st.columns(3)
    
    with scan_col1:
        universe_type = st.selectbox("Market Universe", 
                                   ["Total US Stocks (6,000+)", "Total US ETFs (3,000+)", 
                                    "S&P 500", "NASDAQ 100", "Custom Watchlist"])
        
    with scan_col2:
        mcap_filter = st.select_slider("Size Filter (Market Cap)", 
                                    options=["All", "$500M (Micro+)", "$2B (Small+)", "$10B (Mid+)", "$50B (Large+)", "$200B (Mega+)"],
                                    value="All")
        mcap_map = {
            "All": 0, 
            "$500M (Micro+)": 5e8,
            "$2B (Small+)": 2e9,
            "$10B (Mid+)": 1e10, 
            "$50B (Large+)": 5e10, 
            "$200B (Mega+)": 2e11
        }
        
    with scan_col3:
        scan_depth = st.number_input("Scan Limit (Depth)", min_value=1, max_value=10000, value=50, 
                                    help="Limits number of tickers to scan for speed. 50-100 is recommended for 'Auto' mode.")
        if scan_depth > 500:
            st.warning("⚠️ High Depth: Scanning thousands of assets with deep quant models can take 30+ minutes.")
    
    scan_col1b, scan_col2b, scan_col3b = st.columns(3)
    with scan_col1b:
        scan_regime_mode = st.selectbox("Scanner Regime Mode", 
                                       ["Fixed: 4 States", "Fixed: 2 States", "Fixed: 3 States", "Auto: Best Fit"],
                                       index=0)
        scan_reg_map = {"Fixed: 4 States": 4, "Fixed: 2 States": 2, "Fixed: 3 States": 3, "Auto: Best Fit": "Auto"}
        scan_reg_param = scan_reg_map[scan_regime_mode]
    with scan_col2b:
        scan_opt_goal = st.selectbox("Optimization Goal", ["Robustness (BIC)", "Performance (PnL)"], index=0)
    with scan_col3b:
        scan_freq = st.selectbox("Scanner Frequency", ["Daily", "Weekly"], index=0)
        if scan_regime_mode == "Auto: Best Fit":
            st.info("💡 Auto-Performance mode ensures results match best historical backtest.")
        
    with st.expander("🛠️ Advanced Model Sync (Backtest Alignment)", expanded=False):
        async_col1, async_col2, async_col3 = st.columns(3)
        with async_col1:
            scan_engine = st.selectbox("Model Engine", ["Markov (High Accuracy)", "GMM (Fast)"], index=1,
                                     help="Markov engine matches Backtest Tab exactly but is slower for large lists.")
            scan_engine_param = "Markov" if "Markov" in scan_engine else "GMM"
            scan_initial_cap = st.number_input("Backtest Capital ($)", 1000, 1000000, 10000)
        with async_col2:
            scan_stability = st.slider("Signal Stability (Smoothing)", 0, 10, 4, 
                                      help="Matches 'Signal Stability' in Backtest Tab. Smoothes data before fitting.")
            
            use_scan_trailing = st.toggle("Enable Trailing Stop Loss", value=False, key="scan_ts_toggle")
            scan_trailing_stop = st.slider("Trailing Stop (%)", 0.0, 20.0, 5.0, step=0.5) / 100 if use_scan_trailing else 0.0
            
            use_scan_stop = st.toggle("Enable Hard Stop Loss", value=True, key="scan_sl_toggle")
            scan_stop_loss = st.slider("Hard Stop Loss (%)", 0.0, 30.0, 8.0, step=0.5) / 100 if use_scan_stop else 0.0
        with async_col3:
            scan_switch_vol = st.toggle("Switching Volatility", value=True)
            scan_switch_trend = st.toggle("Switching Mean", value=True)

    if live_mode and scan_freq == "Weekly":
        st.warning("⚠️ **Invalid Combo**: Weekly frequency on Live Intraday data typically has too few bars (< 15) for the models. Scanner may skip all assets. Switch Frequency to 'Daily' or disable 'Live Mode'.")

    # Custom list area only shows if needed
    custom_input = ""
    if universe_type == "Custom Watchlist":
        custom_input = st.text_area("Ticker List (Comma separated)", "AAPL, TSLA, BTC-USD, GC=F")

    st.divider()
    
    if st.button("🚀 EXECUTE TOTAL MARKET SCAN", use_container_width=True, type="primary"):
        # Determine Tickers
        if universe_type == "Total US Stocks (6,000+)":
            full_list = get_total_us_stocks()
        elif universe_type == "Total US ETFs (3,000+)":
            full_list = get_total_us_etfs()
        elif universe_type == "S&P 500":
            full_list = get_sp500_tickers()
        elif universe_type == "NASDAQ 100":
            full_list = get_nasdaq100_tickers()
        else:
            full_list = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

        # Apply Depth
        tickers_to_scan = full_list[:scan_depth]
        
        long_list = []
        cash_list = []
        
        st.session_state.scanner_results = None # Reset previous
        scan_prog = st.progress(0)
        status_text = st.empty()
        
        # -- Worker Function for Multithreading --
        def process_ticker_worker(tick):
            try:
                # 1. Market Cap Check
                mcap = get_market_cap(tick)
                if mcap_filter != "All" and mcap < mcap_map[mcap_filter]:
                    return None
                
                # 2. Fetch Data
                s_df = load_data(tick, start_date, end_date, interval=data_interval if live_mode else '1d')
                if s_df is None or s_df.empty:
                    return None
                
                # 3. Analyze
                s_analysis = get_master_signal(tick, s_df, 
                                              n_regimes=scan_reg_param, 
                                              freq=scan_freq, 
                                              opt_goal=scan_opt_goal,
                                              stability=scan_stability,
                                              switch_vol=scan_switch_vol,
                                              switch_trend=scan_switch_trend,
                                              engine=scan_engine_param,
                                              initial_cap=scan_initial_cap,
                                              trailing_stop=scan_trailing_stop,
                                              stop_loss=scan_stop_loss)
                if not s_analysis:
                    return None
                    
                s_price = s_df['Close'].iloc[-1]
                return {
                    'Ticker': tick,
                    'Price': round(s_price, 2),
                    'Mkt Cap ($B)': round(mcap / 1e9, 2),
                    'Regimes (N)': s_analysis['regime_data'].get('n_states', 4),
                    'Score': s_analysis['sentiment_score'],
                    'Regime': s_analysis['regime_label'],
                    'Trend': f"{s_analysis['trend_diff']:+.2%}",
                    'Action': s_analysis['regime_sig']
                }
            except:
                return None

        # Start Parallel Scan
        from concurrent.futures import as_completed
        
        # We use a reasonable number of workers to balance I/O and CPU
        # yfinance is I/O bound (network), models are CPU bound. 
        max_workers = 10 if scan_engine_param == "Markov" else 20
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tick = {executor.submit(process_ticker_worker, t): t for t in tickers_to_scan}
            
            for i, future in enumerate(as_completed(future_to_tick)):
                tick = future_to_tick[future]
                status_text.text(f"Processing {tick}... ({i+1}/{len(tickers_to_scan)})")
                
                result = future.result()
                if result:
                    if result['Score'] >= 1:
                        long_list.append(result)
                    else:
                        cash_list.append(result)
                
                scan_prog.progress((i + 1) / len(tickers_to_scan))
        
        # Store in session state for persistence
        st.session_state.scanner_results = {'long': long_list, 'cash': cash_list, 'universe': universe_type, 'count': len(tickers_to_scan)}

    # Always display results from session state if they exist
    if 'scanner_results' in st.session_state and st.session_state.scanner_results:
        res = st.session_state.scanner_results
        long_list = res['long']
        cash_list = res['cash']
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(f"🚀 LONG / OPEN ({len(long_list)})")
            if long_list:
                ldf = pd.DataFrame(long_list).sort_values(by='Score', ascending=False)
                st.dataframe(ldf.style.background_gradient(subset=['Score'], cmap='Greens'), use_container_width=True)
            else:
                st.info("No bullish signals found in current scan window.")
                
        with res_col2:
            st.subheader(f"🛑 CLOSED / CASH / HEDGE ({len(cash_list)})")
            if cash_list:
                cdf = pd.DataFrame(cash_list).sort_values(by='Score', ascending=True)
                st.dataframe(cdf.style.background_gradient(subset=['Score'], cmap='Reds'), use_container_width=True)
            else:
                st.info("No bearish/neutral signals found in current scan window.")

        st.divider()
        st.success(f"✅ **Total Market Review Complete**: Analyzed {res['count']} assets from `{res['universe']}` universe.")

with tab12:
    st.write("### 🏦 Federal Reserve Balance Sheet (Assets & Liabilities)")
    st.caption("Macroeconomic dashboard tracking FED liquidity and monetary policy shifts via FRED.")
    
    fed_date_col1, fed_date_col2 = st.columns(2)
    with fed_date_col1:
        fed_start_date = st.date_input("FED History Start", DEFAULT_NONLIVE_START)
    
    @st.fragment
    def render_fed_dashboard():
        with st.status("Fetching FRED Macro Data...", expanded=True) as status:
            # 1. Load Assets
            asset_dfs = {}
            for sid, name in FED_ASSETS.items():
                status.update(label=f"Loading Asset: {name}...")
                df = load_fred_data(sid)
                if df is not None:
                    asset_dfs[name] = df[sid]
            
            # 2. Load Liabilities
            liab_dfs = {}
            for sid, name in FED_LIABILITIES.items():
                status.update(label=f"Loading Liability: {name}...")
                df = load_fred_data(sid)
                if df is not None:
                    liab_dfs[name] = df[sid]
            
            status.update(label="All Macro data synchronized!", state="complete", expanded=False)
        
        if asset_dfs:
            # Use fillna(0) to handle different start dates or frequencies
            assets_master = pd.DataFrame(asset_dfs).fillna(0)
            assets_master = assets_master[assets_master.index >= pd.Timestamp(fed_start_date)]
            
            # Total Assets Plot
            st.subheader("Federal Reserve Assets (Stacked)")
            fig_assets = go.Figure()
            for col in assets_master.columns:
                fig_assets.add_trace(go.Scatter(x=assets_master.index, y=assets_master[col]/1e3, mode='lines', stackgroup='one', name=col))
            fig_assets.update_layout(title="FED Assets: Detailed Breakdown", yaxis_title="Amount (Billions $)", hovermode="x unified", template="plotly_dark", height=500)
            st.plotly_chart(fig_assets, use_container_width=True)
            
            # Total Balance Sheet Weekly Changes
            st.subheader("Weekly Change in Total Assets (WALCL)")
            walcl = load_fred_data("WALCL")
            if walcl is not None:
                walcl = walcl[walcl.index >= pd.Timestamp(fed_start_date)]
                # Weekly change in Billions
                walcl_diff = walcl.diff().dropna() / 1e3
                fig_diff = go.Figure()
                colors = ['green' if x >= 0 else 'red' for x in walcl_diff['WALCL']]
                fig_diff.add_trace(go.Bar(x=walcl_diff.index, y=walcl_diff['WALCL'], marker_color=colors, name="Weekly Change"))
                fig_diff.update_layout(title="FED Balance Sheet: Weekly Expansion/Contraction", yaxis_title="Change (Billions $)", hovermode="x unified", template="plotly_dark", height=400)
                st.plotly_chart(fig_diff, use_container_width=True)
        
        if liab_dfs:
            liabs_master = pd.DataFrame(liab_dfs).fillna(0)
            liabs_master = liabs_master[liabs_master.index >= pd.Timestamp(fed_start_date)]
            
            st.subheader("Federal Reserve Liabilities (Stacked)")
            fig_liabs = go.Figure()
            for col in liabs_master.columns:
                fig_liabs.add_trace(go.Scatter(x=liabs_master.index, y=liabs_master[col]/1e3, mode='lines', stackgroup='one', name=col))
            fig_liabs.update_layout(title="FED Liabilities & Capital Accounts", yaxis_title="Amount (Billions $)", hovermode="x unified", template="plotly_dark", height=500)
            st.plotly_chart(fig_liabs, use_container_width=True)
    
    render_fed_dashboard()


# ==========================================
# TAB 13: OPTIONS IV SURFACE
# ==========================================
with tab13:
    st.write("### 🎲 3D Implied Volatility Surface")
    st.markdown("Visualizes the Volatility Smile and Term Structure using live options data.")
    
    if df_main is None:
        st.warning("Please load a ticker to view its options surface.")
    else:
        with st.spinner(f"Fetching Live Options Chain for {TICKER}..."):
            try:
                tk = yf.Ticker(TICKER)
                expirations = tk.options
                if not expirations:
                    st.error(f"No options data available for {TICKER}.")
                else:
                    max_exp = st.slider("Max Expirations to Fetch", 1, min(20, len(expirations)), min(8, len(expirations)))
                    
                    surface_data = []
                    current_price = df_main['Close'].iloc[-1]
                    
                    for exp in expirations[:max_exp]:
                        opt = tk.option_chain(exp)
                        calls = opt.calls
                        
                        exp_date = datetime.strptime(exp, "%Y-%m-%d")
                        dte = (exp_date - datetime.now()).days
                        if dte < 1: dte = 1
                        
                        for _, row in calls.iterrows():
                            if row['impliedVolatility'] > 0 and row['volume'] > 0:
                                moneyness = row['strike'] / current_price
                                surface_data.append({
                                    'DTE': dte,
                                    'Moneyness': moneyness,
                                    'IV': row['impliedVolatility']
                                })
                                
                    if len(surface_data) > 0:
                        df_surf = pd.DataFrame(surface_data)
                        df_surf['Moneyness_Bin'] = df_surf['Moneyness'].round(2)
                        surf_pivot = df_surf.groupby(['DTE', 'Moneyness_Bin'])['IV'].mean().unstack()
                        
                        # Interpolate to fill grid
                        surf_pivot = surf_pivot.interpolate(method='linear', axis=1).bfill(axis=1).ffill(axis=1)
                        surf_pivot = surf_pivot.interpolate(method='linear', axis=0).bfill(axis=0).ffill(axis=0)
                        
                        fig_3d = go.Figure(data=[go.Surface(
                            z=surf_pivot.values,
                            x=surf_pivot.columns,
                            y=surf_pivot.index,
                            colorscale='Viridis'
                        )])
                        
                        fig_3d.update_layout(
                            title=f"{TICKER} Implied Volatility Surface (Calls)",
                            scene=dict(
                                xaxis_title='Moneyness (Strike/Price)',
                                yaxis_title='Days to Expiration (DTE)',
                                zaxis_title='Implied Volatility'
                            ),
                            template="plotly_dark",
                            height=700
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.warning("Not enough liquid options data to construct a surface.")
            except Exception as e:
                st.error(f"Error fetching options: {e}")

# ==========================================
# TAB 14: HURST EXPONENT
# ==========================================
with tab14:
    if df_main is None:
        st.warning("Please load a ticker to view Hurst Exponent Analysis.")
    else:
        st.write("### 🎲 Institutional Hurst Exponent")
        st.markdown("The Hurst Exponent (H) measures the long-term memory of a time series. \n* **H < 0.5**: Mean-Reverting (Anti-persistent)\n* **H = 0.5**: Random Walk (Geometric Brownian Motion)\n* **H > 0.5**: Trending (Persistent)")
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            h_window = st.slider("Rolling Window (Bars)", min_value=20, max_value=252, value=100, step=10, key="tab14_h_window")
        with col_h2:
            h_trend_thresh = st.slider("Trending Threshold (H >)", min_value=0.4, max_value=0.7, value=0.50, step=0.01, key="tab14_h_thresh")
            
        with st.spinner("Calculating Rolling Hurst Exponent..."):
            try:
                hurst_series = rolling_hurst(df_main['Close'], window=h_window)
                
                fig_hurst = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
                
                fig_hurst.add_trace(go.Scatter(x=df_main.index, y=df_main['Close'], mode='lines', line=dict(color='gray'), opacity=0.8, name='Price'), row=1, col=1)
                
                fig_hurst.add_trace(go.Scatter(x=hurst_series.index, y=hurst_series, mode='lines', line=dict(color='cyan'), name='Hurst (H)'), row=2, col=1)
                fig_hurst.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=1, annotation_text="Random Walk (0.5)")
                fig_hurst.add_hline(y=h_trend_thresh, line_dash="dash", line_color="green", row=2, col=1, annotation_text=f"Trend Entry ({h_trend_thresh})")
                
                # Highlight trending
                is_trending = (hurst_series > h_trend_thresh)
                highlight_plotly_zones(fig_hurst, is_trending, 'green', opacity=0.1, row=1, col=1)
                highlight_plotly_zones(fig_hurst, is_trending, 'green', opacity=0.1, row=2, col=1)
                
                # Highlight mean reverting
                is_mean_rev = (hurst_series < 0.5)
                highlight_plotly_zones(fig_hurst, is_mean_rev, 'purple', opacity=0.1, row=1, col=1)
                highlight_plotly_zones(fig_hurst, is_mean_rev, 'purple', opacity=0.1, row=2, col=1)
                
                fig_hurst.update_layout(title="Asset Memory & Persistence Profile", hovermode="x unified", template="plotly_dark", height=600)
                st.plotly_chart(fig_hurst, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error calculating Hurst Exponent: {str(e)}")

# TAB 15: HOT 10 (DAILY)
# ==========================================
with tab15:
    st.write("### 🔥 Daily Top 10 (Hot Stocks)")
    st.markdown("High-speed pre-scanner for identifying the best tactical momentum plays across the entire US market.")
    
    scan_col1, scan_col2, scan_col3, scan_col4 = st.columns(4)
    with scan_col1:
        hot_universe = st.selectbox("Market Universe", ["Total US Stocks (10,000+)", "S&P 500", "NASDAQ 100"], key="hot_univ")
    with scan_col2:
        hot_min_price = st.number_input("Minimum Price ($)", value=5.0, min_value=0.1)
    with scan_col3:
        vix_multiplier = st.number_input("VIX Volatility Multiplier", value=1.5, min_value=0.5, step=0.1, help="Required daily return relative to VIX implied daily move.")
    with scan_col4:
        top_n_buys = st.number_input("Target Top 'Buys'", value=10, min_value=1, max_value=50, step=1)
        
    if st.button("🚀 SCAN MARKET (HOT LIST)", use_container_width=True, type="primary", key="hot_scan_btn"):
        with st.spinner(f"Fetching {hot_universe} universe..."):
            if hot_universe == "S&P 500":
                all_tickers = get_sp500_tickers()
            elif hot_universe == "NASDAQ 100":
                all_tickers = get_nasdaq100_tickers()
            else:
                all_tickers = get_total_us_stocks()
            
        with st.spinner(f"Bulk downloading intraday market data for {len(all_tickers)} assets + ^VIX... (This may take 30-60 seconds)"):
            try:
                # Add VIX
                dl_tickers = all_tickers + ["^VIX"] if "^VIX" not in all_tickers else all_tickers
                
                import gc
                chunk_size = 500
                dfs = []
                progress_text = st.empty()
                
                for i in range(0, len(dl_tickers), chunk_size):
                    chunk = dl_tickers[i:i+chunk_size]
                    progress_text.text(f"Downloading batch {i//chunk_size + 1}/{(len(dl_tickers)//chunk_size) + 1} (High-Speed Mode)...")
                    
                    # By passing chunks of 500 with threads=True, we safely cap the OS threads to 500, preventing crashes while maximizing speed
                    chunk_df = yf.download(chunk, period="20d", threads=True, progress=False)
                    dfs.append(chunk_df)
                    gc.collect()
                
                progress_text.empty()
                
                if not dfs:
                    st.error("Failed to fetch market data.")
                else:
                    # Combine all downloaded chunks
                    df_bulk = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]
                    
                    closes = df_bulk['Close'].ffill()
                    opens = df_bulk['Open'].ffill() if 'Open' in df_bulk else None
                    vols = df_bulk['Volume'].ffill() if 'Volume' in df_bulk else None
                    
                    if len(closes) >= 2:
                        # Extract VIX
                        if "^VIX" in closes.columns:
                            vix_series = closes["^VIX"].dropna()
                            latest_vix = float(vix_series.iloc[-1]) if len(vix_series) > 0 else 15.0
                        else:
                            latest_vix = 15.0 # Fallback
                            
                        # Expected Daily Market Move (implied by VIX) = VIX / sqrt(252)
                        vix_daily_move_pct = (latest_vix / np.sqrt(252)) / 100
                        adaptive_thresh = vix_daily_move_pct * vix_multiplier
                        
                        st.info(f"📈 Market VIX is {latest_vix:.2f}. Adaptive Daily Move Threshold: {adaptive_thresh*100:.2f}%")
                        
                        # Vectorized calculations across all 6000+ assets instantly
                        last_close = closes.iloc[-1]
                        prev_close = closes.iloc[-2]
                        last_open = opens.iloc[-1] if opens is not None else prev_close
                        
                        daily_ret = (last_close - prev_close) / prev_close
                        gap_pct = (last_open - prev_close) / prev_close
                        intraday_pct = (last_close - last_open) / last_open
                        
                        if vols is not None:
                            last_vol = vols.iloc[-1]
                            if len(vols) >= 20:
                                avg_vol = vols.rolling(window=20).mean().iloc[-1]
                            else:
                                avg_vol = vols.mean()
                            rvol = last_vol / (avg_vol + 1)
                        else:
                            last_vol = pd.Series(0, index=closes.columns)
                            rvol = pd.Series(0, index=closes.columns)
                            
                        dollar_vol = last_close * last_vol
                        
                        # Apply Filters (Price > Min Price, Return > VIX Adaptive Threshold)
                        valid_mask = ((last_close >= hot_min_price) & (daily_ret > adaptive_thresh)).fillna(False)
                        valid_tickers = valid_mask[valid_mask].index.tolist()
                        
                        if "^VIX" in valid_tickers:
                            valid_tickers.remove("^VIX")
                        
                        hot_results = []
                        for tick in valid_tickers:
                            try:
                                score = float(daily_ret[tick]) / float(adaptive_thresh)
                                v = int(last_vol[tick]) if tick in last_vol and pd.notna(last_vol[tick]) else 0
                                rv = float(rvol[tick]) if tick in rvol and pd.notna(rvol[tick]) else 0.0
                                dv = float(dollar_vol[tick]) if tick in dollar_vol and pd.notna(dollar_vol[tick]) else 0.0
                                gp = float(gap_pct[tick]) if tick in gap_pct and pd.notna(gap_pct[tick]) else 0.0
                                intd = float(intraday_pct[tick]) if tick in intraday_pct and pd.notna(intraday_pct[tick]) else 0.0
                                
                                if dv >= 1_000_000:
                                    dv_str = f"${dv/1_000_000:.1f}M"
                                elif dv >= 1_000:
                                    dv_str = f"${dv/1_000:.1f}K"
                                else:
                                    dv_str = f"${dv:.0f}"
                                    
                                hot_results.append({
                                    "Ticker": str(tick),
                                    "Price": round(float(last_close[tick]), 2),
                                    "Daily Return %": round(float(daily_ret[tick]) * 100, 2),
                                    "Gap %": round(gp * 100, 2),
                                    "Intraday %": round(intd * 100, 2),
                                    "VIX Multiple": round(float(score), 2),
                                    "RVOL": round(rv, 2),
                                    "Dollar Vol": dv_str,
                                    "Volume": v
                                })
                            except:
                                pass
                                
                        if not hot_results:
                            st.warning("No stocks met the criteria today.")
                        else:
                            st.write("#### 🧠 Institutional Deep Verification (Strict BUY Filter)")
                            st.caption("Running the advanced Regime Model on high-momentum candidates. Only displaying confirmed pure BUY signals...")
                            
                            # Rank ALL candidates by momentum
                            hot_df_all = pd.DataFrame(hot_results).sort_values(by="VIX Multiple", ascending=False)
                            top_tickers = hot_df_all["Ticker"].tolist()
                            
                            verif_results = []
                            final_hot_list = []
                            prog = st.progress(0)
                            
                            # Scan all momentum candidates until we find our target number of strict buys
                            search_list = top_tickers
                            
                            for i, t in enumerate(search_list):
                                if len(final_hot_list) >= top_n_buys:
                                    break # We found our top N verified buys!
                                    
                                prog.progress((i+1)/len(search_list))
                                
                                # Load full history for deep quant
                                t_df = load_data(t, datetime.now() - timedelta(days=730), datetime.now(), interval='1d')
                                if t_df is not None and not t_df.empty:
                                    ans = get_master_signal(t, t_df, engine="GMM") # Use GMM for speed
                                    if ans:
                                        sig = str(ans.get('regime_sig', '')).upper()
                                        # Strict BUY check
                                        if "LONG" in sig or "BUY" in sig or "ACCUMULATE" in sig:
                                            verif_results.append({
                                                "Ticker": t,
                                                "Regime": ans['regime_label'],
                                                "Trend Deviation": f"{ans['trend_diff']:+.2%}",
                                                "Institutional Verdict": ans['regime_sig']
                                            })
                                            # Add the original momentum stats to the final display list
                                            stock_row = hot_df_all[hot_df_all["Ticker"] == t].iloc[0].to_dict()
                                            final_hot_list.append(stock_row)
                                            
                            prog.progress(1.0)
                            
                            if final_hot_list:
                                st.success(f"🔥 Found {len(final_hot_list)} Institutional-Grade BUY Stocks out of the top momentum leaders!")
                                final_df = pd.DataFrame(final_hot_list)
                                st.dataframe(final_df.style.background_gradient(subset=["Daily Return %", "VIX Multiple"], cmap="YlOrRd"), use_container_width=True)
                                
                                st.write("##### Deep Verification Details")
                                vdf = pd.DataFrame(verif_results)
                                st.dataframe(vdf, use_container_width=True)
                            else:
                                st.error("❌ No momentum candidates passed the strict Institutional BUY Verification today. Cash is a position.")
                    else:
                        st.warning("Not enough data returned from API to compute metrics.")
            except Exception as e:
                st.error(f"Error during bulk scan: {e}")

    st.markdown("---")
    with st.expander("📧 Automated Email Reporter (Full Market Scan)"):
        st.markdown("Run the complete 3-universe scan (S&P 500, NASDAQ 100, Total Market) and dispatch the final Institutional Hot List directly to your inbox.")
        
        email_col1, email_col2 = st.columns(2)
        with email_col1:
            sender_email = st.text_input("Sender Email (e.g., your Gmail)", key="email_sender")
            sender_pass = st.text_input("App Password", type="password", help="Use a 16-character App Password if using Gmail.", key="email_pass")
        with email_col2:
            receiver_email = st.text_input("Receiver Email", key="email_receiver")
            email_top_n = st.number_input("Target Top Buys per Universe", value=20, min_value=1, max_value=50, key="email_top_n")
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_manual = st.button("📨 Run Deep Scan & Send Email Report", use_container_width=True, type="primary")
        with col_btn2:
            run_auto = st.button("⏰ Schedule Daily 8:15 AM Automation", use_container_width=True)
            
        if run_auto:
            if not sender_email or not sender_pass or not receiver_email:
                st.error("Please fill in all email credential fields before scheduling.")
            else:
                import sys
                import os
                import subprocess
                
                # Get the python executable and the exact path to the automated script I built
                python_bin = sys.executable
                script_path = os.path.join(os.path.dirname(__file__), "daily_email_scanner.py")
                
                # The cron string (15 8 * * 1-5 = 8:15 AM Mon-Fri)
                cron_job = f"15 8 * * 1-5 SENDER_EMAIL='{sender_email}' SENDER_PASSWORD='{sender_pass}' RECEIVER_EMAIL='{receiver_email}' {python_bin} {script_path} >> {os.path.dirname(__file__)}/scanner.log 2>&1"
                
                try:
                    # Read existing crontab using absolute path to prevent PATH errors
                    crontab_out = subprocess.run(['/usr/bin/crontab', '-l'], capture_output=True, text=True)
                    existing_cron = crontab_out.stdout if crontab_out.returncode == 0 else ""
                    
                    # Remove any existing scanner jobs to prevent duplicates if they update credentials
                    new_cron = "\n".join([line for line in existing_cron.split('\n') if "daily_email_scanner.py" not in line and line.strip() != ""])
                    
                    # Append the brand new 8:15 AM job
                    new_cron += f"\n{cron_job}\n"
                    
                    # Write back to system crontab
                    process = subprocess.Popen(['/usr/bin/crontab', '-'], stdin=subprocess.PIPE, text=True)
                    process.communicate(new_cron)
                    
                    st.success("✅ **Automation scheduled!** Your Mac will now silently run the scan every weekday at 8:15 AM and email you the results. You do not need to keep the app open.")
                except Exception as e:
                    st.error(f"Failed to schedule automation. Error: {e}")
                    
        if run_manual:
            if not sender_email or not sender_pass or not receiver_email:
                st.error("Please fill in all email credential fields.")
            else:
                st.info("Executing 3-Universe Deep Quant Scan... (This will take a few minutes. Do not refresh the page.)")
                
                # Define headless scan function for the emailer
                def scan_universe_headless(universe_name, top_n_buys, min_price, vix_mult):
                    import gc
                    if universe_name == "S&P 500":
                        all_tick = get_sp500_tickers()
                    elif universe_name == "NASDAQ 100":
                        all_tick = get_nasdaq100_tickers()
                    else:
                        all_tick = get_total_us_stocks()
                        
                    dl_tickers = all_tick + ["^VIX"] if "^VIX" not in all_tick else all_tick
                    import gc
                    chunk_size = 500
                    dfs = []
                    
                    for i in range(0, len(dl_tickers), chunk_size):
                        chunk = dl_tickers[i:i+chunk_size]
                        chunk_df = yf.download(chunk, period="20d", threads=True, progress=False)
                        dfs.append(chunk_df)
                        gc.collect()
                        
                    if not dfs: return pd.DataFrame()
                    
                    df_bulk = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]
                    closes = df_bulk['Close'].ffill()
                    opens = df_bulk['Open'].ffill() if 'Open' in df_bulk else None
                    vols = df_bulk['Volume'].ffill() if 'Volume' in df_bulk else None
                    if len(closes) < 2: return pd.DataFrame()
                    
                    latest_vix = float(closes["^VIX"].dropna().iloc[-1]) if "^VIX" in closes.columns and len(closes["^VIX"].dropna()) > 0 else 15.0
                    vix_daily_move_pct = (latest_vix / np.sqrt(252)) / 100
                    adaptive_thresh = vix_daily_move_pct * vix_mult
                    
                    last_close = closes.iloc[-1]
                    prev_close = closes.iloc[-2]
                    last_open = opens.iloc[-1] if opens is not None else prev_close
                    
                    daily_ret = (last_close - prev_close) / prev_close
                    gap_pct = (last_open - prev_close) / prev_close
                    intraday_pct = (last_close - last_open) / last_open
                    
                    if vols is not None:
                        last_vol = vols.iloc[-1]
                        if len(vols) >= 20:
                            avg_vol = vols.rolling(window=20).mean().iloc[-1]
                        else:
                            avg_vol = vols.mean()
                        rvol = last_vol / (avg_vol + 1)
                    else:
                        last_vol = pd.Series(0, index=closes.columns)
                        rvol = pd.Series(0, index=closes.columns)
                        
                    dollar_vol = last_close * last_vol
                    
                    valid_mask = ((last_close >= min_price) & (daily_ret > adaptive_thresh)).fillna(False)
                    valid_tickers = valid_mask[valid_mask].index.tolist()
                    if "^VIX" in valid_tickers: valid_tickers.remove("^VIX")
                    
                    hot_results = []
                    for tick in valid_tickers:
                        try:
                            score = float(daily_ret[tick]) / float(adaptive_thresh)
                            v = int(last_vol[tick]) if tick in last_vol and pd.notna(last_vol[tick]) else 0
                            rv = float(rvol[tick]) if tick in rvol and pd.notna(rvol[tick]) else 0.0
                            dv = float(dollar_vol[tick]) if tick in dollar_vol and pd.notna(dollar_vol[tick]) else 0.0
                            gp = float(gap_pct[tick]) if tick in gap_pct and pd.notna(gap_pct[tick]) else 0.0
                            intd = float(intraday_pct[tick]) if tick in intraday_pct and pd.notna(intraday_pct[tick]) else 0.0
                            
                            if dv >= 1_000_000:
                                dv_str = f"${dv/1_000_000:.1f}M"
                            elif dv >= 1_000:
                                dv_str = f"${dv/1_000:.1f}K"
                            else:
                                dv_str = f"${dv:.0f}"
                                
                            hot_results.append({
                                "Ticker": str(tick),
                                "Price": round(float(last_close[tick]), 2),
                                "Daily Return %": round(float(daily_ret[tick]) * 100, 2),
                                "Gap %": round(gp * 100, 2),
                                "Intraday %": round(intd * 100, 2),
                                "VIX Multiple": round(float(score), 2),
                                "RVOL": round(rv, 2),
                                "Dollar Vol": dv_str,
                                "Volume": v
                            })
                        except: pass
                        
                    if not hot_results: return pd.DataFrame()
                    
                    hot_df_all = pd.DataFrame(hot_results).sort_values(by="VIX Multiple", ascending=False)
                    top_tickers = hot_df_all["Ticker"].tolist()
                    final_hot_list = []
                    
                    for t in top_tickers:
                        if len(final_hot_list) >= top_n_buys: break
                        t_df = load_data(t, datetime.now() - timedelta(days=730), datetime.now(), interval='1d')
                        if t_df is not None and not t_df.empty:
                            ans = get_master_signal(t, t_df, engine="GMM")
                            if ans:
                                sig = str(ans.get('regime_sig', '')).upper()
                                if "LONG" in sig or "BUY" in sig or "ACCUMULATE" in sig:
                                    stock_row = hot_df_all[hot_df_all["Ticker"] == t].iloc[0].to_dict()
                                    stock_row["Regime"] = ans['regime_label']
                                    stock_row["Trend Deviation"] = f"{ans['trend_diff']:+.2%}"
                                    stock_row["Verdict"] = ans['regime_sig']
                                    final_hot_list.append(stock_row)
                                    
                    return pd.DataFrame(final_hot_list)

                # Run the scans
                universes = {"S&P 500": email_top_n, "NASDAQ 100": email_top_n, "Total US Market": email_top_n}
                results = {}
                
                scan_prog = st.progress(0)
                for idx, (univ, n_buys) in enumerate(universes.items()):
                    st.write(f"🔍 Scanning {univ}...")
                    results[univ] = scan_universe_headless(univ, n_buys, 5.0, 1.5)
                    scan_prog.progress((idx + 1) / len(universes))
                    
                # Format HTML Email
                html = f"""
                <html>
                  <head>
                    <style>
                      body {{ font-family: sans-serif; background-color: #f4f7f6; color: #333; padding: 20px; }}
                      h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                      h3 {{ color: #2980b9; margin-top: 30px; }}
                      table {{ width: 100%; border-collapse: collapse; background-color: #fff; }}
                      th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                      th {{ background-color: #34495e; color: #fff; }}
                    </style>
                  </head>
                  <body>
                    <h2>Institutional Hot List - Daily Momentum Scan</h2>
                    <p>Generated on: <strong>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
                """
                for univ_name, df in results.items():
                    html += f"<h3>{univ_name} (Top {len(df)} Confirmed Setups)</h3>"
                    if df.empty:
                        html += "<p><em>No actionable institutional setups found. Cash is a position.</em></p>"
                    else:
                        html += df.to_html(index=False, border=0)
                html += "</body></html>"
                
                # Send Email
                try:
                    import smtplib
                    from email.mime.multipart import MIMEMultipart
                    from email.mime.text import MIMEText
                    
                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = f"🚀 Institutional Hot List: {datetime.now().strftime('%Y-%m-%d')}"
                    msg['From'] = sender_email
                    msg['To'] = receiver_email
                    msg.attach(MIMEText(html, 'html'))
                    
                    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                    server.login(sender_email, sender_pass)
                    server.sendmail(sender_email, receiver_email, msg.as_string())
                    server.quit()
                    
                    st.success(f"✅ Full report successfully emailed to {receiver_email}!")
                except Exception as e:
                    st.error(f"Failed to send email. Check credentials. Error: {str(e)}")

with tab16:
    st.markdown("## 🎯 Institutional IV-Based Stock Scanner")
    st.markdown("**Precision targeting using Implied Volatility dynamics — the way hedge funds screen for high-conviction setups.**\n\nThe model identifies stocks where IV structure signals institutional accumulation or directional conviction.")
    
    with st.expander("⚙️ Scanner Configuration", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            universe_choice = st.selectbox("Universe", ["S&P 500 (503 stocks)", "NASDAQ 100 (101 stocks)", "Custom Watchlist"])
            scan_depth = st.number_input("Scan Depth (# tickers)", min_value=5, max_value=503, value=50, step=5)
            workers = st.slider("Parallel Workers", 1, 20, 10)
        with col_c2:
            min_ivr = st.slider("Min IV Rank", 0, 100, 0)
            max_ivr = st.slider("Max IV Rank", 0, 100, 100)
            max_pc = st.slider("Max P/C Ratio", 0.1, 3.0, 1.5, step=0.1)
        with col_c3:
            min_score = st.slider("Min Signal Score", 0.0, 6.0, 2.5, step=0.5)
            min_mktcap = st.select_slider("Min Market Cap", options=["Any", "$500M", "$2B", "$10B", "$50B"], value="$2B")
            mktcap_map = {"Any": 0, "$500M": 5e8, "$2B": 2e9, "$10B": 1e10, "$50B": 5e10}
            min_mktcap_val = mktcap_map[min_mktcap]
            
        custom_list = ""
        if universe_choice == "Custom Watchlist":
            custom_list = st.text_area("Tickers (comma separated)", "AAPL, TSLA, NVDA, META, AMZN")
            
        setup_filter = st.multiselect("Include Setups", ["IV Expansion from Low Base", "Bullish Call Skew", "Call Buying Dominance", "IV Crush Setup", "Contango IV Structure", "Heavy Call OI"], default=["IV Expansion from Low Base", "Bullish Call Skew", "Call Buying Dominance"])

    st.subheader("🔍 Single Ticker Deep Dive")
    col_td1, col_td2 = st.columns([1, 3])
    with col_td1:
        single_ticker = st.text_input("Deep Dive Ticker", "AAPL").upper()
        run_single = st.button("Analyze Ticker", type="primary", use_container_width=True)

    if run_single:
        with st.spinner(f"Fetching options data for {single_ticker}..."):
            result = get_iv_metrics(single_ticker)

        if result is None:
            st.error(f"No options data available for {single_ticker}, or insufficient history.")
        else:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Price", f"${result['price']:.2f}")
            c2.metric("ATM IV", f"{result['atm_iv']:.1f}%")
            c3.metric("IV Rank", f"{result['iv_rank']:.0f}/100", delta="High" if result['iv_rank'] > 50 else "Low", delta_color="inverse" if result['iv_rank'] > 70 else "normal")
            c4.metric("IV/HV Ratio", f"{result['iv_hv_ratio']:.2f}", delta="Rich" if result['iv_hv_ratio'] > 1.2 else "Cheap")
            c5.metric("P/C Ratio", f"{result['pc_ratio']:.2f}", delta="Calls dominant" if result['pc_ratio'] < 0.8 else "Puts dominant", delta_color="normal" if result['pc_ratio'] < 0.8 else "inverse")
            c6.metric("Skew", f"{result['skew']:.1f}%", delta="Call skew ↑" if result['skew'] < -2 else "Put skew ↑", delta_color="normal" if result['skew'] < -2 else "inverse")
            st.divider()
            
            score = result['score']
            verdict = result['verdict']
            col_v1, col_v2 = st.columns([1, 2])
            with col_v1:
                st.markdown(f'<div style="background:{result["verdict_color"]}22; border:2px solid {result["verdict_color"]}; border-radius:12px; padding:20px; text-align:center;"><h2 style="color:{result["verdict_color"]}; margin:0;">{verdict}</h2><p style="margin:4px 0; font-size:1.2em;">Score: {score:.1f} / 6.0</p><p style="margin:0; color:#aaa;">{result["name"]} | {result["sector"]}</p></div>', unsafe_allow_html=True)
            with col_v2:
                st.write("#### Active Signals")
                if result['signals']:
                    for sig_name, sig_color, sig_detail in result['signals']:
                        icon = {"green": "✅", "orange": "⚠️", "red": "🔴"}.get(sig_color, "•")
                        st.markdown(f"{icon} **{sig_name}**: {sig_detail}")
                else:
                    st.info("No strong directional signals detected.")
            st.divider()

            col_surf1, col_surf2 = st.columns(2)
            with col_surf1:
                st.write("#### IV Smile (Near-Term)")
                surface_df = build_iv_surface(single_ticker, result['price'])
                if surface_df is not None and not surface_df.empty:
                    dte_options = sorted(surface_df['DTE'].unique())
                    fig_smile = go.Figure()
                    colors = ['#00f2ff', '#ff6b35', '#a855f7', '#22c55e']
                    for i, dte in enumerate(dte_options[:4]):
                        smile_data = surface_df[surface_df['DTE'] == dte].sort_values('Moneyness')
                        if len(smile_data) >= 3:
                            fig_smile.add_trace(go.Scatter(x=smile_data['Moneyness'], y=smile_data['IV'], mode='lines+markers', name=f"{dte}d", line=dict(color=colors[i % len(colors)], width=2), marker=dict(size=5)))
                    fig_smile.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="ATM")
                    fig_smile.update_layout(title="IV Smile by Expiration", xaxis_title="Moneyness (% from ATM)", yaxis_title="Implied Volatility (%)", template="plotly_dark", height=350, legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig_smile, use_container_width=True)
                else:
                    st.info("Insufficient options data for smile chart.")
            with col_surf2:
                st.write("#### Historical Vol Comparison")
                fig_hv = go.Figure()
                categories = ['HV 30d', 'HV 60d', 'HV 252d', 'ATM IV']
                values = [result['hv_30'], result['hv_60'], result['hv_252'], result['atm_iv']]
                fig_hv.add_trace(go.Bar(x=categories, y=values, marker_color=['#4488ff', '#3377ee', '#2266dd', '#ff6b35'], text=[f"{v:.1f}%" for v in values], textposition='outside'))
                fig_hv.update_layout(title="IV vs Historical Volatility", yaxis_title="Volatility (%)", template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig_hv, use_container_width=True)

            st.write("#### IV Rank Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=result['iv_rank'], domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "IV Rank (0=Historically Low, 100=Historically High)", 'font': {'color': 'white'}},
                delta={'reference': 50, 'increasing': {'color': "orange"}, 'decreasing': {'color': "green"}},
                gauge={'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': result['verdict_color']}, 'bgcolor': "gray", 'bordercolor': "white", 'steps': [{'range': [0, 30], 'color': '#00441b'}, {'range': [30, 70], 'color': '#525252'}, {'range': [70, 100], 'color': '#67000d'}], 'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.75, 'value': result['iv_rank']}}
            ))
            fig_gauge.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    st.subheader("📡 Bulk IV Scanner")
    st.markdown("Scans the selected universe and ranks stocks by institutional IV conviction.")
    
    if st.button("🚀 Run IV Scan", type="primary", use_container_width=True):
        if universe_choice == "S&P 500 (503 stocks)":
            full_universe = get_sp500()
        elif universe_choice == "NASDAQ 100 (101 stocks)":
            full_universe = get_nasdaq100()
        else:
            full_universe = [t.strip().upper() for t in custom_list.split(",") if t.strip()]
            
        universe = full_universe[:scan_depth]
        st.info(f"Scanning {len(universe)} tickers from {universe_choice}...")
        
        results = []
        prog = st.progress(0)
        status = st.empty()
        errors = []
        
        def worker(ticker):
            return get_iv_metrics(ticker)
            
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, t): t for t in universe}
            for i, future in enumerate(as_completed(futures)):
                t = futures[future]
                try:
                    r = future.result()
                    if r is not None:
                        results.append(r)
                except Exception as e:
                    errors.append(t)
                prog.progress((i + 1) / len(universe))
                status.text(f"Processed {i+1}/{len(universe)} | Found {len(results)} valid | Skipped {len(errors)}")
                
        prog.empty()
        status.empty()
        
        if not results:
            st.error("No results returned. Check ticker symbols or try a different universe.")
        else:
            df_results = pd.DataFrame(results)
            filtered = df_results.copy()
            if min_mktcap_val > 0:
                filtered = filtered[filtered['mkt_cap'] >= min_mktcap_val]
            filtered = filtered[(filtered['iv_rank'] >= min_ivr) & (filtered['iv_rank'] <= max_ivr) & (filtered['pc_ratio'] <= max_pc) & (filtered['score'] >= min_score)]
            
            if setup_filter:
                def has_setup(signals):
                    return any(sf in [s[0] for s in signals] for sf in setup_filter)
                filtered = filtered[filtered['signals'].apply(has_setup)]
                
            filtered = filtered.sort_values('score', ascending=False)
            st.write(f"### Results: {len(filtered)} stocks match criteria (from {len(df_results)} analyzed)")
            
            if filtered.empty:
                st.warning("No stocks matched all filters. Try relaxing the criteria in the sidebar.")
            else:
                display_cols = ['ticker', 'name', 'price', 'atm_iv', 'iv_rank', 'iv_hv_ratio', 'skew', 'pc_ratio', 'score', 'verdict', 'sector']
                display_df = filtered[display_cols].copy()
                display_df.columns = ['Ticker', 'Name', 'Price', 'ATM IV%', 'IVR', 'IV/HV', 'Skew%', 'P/C', 'Score', 'Verdict', 'Sector']
                
                def style_verdict(val):
                    return {'STRONG BUY': 'background-color: #00441b; color: #00ff88', 'BUY': 'background-color: #1a472a; color: #44cc66', 'WATCH': 'background-color: #3d3000; color: #ffcc00', 'NEUTRAL': 'background-color: #2a2a2a; color: #aaaaaa', 'AVOID': 'background-color: #3d0000; color: #ff4444'}.get(val, '')
                    
                styled = display_df.style.format({'Price': '${:.2f}', 'ATM IV%': '{:.1f}%', 'IVR': '{:.0f}', 'IV/HV': '{:.2f}', 'Skew%': '{:.1f}', 'P/C': '{:.2f}', 'Score': '{:.1f}'}).map(style_verdict, subset=['Verdict']).background_gradient(subset=['Score'], cmap='RdYlGn')
                st.dataframe(styled, use_container_width=True, height=500)
                
                csv = display_df.to_csv(index=False)
                st.download_button("📥 Download Results CSV", csv, file_name=f"iv_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
                
                st.divider()
                st.write("#### Top 10 Institutional IV Setups")
                top10 = filtered.head(10)
                for _, row in top10.iterrows():
                    with st.expander(f"{'🟢' if 'BUY' in row['verdict'] else '🟡'} {row['ticker']} — {row['name']} | Score: {row['score']:.1f} | {row['verdict']}"):
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Price", f"${row['price']:.2f}")
                        m2.metric("ATM IV", f"{row['atm_iv']:.1f}%")
                        m3.metric("IV Rank", f"{row['iv_rank']:.0f}")
                        m4.metric("IV/HV", f"{row['iv_hv_ratio']:.2f}")
                        m5.metric("P/C", f"{row['pc_ratio']:.2f}")
                        st.write("**Active signals:**")
                        for sig_name, sig_color, sig_detail in row['signals']:
                            icon = "✅" if sig_color == "green" else "⚠️" if sig_color == "orange" else "🔴"
                            st.markdown(f"{icon} **{sig_name}**: {sig_detail}")
                            
                        fig_mini = go.Figure(go.Bar(x=['HV 30d', 'HV 252d', 'ATM IV'], y=[row['hv_30'], row['hv_252'], row['atm_iv']], marker_color=['#4488ff', '#2266dd', '#ff6b35'], text=[f"{v:.1f}%" for v in [row['hv_30'], row['hv_252'], row['atm_iv']]], textposition='outside'))
                        fig_mini.update_layout(height=200, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10), showlegend=False, yaxis_title="Vol %")
                        st.plotly_chart(fig_mini, use_container_width=True)

                st.divider()
                st.write("#### Sector Distribution of Signals")
                sector_counts = filtered.groupby('sector')['score'].agg(['count', 'mean']).reset_index()
                sector_counts.columns = ['Sector', 'Count', 'Avg Score']
                sector_counts = sector_counts.sort_values('Count', ascending=True)
                fig_sector = go.Figure(go.Bar(x=sector_counts['Count'], y=sector_counts['Sector'], orientation='h', marker=dict(color=sector_counts['Avg Score'], colorscale='RdYlGn', showscale=True, colorbar=dict(title="Avg Score")), text=[f"{c} stocks (avg {s:.1f})" for c, s in zip(sector_counts['Count'], sector_counts['Avg Score'])], textposition='outside'))
                fig_sector.update_layout(title="Sectors with IV Conviction Signals", template="plotly_dark", height=max(300, len(sector_counts) * 35), xaxis_title="Number of Stocks")
                st.plotly_chart(fig_sector, use_container_width=True)
                # ==========================================
# *** PASTE INSTRUCTIONS ***
# 1. In the existing tabs = st.tabs([...]) block, ADD these 3 entries at the end of the list:
#    "📊 CVD & Volume Delta",
#    "📈 Institutional VWAP",
#    "🔬 Time Series Analysis"
#
# 2. Change the unpacking line from:
#    tab0, tab1, ... tab16 = tabs
#    TO:
#    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19 = tabs
#
# 3. DELETE the last 2 footer lines from original file:
#    st.markdown("---")
#    st.caption("Generated via Gemini 2.0 Flash | Robust Financial Thesis Implementation")
#
# 4. PASTE everything below at the end of the file.
# ==========================================


# ==========================================
# TAB 17: CUMULATIVE VOLUME DELTA (CVD)
# ==========================================
with tab17:
    st.write("### 📊 Institutional Cumulative Volume Delta (CVD)")
    st.markdown("""
    **CVD** measures the net buying vs selling pressure over time by classifying each bar's volume
    as buy-initiated or sell-initiated. Rising CVD + Rising Price = Confirmed Uptrend.
    Divergence between CVD and Price = Institutional Warning Signal.
    """)

    if df_main is None:
        st.warning("Please load a ticker to view CVD analysis.")
    else:
        # ── Configuration ──────────────────────────────────────────────────
        cvd_col1, cvd_col2, cvd_col3 = st.columns(3)
        with cvd_col1:
            cvd_method = st.selectbox("Volume Classification Method", [
                "Aggressive (Close vs Open)",
                "Tick Rule (Close vs Prior Close)",
                "High-Low Weighted",
                "True Strength (OHLCV)"
            ], help="How each bar's volume is split into buy vs sell pressure.")
        with cvd_col2:
            cvd_smooth = st.slider("CVD Smoothing (EMA span)", 1, 20, 3,
                                   help="1 = raw CVD, higher = smoother signal")
        with cvd_col3:
            cvd_lookback = st.slider("Divergence Lookback (bars)", 5, 60, 20)

        df_cvd = df_main.copy()

        # ── Volume Delta Calculation ────────────────────────────────────────
        try:
            hi = df_cvd['High']
            lo = df_cvd['Low']
            op = df_cvd['Open']
            cl = df_cvd['Close']
            vol = df_cvd['Volume'] if 'Volume' in df_cvd.columns else pd.Series(
                np.ones(len(df_cvd)), index=df_cvd.index)

            if cvd_method == "Aggressive (Close vs Open)":
                # Positive delta when close > open (buyers won the bar)
                direction = np.sign(cl - op)
                delta = vol * direction

            elif cvd_method == "Tick Rule (Close vs Prior Close)":
                direction = np.sign(cl - cl.shift(1)).fillna(0)
                delta = vol * direction

            elif cvd_method == "High-Low Weighted":
                # Fraction of vol attributed to buys based on where close lands in H-L range
                hl_range = (hi - lo).replace(0, np.nan)
                buy_frac = ((cl - lo) / hl_range).fillna(0.5)
                sell_frac = 1 - buy_frac
                delta = vol * (buy_frac - sell_frac)

            else:  # True Strength (OHLCV)
                # Kauffman-style: weights upper wick as selling, lower wick as buying
                hl_range = (hi - lo).replace(0, np.nan)
                upper_wick = hi - np.maximum(op, cl)
                lower_wick = np.minimum(op, cl) - lo
                body = np.abs(cl - op)
                buy_pressure = lower_wick + 0.5 * body * (cl > op).astype(float)
                sell_pressure = upper_wick + 0.5 * body * (cl <= op).astype(float)
                delta = vol * (buy_pressure - sell_pressure) / hl_range.fillna(1)

            delta = delta.fillna(0)
            cvd = delta.cumsum()

            # Optional smoothing
            if cvd_smooth > 1:
                cvd_plot = cvd.ewm(span=cvd_smooth, adjust=False).mean()
            else:
                cvd_plot = cvd

            # ── Divergence Detection ────────────────────────────────────────
            price_change = cl - cl.shift(cvd_lookback)
            cvd_change   = cvd - cvd.shift(cvd_lookback)

            bull_div = (price_change < 0) & (cvd_change > 0)   # Price down, CVD up  → hidden bull
            bear_div = (price_change > 0) & (cvd_change < 0)   # Price up, CVD down  → hidden bear

            # ── Rolling Delta Bars (daily net flow) ─────────────────────────
            rolling_delta = delta.rolling(window=5).sum()

            # ── Metrics ────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Cumulative Delta (Total)", f"{cvd.iloc[-1]:+,.0f}")
            m2.metric("Last Bar Delta", f"{delta.iloc[-1]:+,.0f}")
            m3.metric("5-Bar Rolling Delta", f"{rolling_delta.iloc[-1]:+,.0f}")
            latest_pressure = "BUYING" if delta.iloc[-1] > 0 else "SELLING"
            m4.metric("Latest Pressure", latest_pressure,
                      delta="Bullish" if latest_pressure == "BUYING" else "Bearish",
                      delta_color="normal" if latest_pressure == "BUYING" else "inverse")

            if bear_div.iloc[-1]:
                st.error("🚨 **BEARISH DIVERGENCE**: Price rising but CVD declining — institutional distribution detected.")
            elif bull_div.iloc[-1]:
                st.success("✅ **BULLISH DIVERGENCE**: Price falling but CVD rising — institutional accumulation detected.")
            else:
                st.info("📊 No significant CVD divergence at current bar.")

            # ── Main Chart ─────────────────────────────────────────────────
            fig_cvd = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.35, 0.25, 0.25, 0.15],
                subplot_titles=(
                    f"{TICKER} Price",
                    "Cumulative Volume Delta (CVD)",
                    "Bar-by-Bar Volume Delta",
                    "5-Bar Rolling Net Flow"
                )
            )

            # Row 1 – Price (candlestick if possible, else line)
            if all(c in df_cvd.columns for c in ['Open', 'High', 'Low', 'Close']):
                fig_cvd.add_trace(go.Candlestick(
                    x=df_cvd.index, open=df_cvd['Open'], high=df_cvd['High'],
                    low=df_cvd['Low'], close=df_cvd['Close'],
                    name="Price", increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350', showlegend=False
                ), row=1, col=1)
            else:
                fig_cvd.add_trace(go.Scatter(
                    x=df_cvd.index, y=cl, mode='lines',
                    line=dict(color='gray', width=1.5), name="Price"
                ), row=1, col=1)

            # Mark divergences on price chart
            bull_dates = df_cvd.index[bull_div]
            bear_dates = df_cvd.index[bear_div]
            if len(bull_dates) > 0:
                fig_cvd.add_trace(go.Scatter(
                    x=bull_dates, y=cl[bull_div], mode='markers',
                    marker=dict(symbol='triangle-up', color='lime', size=10),
                    name="Bull Divergence"
                ), row=1, col=1)
            if len(bear_dates) > 0:
                fig_cvd.add_trace(go.Scatter(
                    x=bear_dates, y=cl[bear_div], mode='markers',
                    marker=dict(symbol='triangle-down', color='red', size=10),
                    name="Bear Divergence"
                ), row=1, col=1)

            # Row 2 – CVD
            fig_cvd.add_trace(go.Scatter(
                x=df_cvd.index, y=cvd_plot, mode='lines',
                line=dict(color='#00f2ff', width=2), name="CVD"
            ), row=2, col=1)
            fig_cvd.add_hline(y=0, line_dash="dash", line_color="white",
                              opacity=0.3, row=2, col=1)
            # Shade positive / negative CVD
            cvd_pos = cvd_plot.copy(); cvd_pos[cvd_pos < 0] = 0
            cvd_neg = cvd_plot.copy(); cvd_neg[cvd_neg > 0] = 0
            fig_cvd.add_trace(go.Scatter(
                x=df_cvd.index, y=cvd_pos, fill='tozeroy',
                mode='lines', line=dict(width=0),
                fillcolor='rgba(0,255,100,0.15)', showlegend=False
            ), row=2, col=1)
            fig_cvd.add_trace(go.Scatter(
                x=df_cvd.index, y=cvd_neg, fill='tozeroy',
                mode='lines', line=dict(width=0),
                fillcolor='rgba(255,50,50,0.15)', showlegend=False
            ), row=2, col=1)

            # Row 3 – Bar Delta (colored bars)
            bar_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in delta]
            fig_cvd.add_trace(go.Bar(
                x=df_cvd.index, y=delta,
                marker_color=bar_colors, name="Bar Delta"
            ), row=3, col=1)

            # Row 4 – Rolling Net Flow
            roll_colors = ['#00ff88' if v >= 0 else '#ff4444' for v in rolling_delta]
            fig_cvd.add_trace(go.Bar(
                x=df_cvd.index, y=rolling_delta,
                marker_color=roll_colors, name="5-Bar Flow"
            ), row=4, col=1)

            fig_cvd.update_layout(
                height=900, hovermode="x unified", template="plotly_dark",
                title=f"Institutional CVD Dashboard — {TICKER}",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_cvd, use_container_width=True)

            # ── CVD Trade Log ──────────────────────────────────────────────
            st.divider()
            st.write("#### 📝 CVD Trade Signal Log")
            st.caption("Signals fired when CVD crosses its own rolling mean — institutional-grade entry/exit confirmation.")

            cvd_ma = cvd.rolling(window=cvd_lookback).mean()
            cvd_cross_up   = (cvd > cvd_ma) & (cvd.shift(1) <= cvd_ma.shift(1))
            cvd_cross_down = (cvd < cvd_ma) & (cvd.shift(1) >= cvd_ma.shift(1))

            trade_log_rows = []
            for dt in df_cvd.index[cvd_cross_up]:
                trade_log_rows.append({
                    "Date": dt.date(),
                    "Signal": "🟢 BUY (CVD Cross Up)",
                    "Price": round(float(cl.loc[dt]), 2),
                    "CVD at Signal": round(float(cvd.loc[dt]), 0),
                    "Bar Delta": round(float(delta.loc[dt]), 0),
                    "5-Bar Flow": round(float(rolling_delta.loc[dt]), 0),
                    "Divergence": "Bull Div" if bull_div.loc[dt] else "None"
                })
            for dt in df_cvd.index[cvd_cross_down]:
                trade_log_rows.append({
                    "Date": dt.date(),
                    "Signal": "🔴 SELL (CVD Cross Down)",
                    "Price": round(float(cl.loc[dt]), 2),
                    "CVD at Signal": round(float(cvd.loc[dt]), 0),
                    "Bar Delta": round(float(delta.loc[dt]), 0),
                    "5-Bar Flow": round(float(rolling_delta.loc[dt]), 0),
                    "Divergence": "Bear Div" if bear_div.loc[dt] else "None"
                })

            if trade_log_rows:
                tlog_df = pd.DataFrame(trade_log_rows).sort_values("Date", ascending=False)
                st.dataframe(tlog_df, use_container_width=True)

                # CVD performance quick-check
                buys  = tlog_df[tlog_df['Signal'].str.contains("BUY")]
                sells = tlog_df[tlog_df['Signal'].str.contains("SELL")]
                st.caption(f"Total CVD Signals: {len(tlog_df)} | Buy Signals: {len(buys)} | Sell Signals: {len(sells)}")

                # Download
                csv_cvd = tlog_df.to_csv(index=False)
                st.download_button("📥 Download CVD Signal Log", csv_cvd,
                                   file_name=f"CVD_SignalLog_{TICKER}.csv", mime="text/csv")
            else:
                st.info("No CVD crossover signals in the selected date range.")

            # ── CVD Adaptive Strategy Backtest ──────────────────────────────
            st.divider()
            st.write("#### 🧪 CVD Strategy Backtest")
            st.caption("Goal: beat buy & hold by using CVD confirmation, price trend, and risk-off exits instead of one weak CVD mean-cross rule.")

            cvd_ma = cvd.rolling(window=cvd_lookback).mean()
            cvd_fast = cvd.ewm(span=max(3, cvd_lookback // 3), adjust=False).mean()
            cvd_slow = cvd.ewm(span=max(8, cvd_lookback), adjust=False).mean()
            ema20 = cl.ewm(span=20, adjust=False).mean()
            ema50 = cl.ewm(span=50, adjust=False).mean()
            ema200 = cl.ewm(span=200, adjust=False).mean()
            price_mom_5 = cl.pct_change(5)
            price_mom_20 = cl.pct_change(20)
            cvd_mom_5 = cvd.diff(5)
            cvd_mom_20 = cvd.diff(20)
            flow_z = (rolling_delta - rolling_delta.rolling(50).mean()) / (rolling_delta.rolling(50).std() + 1e-9)
            cvd_high = cvd.rolling(cvd_lookback).max().shift(1)
            cvd_low = cvd.rolling(cvd_lookback).min().shift(1)

            # Uses the exact CVD line shown in chart row 2.
            # Long when CVD flips above zero / green zone.
            # Exit immediately when CVD flips below zero / red zone.
            cvd_zero_cross_up = (cvd_plot > 0) & (cvd_plot.shift(1) <= 0)
            cvd_zero_cross_down = (cvd_plot < 0) & (cvd_plot.shift(1) >= 0)

            cvd_position_basic = make_stateful_position(cvd_cross_up, cvd_cross_down, df_cvd.index)
            cvd_position_zero_flip = make_stateful_position(
                cvd_zero_cross_up,
                (cvd_plot < 0) | cvd_zero_cross_down,
                df_cvd.index
            )
            cvd_position_confirmed = make_stateful_position(
                (cvd > cvd_ma) & (rolling_delta > 0) & (cl > ema20),
                (cvd < cvd_ma) | (rolling_delta < 0) | (cl < ema20),
                df_cvd.index
            )
            cvd_position_breakout = make_stateful_position(
                (cvd > cvd_high) & (cl > ema50) & (price_mom_20 > 0),
                (cvd < cvd_ma) | (cl < ema20) | (cvd < cvd_low),
                df_cvd.index
            )
            cvd_position_smart_money = make_stateful_position(
                (cvd_fast > cvd_slow) & (cvd_mom_5 > 0) & (rolling_delta > 0) & (cl > ema50),
                (cvd_fast < cvd_slow) | (rolling_delta < 0) | (cl < ema50),
                df_cvd.index
            )
            cvd_position_accumulation = make_stateful_position(
                (cvd_mom_20 > 0) & (price_mom_5 > -0.03) & (cl > ema200) & (flow_z > -0.5),
                (cvd_mom_20 < 0) | (cl < ema50) | (flow_z < -1.25),
                df_cvd.index
            )
            cvd_position_risk_on = make_stateful_position(
                (cl > ema20) & (ema20 > ema50) & (cvd > cvd_slow) & (rolling_delta > 0),
                (cl < ema20) | (cvd < cvd_slow) | (rolling_delta < 0),
                df_cvd.index
            )

            cvd_candidates = [
                ("CVD Zero-Line Flip", "Uses the CVD graph row 2 directly: long when CVD crosses above zero/green, cash immediately when CVD goes below zero/red.", cvd_position_zero_flip),
                ("CVD Mean Cross", "Long after CVD crosses above its rolling mean; cash after CVD crosses below.", cvd_position_basic),
                ("CVD Confirmed Trend", "Long only when CVD is above mean, rolling flow is positive, and price is above EMA20.", cvd_position_confirmed),
                ("CVD Breakout + Price Momentum", "Long when CVD breaks its rolling high while price is above EMA50 and 20-bar momentum is positive.", cvd_position_breakout),
                ("Smart-Money CVD Flow", "Long when fast CVD is above slow CVD, CVD momentum is positive, rolling delta is positive, and price is above EMA50.", cvd_position_smart_money),
                ("Accumulation Filter", "Long when 20-bar CVD momentum is positive, price is above EMA200, and flow is not strongly negative.", cvd_position_accumulation),
                ("Risk-On CVD Trend", "Long only when EMA20 > EMA50, price is above EMA20, CVD is above slow CVD, and rolling delta is positive.", cvd_position_risk_on),
            ]

            display_adaptive_strategy_lab("CVD", cl, cvd_candidates, file_prefix="CVD_Adaptive_Strategy")

            # ── Delta Profile (Volume at Price bucket) ─────────────────────
            st.divider()
            st.write("#### 📊 Delta Profile (Buy vs Sell by Price Bucket)")
            n_buckets = st.slider("Price Buckets", 10, 50, 20)

            price_min = float(cl.min())
            price_max = float(cl.max())
            buckets = np.linspace(price_min, price_max, n_buckets + 1)
            bucket_labels = [f"{b:.2f}" for b in buckets[:-1]]

            buy_vol  = pd.cut(cl, bins=buckets, labels=bucket_labels).astype(str)
            buy_profile  = delta.clip(lower=0).groupby(buy_vol).sum()
            sell_profile = delta.clip(upper=0).abs().groupby(buy_vol).sum()

            fig_profile = go.Figure()
            fig_profile.add_trace(go.Bar(
                y=buy_profile.index, x=buy_profile.values,
                orientation='h', name='Buy Volume',
                marker_color='rgba(0,200,100,0.7)'
            ))
            fig_profile.add_trace(go.Bar(
                y=sell_profile.index, x=-sell_profile.values,
                orientation='h', name='Sell Volume',
                marker_color='rgba(255,80,80,0.7)'
            ))
            fig_profile.update_layout(
                barmode='overlay', template="plotly_dark",
                title="Volume Delta Profile (Price × Buy/Sell Pressure)",
                xaxis_title="Delta Volume (Buy=+, Sell=-)",
                yaxis_title="Price Level",
                height=500, hovermode="y unified"
            )
            st.plotly_chart(fig_profile, use_container_width=True)

            if st.session_state.report_gen:
                st.session_state.report_gen.add_plot("CVD Dashboard", fig_cvd)
                if trade_log_rows:
                    st.session_state.report_gen.add_data("CVD Trade Log", tlog_df)

        except Exception as e:
            st.error(f"CVD calculation failed: {e}")
            st.info("Ensure the ticker has OHLCV data (Volume column required for full analysis).")


# ==========================================
# TAB 18: INSTITUTIONAL VWAP
# ==========================================
with tab18:
    st.write("### 📈 Institutional VWAP Suite")
    st.markdown("""
    **VWAP** (Volume Weighted Average Price) is the primary institutional execution benchmark.
    Price above VWAP = bullish bias; below = bearish. **Anchored VWAP** from key dates
    reveals where major players are positioned. Standard deviation bands act as dynamic
    support/resistance used by market makers.
    """)

    if df_main is None:
        st.warning("Please load a ticker to view VWAP analysis.")
    else:
        vwap_col1, vwap_col2, vwap_col3 = st.columns(3)
        with vwap_col1:
            vwap_type = st.selectbox("VWAP Type", [
                "Standard Daily VWAP",
                "Rolling VWAP (N-bar)",
                "Anchored VWAP (from date)",
                "Multi-Timeframe VWAP",
                "VWAP + Volume Profile"
            ])
        with vwap_col2:
            vwap_bands = st.multiselect(
                "Standard Deviation Bands",
                ["1σ", "2σ", "3σ"], default=["1σ", "2σ"]
            )
        with vwap_col3:
            vwap_reset = st.selectbox("VWAP Reset Period", ["Daily", "Weekly", "Monthly", "None (Cumulative)"])

        df_vwap = df_main.copy()

        # Require Volume
        if 'Volume' not in df_vwap.columns or df_vwap['Volume'].sum() == 0:
            st.warning("Volume data unavailable or zero. Using price-weighted approximation.")
            df_vwap['Volume'] = 1.0  # Fallback: equal weight = simple MA

        hi  = df_vwap['High']
        lo  = df_vwap['Low']
        cl  = df_vwap['Close']
        vol = df_vwap['Volume'].replace(0, np.nan).fillna(1)
        tp  = (hi + lo + cl) / 3.0   # Typical Price

        # ── VWAP Calculation Helpers ────────────────────────────────────────
        def compute_rolling_vwap(tp_series, vol_series, window):
            tp_vol = tp_series * vol_series
            vwap_v = tp_vol.rolling(window).sum() / vol_series.rolling(window).sum()
            dev = (tp_series - vwap_v) ** 2
            std_v = np.sqrt((dev * vol_series).rolling(window).sum() / vol_series.rolling(window).sum())
            return vwap_v, std_v

        def compute_anchored_vwap(tp_series, vol_series, anchor_idx):
            tp_vol_cum = (tp_series * vol_series).loc[anchor_idx:].cumsum()
            vol_cum    = vol_series.loc[anchor_idx:].cumsum()
            avwap = tp_vol_cum / vol_cum
            dev = (tp_series.loc[anchor_idx:] - avwap) ** 2
            astd = np.sqrt((dev * vol_series.loc[anchor_idx:]).cumsum() / vol_cum)
            return avwap, astd

        def compute_reset_vwap(tp_series, vol_series, reset='Daily'):
            if reset == 'Daily':
                groups = tp_series.index.date
            elif reset == 'Weekly':
                groups = tp_series.index.to_period('W')
            elif reset == 'Monthly':
                groups = tp_series.index.to_period('M')
            else:
                groups = ['ALL'] * len(tp_series)

            vwap_out = pd.Series(np.nan, index=tp_series.index)
            std_out  = pd.Series(np.nan, index=tp_series.index)

            for grp in pd.Series(groups).unique():
                if reset == 'Daily':
                    mask = pd.Series(tp_series.index.date) == grp
                elif reset == 'Weekly':
                    mask = tp_series.index.to_period('W') == grp
                elif reset == 'Monthly':
                    mask = tp_series.index.to_period('M') == grp
                else:
                    mask = pd.Series([True] * len(tp_series))

                mask = mask.values if hasattr(mask, 'values') else mask
                tp_g  = tp_series.iloc[mask] if hasattr(mask, '__len__') else tp_series
                vol_g = vol_series.iloc[mask] if hasattr(mask, '__len__') else vol_series

                try:
                    idx  = tp_series.index[mask]
                    tpv  = (tp_g.values * vol_g.values).cumsum()
                    vc   = vol_g.values.cumsum()
                    v_   = tpv / vc
                    dev_ = np.sqrt(np.cumsum(((tp_g.values - v_) ** 2) * vol_g.values) / vc)
                    vwap_out.loc[idx] = v_
                    std_out.loc[idx]  = dev_
                except Exception:
                    pass

            return vwap_out, std_out

        # ── Build chart ─────────────────────────────────────────────────────
        try:
            fig_vwap = make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                row_heights=[0.55, 0.25, 0.20],
                subplot_titles=(
                    f"{TICKER} Price + VWAP",
                    "Distance from VWAP (%)",
                    "Volume"
                )
            )

            # Price
            if all(c in df_vwap.columns for c in ['Open', 'High', 'Low', 'Close']):
                fig_vwap.add_trace(go.Candlestick(
                    x=df_vwap.index, open=df_vwap['Open'], high=df_vwap['High'],
                    low=df_vwap['Low'], close=cl,
                    name="Price", increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350', showlegend=False
                ), row=1, col=1)
            else:
                fig_vwap.add_trace(go.Scatter(
                    x=df_vwap.index, y=cl, mode='lines',
                    line=dict(color='gray', width=1.5), name="Price"
                ), row=1, col=1)

            sigma_colors = {'1σ': ('rgba(255,215,0,0.3)', 'rgba(255,215,0,0.3)'),
                            '2σ': ('rgba(255,140,0,0.25)', 'rgba(255,140,0,0.25)'),
                            '3σ': ('rgba(255,50,50,0.2)', 'rgba(255,50,50,0.2)')}
            sigma_vals   = {'1σ': 1, '2σ': 2, '3σ': 3}

            if vwap_type == "Rolling VWAP (N-bar)":
                roll_win = st.slider("Rolling VWAP Window (bars)", 5, 252, 20)
                vwap_s, std_s = compute_rolling_vwap(tp, vol, roll_win)
                vwap_label = f"RVWAP({roll_win})"

            elif vwap_type == "Anchored VWAP (from date)":
                min_d = df_vwap.index[0].date()
                max_d = df_vwap.index[-1].date()
                default_anchor_d = DEFAULT_NONLIVE_START.date()
                if default_anchor_d < min_d:
                    default_anchor_d = min_d
                if default_anchor_d > max_d:
                    default_anchor_d = min_d
                anchor_date = st.date_input("Anchor Date", min_value=min_d,
                                             max_value=max_d,
                                             value=default_anchor_d,
                                             key="avwap_anchor")
                # Find nearest index
                anchor_ts = pd.Timestamp(anchor_date)
                anchor_idx = df_vwap.index[df_vwap.index >= anchor_ts][0] if any(df_vwap.index >= anchor_ts) else df_vwap.index[0]
                vwap_s, std_s = compute_anchored_vwap(tp, vol, anchor_idx)
                vwap_s = vwap_s.reindex(df_vwap.index)
                std_s  = std_s.reindex(df_vwap.index)
                vwap_label = f"AVWAP (from {anchor_date})"

            elif vwap_type == "Multi-Timeframe VWAP":
                vwap_daily,  std_daily  = compute_rolling_vwap(tp, vol, 20)
                vwap_weekly, std_weekly = compute_rolling_vwap(tp, vol, 60)
                vwap_monthly,std_monthly= compute_rolling_vwap(tp, vol, 126)

                for v_, lbl, col_ in [
                    (vwap_daily,   "VWAP 20",  '#ffcc00'),
                    (vwap_weekly,  "VWAP 60",  '#00f2ff'),
                    (vwap_monthly, "VWAP 126", '#ff6b35'),
                ]:
                    fig_vwap.add_trace(go.Scatter(
                        x=df_vwap.index, y=v_, mode='lines',
                        line=dict(color=col_, width=1.5, dash='dot'), name=lbl
                    ), row=1, col=1)

                vwap_s = vwap_daily; std_s = std_daily; vwap_label = "VWAP 20"

            elif vwap_type == "VWAP + Volume Profile":
                vwap_s, std_s = compute_reset_vwap(tp, vol, vwap_reset)
                vwap_label = f"VWAP ({vwap_reset})"

            else:  # Standard Daily VWAP
                vwap_s, std_s = compute_reset_vwap(tp, vol, vwap_reset)
                vwap_label = f"VWAP ({vwap_reset})"

            # Draw main VWAP line
            fig_vwap.add_trace(go.Scatter(
                x=df_vwap.index, y=vwap_s, mode='lines',
                line=dict(color='#ff6b35', width=2.5), name=vwap_label
            ), row=1, col=1)

            # Draw SD bands
            band_traces_added = set()
            for band in vwap_bands:
                n = sigma_vals[band]
                upper = vwap_s + n * std_s
                lower = vwap_s - n * std_s
                c_fill = sigma_colors[band][0]

                if band not in band_traces_added:
                    fig_vwap.add_trace(go.Scatter(
                        x=df_vwap.index, y=upper, mode='lines',
                        line=dict(color=c_fill.replace('0.3', '0.8').replace('0.25', '0.8').replace('0.2', '0.8'), width=1, dash='dot'),
                        name=f"+{band}", showlegend=True
                    ), row=1, col=1)
                    fig_vwap.add_trace(go.Scatter(
                        x=df_vwap.index, y=lower, mode='lines',
                        line=dict(color=c_fill.replace('0.3', '0.8').replace('0.25', '0.8').replace('0.2', '0.8'), width=1, dash='dot'),
                        fill='tonexty', fillcolor=c_fill,
                        name=f"-{band}", showlegend=True
                    ), row=1, col=1)
                    band_traces_added.add(band)

            # Row 2 – Distance from VWAP
            dist_pct = (cl - vwap_s) / vwap_s * 100
            dist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in dist_pct]
            fig_vwap.add_trace(go.Bar(
                x=df_vwap.index, y=dist_pct,
                marker_color=dist_colors, name="Dist from VWAP %"
            ), row=2, col=1)
            fig_vwap.add_hline(y=0, line_dash="dash", line_color="white",
                               opacity=0.4, row=2, col=1)

            # Row 3 – Volume bars
            if 'Volume' in df_vwap.columns:
                vol_colors = ['#26a69a' if c >= o else '#ef5350'
                              for c, o in zip(df_vwap['Close'], df_vwap.get('Open', df_vwap['Close']))]
                fig_vwap.add_trace(go.Bar(
                    x=df_vwap.index, y=df_vwap['Volume'],
                    marker_color=vol_colors, name="Volume", showlegend=False
                ), row=3, col=1)

            fig_vwap.update_layout(
                height=850, hovermode="x unified", template="plotly_dark",
                title=f"Institutional VWAP Suite — {TICKER}",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig_vwap, use_container_width=True)

            # ── VWAP Metrics Dashboard ──────────────────────────────────────
            st.divider()
            st.write("#### 📊 VWAP Institutional Metrics")

            curr_price = float(cl.iloc[-1])
            curr_vwap  = float(vwap_s.iloc[-1]) if not pd.isna(vwap_s.iloc[-1]) else curr_price
            curr_dist  = (curr_price - curr_vwap) / curr_vwap * 100
            curr_std   = float(std_s.iloc[-1]) if not pd.isna(std_s.iloc[-1]) else 0

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Current VWAP", f"{CURRENCY}{curr_vwap:.2f}")
            v2.metric("Price vs VWAP", f"{curr_dist:+.2f}%",
                      delta="Above" if curr_dist > 0 else "Below",
                      delta_color="normal" if curr_dist > 0 else "inverse")
            v3.metric("VWAP Std Dev", f"{CURRENCY}{curr_std:.2f}")
            v4.metric("1σ Upper Band", f"{CURRENCY}{curr_vwap + curr_std:.2f}")

            if curr_dist > 2:
                st.warning(f"⚠️ Price is **{curr_dist:.1f}%** above VWAP — extended, watch for mean reversion to {CURRENCY}{curr_vwap:.2f}.")
            elif curr_dist < -2:
                st.success(f"✅ Price is **{abs(curr_dist):.1f}%** below VWAP — potential institutional buy zone near {CURRENCY}{curr_vwap:.2f}.")
            else:
                st.info(f"📍 Price is hugging VWAP (±{abs(curr_dist):.1f}%) — balanced order flow.")

            # ── VWAP Touch Log (Support/Resistance Tests) ─────────────────
            st.divider()
            st.write("#### 📝 VWAP Touch Log (Institutional S/R Tests)")
            st.caption("Logs every time price crosses VWAP — key institutional re-pricing events.")

            cross_up   = (cl > vwap_s) & (cl.shift(1) <= vwap_s.shift(1))
            cross_down = (cl < vwap_s) & (cl.shift(1) >= vwap_s.shift(1))

            vwap_log = []
            for dt in df_vwap.index[cross_up]:
                vwap_log.append({
                    "Date": dt.date(), "Event": "🟢 Price crossed ABOVE VWAP",
                    "Price": round(float(cl.loc[dt]), 2),
                    "VWAP": round(float(vwap_s.loc[dt]), 2),
                    "Dist %": round(float((cl.loc[dt] - vwap_s.loc[dt]) / vwap_s.loc[dt] * 100), 3),
                    "Volume": int(vol.loc[dt]) if dt in vol.index else 0
                })
            for dt in df_vwap.index[cross_down]:
                vwap_log.append({
                    "Date": dt.date(), "Event": "🔴 Price crossed BELOW VWAP",
                    "Price": round(float(cl.loc[dt]), 2),
                    "VWAP": round(float(vwap_s.loc[dt]), 2),
                    "Dist %": round(float((cl.loc[dt] - vwap_s.loc[dt]) / vwap_s.loc[dt] * 100), 3),
                    "Volume": int(vol.loc[dt]) if dt in vol.index else 0
                })

            if vwap_log:
                vlog_df = pd.DataFrame(vwap_log).sort_values("Date", ascending=False)
                st.dataframe(vlog_df, use_container_width=True)
                csv_vwap = vlog_df.to_csv(index=False)
                st.download_button("📥 Download VWAP Touch Log", csv_vwap,
                                   file_name=f"VWAP_Log_{TICKER}.csv", mime="text/csv")
            else:
                st.info("No VWAP crossovers in the selected date range.")

            # ── VWAP Adaptive Strategy Backtest ────────────────────────────
            st.divider()
            st.write("#### 🧪 VWAP Strategy Backtest")
            st.caption("Adaptive VWAP rules: trend, reclaim, band breakout, and mean-reversion candidates ranked against buy & hold.")

            ema20_v = cl.ewm(span=20, adjust=False).mean()
            ema50_v = cl.ewm(span=50, adjust=False).mean()
            vwap_dist = (cl - vwap_s) / (vwap_s + 1e-9)
            vwap_dist_z = (vwap_dist - vwap_dist.rolling(50).mean()) / (vwap_dist.rolling(50).std() + 1e-9)
            vol_ma = vol.rolling(20).mean()
            high20 = cl.rolling(20).max().shift(1)
            upper_band = (vwap_s + std_s).reindex(df_vwap.index)
            lower_band = (vwap_s - std_s).reindex(df_vwap.index)

            vwap_above = (cl > vwap_s).astype(int).reindex(df_vwap.index).fillna(0)
            vwap_reclaim = make_stateful_position(
                (cl > vwap_s) & (cl.shift(1) <= vwap_s.shift(1)) & (vol > vol_ma),
                (cl < vwap_s) | (cl < ema20_v),
                df_vwap.index
            )
            vwap_trend = make_stateful_position(
                (cl > vwap_s) & (cl > ema20_v) & (ema20_v > ema50_v),
                (cl < vwap_s) | (ema20_v < ema50_v),
                df_vwap.index
            )
            vwap_breakout = make_stateful_position(
                (cl > upper_band) & (cl > high20) & (vol > vol_ma),
                (cl < vwap_s) | (cl < ema20_v),
                df_vwap.index
            )
            vwap_mean_revert = make_stateful_position(
                (vwap_dist_z < -1.25) & (cl > ema50_v),
                (vwap_dist_z > 0.0) | (cl < ema50_v),
                df_vwap.index
            )
            vwap_pullback = make_stateful_position(
                (cl > ema50_v) & (vwap_dist < 0.01) & (vwap_dist > -0.015) & (cl > cl.shift(1)),
                (cl < ema50_v) | (vwap_dist < -0.025),
                df_vwap.index
            )

            vwap_candidates = [
                ("VWAP Above/Below", "Long when price is above VWAP; cash below VWAP.", vwap_above),
                ("VWAP Reclaim + Volume", "Long only after price reclaims VWAP with above-average volume; exit below VWAP or EMA20.", vwap_reclaim),
                ("VWAP Trend Stack", "Long when price is above VWAP and EMA20 > EMA50; exit when VWAP/trend stack breaks.", vwap_trend),
                ("VWAP Band Breakout", "Long when price breaks above the upper VWAP band and 20-bar high with volume confirmation.", vwap_breakout),
                ("VWAP Mean Reversion", "Long when price is stretched below VWAP but still above EMA50; exit when it normalizes.", vwap_mean_revert),
                ("VWAP Pullback Buy", "Long on shallow pullbacks near VWAP while price remains above EMA50.", vwap_pullback),
            ]

            display_adaptive_strategy_lab("VWAP", cl, vwap_candidates, file_prefix="VWAP_Adaptive_Strategy")

            if st.session_state.report_gen:
                st.session_state.report_gen.add_plot("VWAP Suite", fig_vwap)
                if vwap_log:
                    st.session_state.report_gen.add_data("VWAP Touch Log", vlog_df)

        except Exception as e:
            st.error(f"VWAP calculation error: {e}")


# ==========================================
# TAB 19: TIME SERIES ANALYSIS
# ==========================================
with tab19:
    st.write("### 🔬 Institutional Time Series Analysis")
    st.markdown("""
    Deep statistical analysis of price and return dynamics:
    **Stationarity** | **ACF/PACF** | **ARIMA Forecasting** | **Cointegration** |
    **Granger Causality** | **Spectral Analysis** | **Long Memory (ARFIMA)**
    """)

    if df_main is None:
        st.warning("Please load a ticker to run Time Series Analysis.")
    else:
        ts_subtab = st.tabs([
            "📐 Stationarity Tests",
            "📊 ACF / PACF",
            "🔮 ARIMA Forecast",
            "🔗 Cointegration & Causality",
            "🌊 Spectral / Frequency",
            "📉 Long Memory (ARFIMA)",
            "🧪 TS Signal Backtest"
        ])

        ts_series_choice = st.radio(
            "Series to Analyze", ["Log Returns", "Close Price"],
            horizontal=True, key="ts_series_choice"
        )
        ts_data = df_main['Log_Returns'].dropna() if ts_series_choice == "Log Returns" else df_main['Close'].dropna()

        # ── Sub-tab 1: Stationarity ─────────────────────────────────────────
        with ts_subtab[0]:
            st.write("#### 📐 Stationarity & Unit Root Tests")
            st.caption("""
            **ADF**: Null = unit root (non-stationary). Reject → stationary.
            **KPSS**: Null = stationary. Reject → non-stationary.
            **PP**: Phillips-Perron skipped here because your statsmodels build does not include it.
            """)

            try:
                from statsmodels.tsa.stattools import adfuller, kpss

                # ADF Test
                adf_result = adfuller(ts_data, autolag='AIC')
                adf_row = {
                    "Test": "ADF (Augmented Dickey-Fuller)",
                    "Statistic": round(adf_result[0], 4),
                    "p-value": round(adf_result[1], 4),
                    "Lags Used": adf_result[2],
                    "Critical Value 5%": round(adf_result[4]['5%'], 4),
                    "Conclusion": "✅ Stationary" if adf_result[1] < 0.05 else "❌ Non-Stationary"
                }

                # KPSS Test
                kpss_result = kpss(ts_data, regression='c', nlags='auto')
                kpss_row = {
                    "Test": "KPSS",
                    "Statistic": round(kpss_result[0], 4),
                    "p-value": round(kpss_result[1], 4),
                    "Lags Used": kpss_result[2],
                    "Critical Value 5%": round(kpss_result[3]['5%'], 4),
                    "Conclusion": "❌ Non-Stationary" if kpss_result[1] < 0.05 else "✅ Stationary"
                }

                stat_df = pd.DataFrame([adf_row, kpss_row])
                st.dataframe(stat_df.set_index("Test"), use_container_width=True)

                # Interpretation
                adf_stat  = adf_result[1] < 0.05
                kpss_stat = kpss_result[1] >= 0.05

                if adf_stat and kpss_stat:
                    st.success("🟢 **STATIONARY**: Both ADF rejects unit root AND KPSS fails to reject stationarity. Series is mean-reverting and safe for linear models.")
                elif not adf_stat and not kpss_stat:
                    st.error("🔴 **NON-STATIONARY**: ADF fails to reject unit root AND KPSS rejects stationarity. Difference the series before modeling.")
                else:
                    st.warning("🟡 **FRACTIONALLY INTEGRATED**: Tests conflict — likely long-memory process. Use ARFIMA or consider first-differencing.")

                # Rolling ADF p-value (structural stability)
                st.divider()
                st.write("##### Rolling Stationarity (60-bar ADF p-value)")
                roll_adf = []
                win = 60
                for i in range(win, len(ts_data)):
                    try:
                        res_ = adfuller(ts_data.iloc[i-win:i], autolag='AIC')
                        roll_adf.append(res_[1])
                    except Exception:
                        roll_adf.append(np.nan)

                roll_adf_s = pd.Series(roll_adf, index=ts_data.index[win:])
                fig_radf = go.Figure()
                fig_radf.add_trace(go.Scatter(
                    x=roll_adf_s.index, y=roll_adf_s,
                    mode='lines', line=dict(color='cyan', width=1.5),
                    name="Rolling ADF p-value"
                ))
                fig_radf.add_hline(y=0.05, line_dash="dash", line_color="red",
                                   annotation_text="5% threshold (stationary below)")
                fig_radf.update_layout(
                    title="Rolling ADF p-value (Stationarity Over Time)",
                    template="plotly_dark", height=350,
                    yaxis_title="p-value", hovermode="x unified"
                )
                st.plotly_chart(fig_radf, use_container_width=True)

            except Exception as e:
                st.error(f"Stationarity test error: {e}")

        # ── Sub-tab 2: ACF / PACF ───────────────────────────────────────────
        with ts_subtab[1]:
            st.write("#### 📊 Autocorrelation & Partial Autocorrelation")
            st.caption("""
            **ACF**: Measures correlation with lagged values. Tailing off → AR process.
            **PACF**: Removes intermediate lag effects. Sharp cutoff → order of AR.
            """)

            try:
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                from statsmodels.stats.stattools import durbin_watson

                max_lags = st.slider("Max Lags", 10, 100, 40, key="acf_lags")

                fig_acf_pacf, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
                plt.style.use('dark_background')
                plot_acf(ts_data, lags=max_lags, ax=ax1, color='cyan',
                         vlines_kwargs={"colors": "cyan"}, title=f"ACF — {TICKER} ({ts_series_choice})")
                plot_pacf(ts_data, lags=max_lags, ax=ax2, color='orange',
                          vlines_kwargs={"colors": "orange"}, title=f"PACF — {TICKER} ({ts_series_choice})")
                ax1.set_facecolor('#1a1a2e'); ax2.set_facecolor('#1a1a2e')
                fig_acf_pacf.patch.set_facecolor('#1a1a2e')
                plt.tight_layout()
                st.pyplot(fig_acf_pacf)
                plt.close()

                # Ljung-Box Table
                st.write("##### Ljung-Box Test (Serial Correlation by Lag)")
                lb = acorr_ljungbox(ts_data, lags=list(range(1, min(21, max_lags))), return_df=True)
                lb['Conclusion'] = lb['lb_pvalue'].apply(
                    lambda p: "✅ No autocorr" if p > 0.05 else "❌ Autocorr present"
                )
                st.dataframe(lb.style.format({"lb_stat": "{:.3f}", "lb_pvalue": "{:.4f}"}),
                             use_container_width=True)

                # Durbin-Watson
                dw = durbin_watson(ts_data)
                st.metric("Durbin-Watson Statistic", f"{dw:.4f}",
                          help="~2 = no autocorrelation. <1 or >3 = strong autocorrelation.")
                if dw < 1.5:
                    st.warning("Positive autocorrelation detected (DW < 1.5). Trending behaviour.")
                elif dw > 2.5:
                    st.warning("Negative autocorrelation detected (DW > 2.5). Mean-reverting behaviour.")
                else:
                    st.success("No significant autocorrelation (DW ≈ 2).")

            except Exception as e:
                st.error(f"ACF/PACF error: {e}")

        # ── Sub-tab 3: ARIMA Forecast ───────────────────────────────────────
        with ts_subtab[2]:
            st.write("#### 🔮 ARIMA / SARIMA Forecast")

            try:
                from statsmodels.tsa.arima.model import ARIMA

                arima_col1, arima_col2, arima_col3 = st.columns(3)
                with arima_col1:
                    p = st.slider("AR order (p)", 0, 5, 1, key="arima_p")
                    d = st.slider("Integration (d)", 0, 2, 1, key="arima_d")
                    q = st.slider("MA order (q)", 0, 5, 1, key="arima_q")
                with arima_col2:
                    n_forecast = st.slider("Forecast Steps", 5, 60, 21, key="arima_fc")
                    auto_order = st.checkbox("Auto-select order (AIC grid search)", value=False)
                with arima_col3:
                    arima_series = df_main['Close'].dropna() if ts_series_choice == "Close Price" else df_main['Log_Returns'].dropna()

                if auto_order:
                    with st.spinner("Grid searching ARIMA order (p,d,q ∈ {0,1,2})..."):
                        best_aic = float('inf')
                        best_order = (1, 1, 1)
                        for p_ in range(3):
                            for d_ in range(2):
                                for q_ in range(3):
                                    try:
                                        m_ = ARIMA(arima_series, order=(p_, d_, q_)).fit()
                                        if m_.aic < best_aic:
                                            best_aic = m_.aic
                                            best_order = (p_, d_, q_)
                                    except Exception:
                                        pass
                        p, d, q = best_order
                        st.success(f"Best order: ARIMA({p},{d},{q}) — AIC: {best_aic:.2f}")

                with st.spinner(f"Fitting ARIMA({p},{d},{q})..."):
                    arima_model = ARIMA(arima_series, order=(p, d, q)).fit()

                # Metrics
                am1, am2, am3 = st.columns(3)
                am1.metric("AIC",  f"{arima_model.aic:.2f}")
                am2.metric("BIC",  f"{arima_model.bic:.2f}")
                am3.metric("HQIC", f"{arima_model.hqic:.2f}")

                # Forecast
                forecast_res = arima_model.get_forecast(steps=n_forecast)
                fc_mean = forecast_res.predicted_mean
                fc_ci   = forecast_res.conf_int(alpha=0.05)

                last_dt  = arima_series.index[-1]
                fc_dates = [last_dt + timedelta(days=i+1) for i in range(n_forecast)]
                fc_mean.index  = fc_dates
                fc_ci.index    = fc_dates

                # Plot
                hist_n = 120
                fig_arima = go.Figure()
                fig_arima.add_trace(go.Scatter(
                    x=arima_series.index[-hist_n:], y=arima_series.iloc[-hist_n:],
                    mode='lines', line=dict(color='gray', width=1.5), name="Historical"
                ))
                fig_arima.add_trace(go.Scatter(
                    x=fc_dates, y=fc_mean,
                    mode='lines+markers', line=dict(color='#00f2ff', width=2.5, dash='dot'),
                    name="ARIMA Forecast"
                ))
                fig_arima.add_trace(go.Scatter(
                    x=fc_dates + fc_dates[::-1],
                    y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0])[::-1],
                    fill='toself', fillcolor='rgba(0,242,255,0.12)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name="95% CI"
                ))
                fig_arima.update_layout(
                    title=f"ARIMA({p},{d},{q}) — {n_forecast}-Step Forecast",
                    template="plotly_dark", height=450,
                    hovermode="x unified"
                )
                st.plotly_chart(fig_arima, use_container_width=True)

                # Residual diagnostics
                with st.expander("📋 ARIMA Residual Diagnostics"):
                    resid = arima_model.resid.dropna()
                    lb_arima = acorr_ljungbox(resid, lags=[10], return_df=True)
                    arch_arima = het_arch(resid)

                    diag_arima = {
                        "Test": ["Ljung-Box (residuals)", "ARCH-LM (residuals)"],
                        "p-value": [lb_arima['lb_pvalue'].iloc[0], arch_arima[1]],
                        "Pass?": [
                            "✅ Pass" if lb_arima['lb_pvalue'].iloc[0] > 0.05 else "❌ Fail",
                            "✅ Pass" if arch_arima[1] > 0.05 else "❌ Fail"
                        ]
                    }
                    st.table(pd.DataFrame(diag_arima).set_index("Test"))

                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(
                        x=resid.index, y=resid, mode='lines',
                        line=dict(color='orange', width=1), name="Residuals"
                    ))
                    fig_resid.update_layout(title="ARIMA Residuals",
                                            template="plotly_dark", height=300)
                    st.plotly_chart(fig_resid, use_container_width=True)

            except Exception as e:
                st.error(f"ARIMA error: {e}")

        # ── Sub-tab 4: Cointegration & Granger ─────────────────────────────
        with ts_subtab[3]:
            st.write("#### 🔗 Cointegration & Granger Causality")

            pair_ticker_ts = st.text_input("Second Ticker for Cointegration / Causality", "SPY", key="ts_pair")

            if st.button("Run Cointegration & Granger Tests", key="ts_coint_btn"):
                try:
                    from statsmodels.tsa.stattools import coint, grangercausalitytests

                    df_pair_ts = load_data(pair_ticker_ts, start_date, end_date)

                    if df_pair_ts is None:
                        st.error(f"Could not load {pair_ticker_ts}.")
                    else:
                        common_idx = df_main.index.intersection(df_pair_ts.index)
                        s1 = df_main.loc[common_idx, 'Close'].dropna()
                        s2 = df_pair_ts.loc[common_idx, 'Close'].dropna()
                        common_idx = s1.index.intersection(s2.index)
                        s1 = s1.loc[common_idx]
                        s2 = s2.loc[common_idx]

                        # Cointegration
                        st.write("##### Engle-Granger Cointegration Test")
                        coint_t, coint_p, crit_vals = coint(s1, s2)
                        coint_row = pd.DataFrame([{
                            "t-statistic": round(coint_t, 4),
                            "p-value": round(coint_p, 4),
                            "Crit 1%": round(crit_vals[0], 4),
                            "Crit 5%": round(crit_vals[1], 4),
                            "Crit 10%": round(crit_vals[2], 4),
                            "Cointegrated?": "✅ YES (p<0.05)" if coint_p < 0.05 else "❌ NO"
                        }])
                        st.table(coint_row.set_index("Cointegrated?"))

                        if coint_p < 0.05:
                            st.success(f"**{TICKER}** and **{pair_ticker_ts}** are cointegrated — pairs trade opportunity exists!")
                            # Spread
                            spread = s1 - s2
                            spread_z = (spread - spread.mean()) / spread.std()

                            fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                       subplot_titles=("Spread", "Z-Score"))
                            fig_spread.add_trace(go.Scatter(
                                x=spread.index, y=spread, mode='lines',
                                line=dict(color='cyan'), name="Spread"
                            ), row=1, col=1)
                            fig_spread.add_trace(go.Scatter(
                                x=spread_z.index, y=spread_z, mode='lines',
                                line=dict(color='orange'), name="Z-Score"
                            ), row=2, col=1)
                            fig_spread.add_hline(y=2,  line_dash="dash", line_color="red",   row=2, col=1)
                            fig_spread.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
                            fig_spread.add_hline(y=0,  line_dash="dot",  line_color="white", opacity=0.3, row=2, col=1)
                            fig_spread.update_layout(height=500, template="plotly_dark",
                                                     hovermode="x unified",
                                                     title=f"Pairs Spread: {TICKER} - {pair_ticker_ts}")
                            st.plotly_chart(fig_spread, use_container_width=True)
                        else:
                            st.info("No cointegration detected. Pairs trade approach may not be appropriate.")

                        # Granger Causality
                        st.write("##### Granger Causality Test")
                        st.caption(f"Does **{pair_ticker_ts}** Granger-cause **{TICKER}** returns?")
                        r1 = s1.pct_change().dropna()
                        r2 = s2.pct_change().dropna()
                        common_r = r1.index.intersection(r2.index)
                        granger_data = pd.DataFrame({'target': r1.loc[common_r], 'cause': r2.loc[common_r]})

                        max_lag_gc = st.slider("Max Granger Lag", 1, 10, 5, key="gc_lag")
                        gc_results = grangercausalitytests(granger_data[['target', 'cause']], maxlag=max_lag_gc, verbose=False)

                        gc_rows = []
                        for lag, res in gc_results.items():
                            f_test = res[0]['ssr_ftest']
                            gc_rows.append({
                                "Lag": lag,
                                "F-stat": round(f_test[0], 4),
                                "p-value": round(f_test[1], 4),
                                "Granger Cause?": "✅ YES" if f_test[1] < 0.05 else "❌ NO"
                            })
                        gc_df = pd.DataFrame(gc_rows).set_index("Lag")
                        st.dataframe(gc_df.style.applymap(
                            lambda v: 'color: lime' if '✅' in str(v) else 'color: salmon',
                            subset=["Granger Cause?"]
                        ), use_container_width=True)

                except Exception as e:
                    st.error(f"Cointegration/Granger error: {e}")

        # ── Sub-tab 5: Spectral Analysis ────────────────────────────────────
        with ts_subtab[4]:
            st.write("#### 🌊 Spectral & Frequency Domain Analysis")
            st.caption("""
            **Power Spectral Density** decomposes the return series into frequency components.
            Dominant frequencies reveal hidden market cycles (weekly, monthly, quarterly rhythms).
            """)

            try:
                from scipy.signal import periodogram, welch

                spec_method = st.radio("Spectral Estimator", ["Periodogram (raw)", "Welch (smoothed)"], horizontal=True)
                log_scale   = st.checkbox("Log scale Y-axis", value=True)

                if spec_method == "Periodogram (raw)":
                    freqs, psd = periodogram(ts_data.values, fs=1.0)
                else:
                    nperseg = st.slider("Welch Segment Length", 32, 256, 64)
                    freqs, psd = welch(ts_data.values, fs=1.0, nperseg=nperseg)

                # Convert frequency to period (days)
                with np.errstate(divide='ignore'):
                    periods = 1.0 / freqs[1:]   # skip DC component
                psd_plot = psd[1:]

                fig_spec = go.Figure()
                fig_spec.add_trace(go.Scatter(
                    x=periods, y=np.log10(psd_plot) if log_scale else psd_plot,
                    mode='lines', line=dict(color='#a855f7', width=1.5),
                    name="Power Spectral Density"
                ))

                # Mark key market cycles
                for cycle_period, label in [(5, "Weekly"), (21, "Monthly"), (63, "Quarterly"), (126, "Semi-annual"), (252, "Annual")]:
                    fig_spec.add_vline(
                        x=cycle_period, line_dash="dash", line_color="orange",
                        annotation_text=label, annotation_position="top right"
                    )

                fig_spec.update_layout(
                    title="Power Spectral Density (Market Cycle Detection)",
                    xaxis_title="Period (Trading Days)",
                    yaxis_title="Log10(PSD)" if log_scale else "PSD",
                    template="plotly_dark", height=450,
                    xaxis=dict(range=[2, 252])
                )
                st.plotly_chart(fig_spec, use_container_width=True)

                # Top cycles
                top_n_cycles = 5
                top_idx  = np.argsort(psd_plot)[::-1][:top_n_cycles]
                top_periods = periods[top_idx]
                top_power   = psd_plot[top_idx]
                cycle_df = pd.DataFrame({
                    "Dominant Period (days)": np.round(top_periods, 1),
                    "Approximate Cycle": [
                        "~Weekly" if 4 <= p <= 6 else
                        "~Monthly" if 18 <= p <= 24 else
                        "~Quarterly" if 55 <= p <= 70 else
                        "~Semi-annual" if 110 <= p <= 140 else
                        "~Annual" if 220 <= p <= 280 else
                        f"Custom ({p:.0f}d)" for p in np.round(top_periods, 1)
                    ],
                    "Relative Power": np.round(top_power / top_power.max() * 100, 1)
                })
                st.write("##### Top Dominant Cycles")
                st.table(cycle_df)

            except Exception as e:
                st.error(f"Spectral analysis error: {e}")

        # ── Sub-tab 6: Long Memory (ARFIMA) ────────────────────────────────
        with ts_subtab[5]:
            st.write("#### 📉 Long Memory & Fractional Integration (ARFIMA)")
            st.caption("""
            **ARFIMA(p,d,q)** models long-range dependence. The fractional differencing parameter **d**
            captures how strongly past shocks persist:
            - **d = 0**: Short memory (standard ARMA)
            - **0 < d < 0.5**: Long memory, stationary
            - **d ≥ 0.5**: Non-stationary long memory
            """)

            try:
                # Estimate Hurst exponent as proxy for d = H - 0.5
                from scipy.stats import linregress

                lm_data = ts_data.values

                def estimate_hurst_rs(series, min_n=10):
                    """R/S analysis for Hurst exponent estimation."""
                    n = len(series)
                    rs_vals, ns_vals = [], []
                    for size in [n // k for k in range(2, min(20, n // min_n + 1))]:
                        if size < min_n:
                            continue
                        chunks = [series[i:i+size] for i in range(0, n - size + 1, size)]
                        rs_chunk = []
                        for chunk in chunks:
                            mean_c = np.mean(chunk)
                            dev = np.cumsum(chunk - mean_c)
                            r_s = (np.max(dev) - np.min(dev)) / (np.std(chunk) + 1e-10)
                            rs_chunk.append(r_s)
                        rs_vals.append(np.mean(rs_chunk))
                        ns_vals.append(size)
                    if len(rs_vals) < 3:
                        return 0.5
                    slope, _, _, _, _ = linregress(np.log(ns_vals), np.log(rs_vals))
                    return slope

                lm_col1, lm_col2 = st.columns(2)

                with lm_col1:
                    with st.spinner("Estimating Hurst exponent (R/S Analysis)..."):
                        H_rs = estimate_hurst_rs(lm_data)
                        d_est = H_rs - 0.5

                    st.metric("Hurst Exponent (R/S)", f"{H_rs:.4f}")
                    st.metric("Fractional d (H - 0.5)", f"{d_est:+.4f}")

                    if H_rs > 0.55:
                        st.success(f"**Long Memory / Trending** (H={H_rs:.3f} > 0.5). Momentum strategies favored.")
                    elif H_rs < 0.45:
                        st.warning(f"**Anti-Persistent / Mean-Reverting** (H={H_rs:.3f} < 0.5). Mean reversion strategies favored.")
                    else:
                        st.info(f"**Near Random Walk** (H={H_rs:.3f} ≈ 0.5). Standard ARMA adequate.")

                with lm_col2:
                    # Variance ratio test (Lo-MacKinlay style)
                    st.write("##### Variance Ratio Test (Random Walk)")
                    vr_lags = [2, 4, 8, 16, 32]
                    vr_rows = []
                    var1 = np.var(np.diff(lm_data))
                    for lag in vr_lags:
                        diff_k = lm_data[lag:] - lm_data[:-lag]
                        var_k = np.var(diff_k) / lag
                        vr = var_k / (var1 + 1e-10)
                        n = len(lm_data)
                        # Asymptotic z-stat under RW null
                        z = (vr - 1) * np.sqrt(n * lag / (2 * (2 * lag - 1) * (lag - 1) / 3))
                        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
                        vr_rows.append({
                            "Lag q": lag,
                            "Variance Ratio": round(vr, 4),
                            "Z-stat": round(z, 3),
                            "p-value": round(p_val, 4),
                            "Random Walk?": "✅ Fail to reject" if p_val > 0.05 else "❌ Reject RW"
                        })
                    vr_df = pd.DataFrame(vr_rows).set_index("Lag q")
                    st.dataframe(vr_df, use_container_width=True)

                # Log RS plot
                st.write("##### R/S Log-Log Plot (Hurst Estimation)")
                n = len(lm_data)
                rs_plot_vals, ns_plot_vals = [], []
                for size in [n // k for k in range(2, min(30, n // 8 + 1))]:
                    if size < 8:
                        continue
                    chunks = [lm_data[i:i+size] for i in range(0, n - size + 1, size)]
                    rs_c = []
                    for chunk in chunks:
                        mean_c = np.mean(chunk)
                        dev = np.cumsum(chunk - mean_c)
                        rs_c.append((np.max(dev) - np.min(dev)) / (np.std(chunk) + 1e-10))
                    rs_plot_vals.append(np.mean(rs_c))
                    ns_plot_vals.append(size)

                if len(ns_plot_vals) >= 3:
                    log_ns = np.log(ns_plot_vals)
                    log_rs = np.log(rs_plot_vals)
                    slope, intercept, _, _, _ = linregress(log_ns, log_rs)
                    trend_line = slope * log_ns + intercept

                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(
                        x=log_ns, y=log_rs, mode='markers',
                        marker=dict(color='cyan', size=8), name="log(R/S)"
                    ))
                    fig_rs.add_trace(go.Scatter(
                        x=log_ns, y=trend_line, mode='lines',
                        line=dict(color='orange', dash='dash'),
                        name=f"Fit (H={slope:.3f})"
                    ))
                    # Benchmark: H=0.5 (Random Walk)
                    rw_line = 0.5 * log_ns + intercept
                    fig_rs.add_trace(go.Scatter(
                        x=log_ns, y=rw_line, mode='lines',
                        line=dict(color='white', dash='dot', width=1),
                        name="H=0.5 (Random Walk)"
                    ))
                    fig_rs.update_layout(
                        title="R/S Log-Log Analysis (Hurst Exponent)",
                        xaxis_title="log(n)", yaxis_title="log(R/S)",
                        template="plotly_dark", height=400
                    )
                    st.plotly_chart(fig_rs, use_container_width=True)

            except Exception as e:
                st.error(f"Long memory analysis error: {e}")



        # ── Sub-tab 7: Time Series Signal Backtest ─────────────────────────
        with ts_subtab[6]:
            st.write("#### 🧪 Time Series Adaptive Backtest")
            st.caption("Instead of one fixed signal, this tests several time-series rules and ranks them versus buy & hold.")

            try:
                ts_bt_col1, ts_bt_col2 = st.columns(2)
                with ts_bt_col1:
                    ts_win = st.slider("Rolling Window", 20, 150, 60, key="ts_bt_win")
                with ts_bt_col2:
                    ts_initial_cap = st.number_input("Initial Capital", 1000, 1000000, 10000, key="ts_bt_cap")

                ts_prices = df_main['Close'].dropna()
                ts_rets = np.log(ts_prices / ts_prices.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
                ts_ema20 = ts_prices.ewm(span=20, adjust=False).mean()
                ts_ema50 = ts_prices.ewm(span=50, adjust=False).mean()
                ts_ema200 = ts_prices.ewm(span=200, adjust=False).mean()
                ts_mom5 = ts_prices.pct_change(5)
                ts_mom20 = ts_prices.pct_change(20)
                rolling_ac = ts_rets.rolling(ts_win).corr(ts_rets.shift(1)).reindex(ts_prices.index)
                ma = ts_prices.rolling(ts_win).mean()
                sd = ts_prices.rolling(ts_win).std()
                z = (ts_prices - ma) / (sd + 1e-9)
                h = rolling_hurst(ts_prices, window=ts_win, max_lag=min(20, max(5, ts_win // 3))).reindex(ts_prices.index)
                vol20 = ts_rets.rolling(20).std().reindex(ts_prices.index)
                vol100 = ts_rets.rolling(100).std().reindex(ts_prices.index)

                ts_ar_mom = ((rolling_ac > 0) & (ts_mom5 > 0)).astype(int).reindex(ts_prices.index).fillna(0)
                ts_mean_rev = make_stateful_position(z < -1.0, z > 0.0, ts_prices.index)
                ts_hurst_trend = ((h > 0.55) & (ts_prices > ts_ema50)).astype(int).reindex(ts_prices.index).fillna(0)
                ts_trend_stack = make_stateful_position(
                    (ts_prices > ts_ema20) & (ts_ema20 > ts_ema50) & (ts_mom20 > 0),
                    (ts_prices < ts_ema20) | (ts_ema20 < ts_ema50),
                    ts_prices.index
                )
                ts_low_vol_momentum = make_stateful_position(
                    (ts_mom20 > 0) & (ts_prices > ts_ema50) & (vol20 < vol100),
                    (ts_mom20 < 0) | (ts_prices < ts_ema50) | (vol20 > vol100 * 1.5),
                    ts_prices.index
                )
                ts_regime_combo = make_stateful_position(
                    (h > 0.50) & (ts_prices > ts_ema200) & (ts_mom20 > 0),
                    (h < 0.45) | (ts_prices < ts_ema50) | (ts_mom20 < -0.05),
                    ts_prices.index
                )

                ts_candidates = [
                    ("AR(1) Momentum", "Long when serial correlation is positive and 5-bar momentum is positive.", ts_ar_mom),
                    ("Mean Reversion Z-Score", "Buy when price is more than 1σ below rolling mean; exit near mean.", ts_mean_rev),
                    ("Hurst Trend Regime", "Long only when Hurst suggests trend behavior and price is above EMA50.", ts_hurst_trend),
                    ("EMA Trend Stack", "Long when price > EMA20 > EMA50 with positive 20-bar momentum.", ts_trend_stack),
                    ("Low-Vol Momentum", "Long when momentum is positive, price is above EMA50, and realized volatility is calm.", ts_low_vol_momentum),
                    ("Regime Combo", "Long when Hurst is above random, price is above EMA200, and 20-bar momentum is positive.", ts_regime_combo),
                ]

                display_adaptive_strategy_lab(
                    "Time Series",
                    ts_prices,
                    ts_candidates,
                    initial_capital=float(ts_initial_cap),
                    file_prefix="TimeSeries_Adaptive_Strategy"
                )

            except Exception as e:
                st.error(f"Time Series backtest error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Enhanced with: Institutional CVD | VWAP Suite | Time Series Analysis | Quant Thesis v2.0")

# Footer
st.markdown("---")
st.caption("Generated via Gemini 2.0 Flash | Robust Financial Thesis Implementation")
