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
import io
import time
from concurrent.futures import ThreadPoolExecutor

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
    if len(dates) == 0:
        return
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)
    span_days = (dates[-1] - dates[0]).days
    if span_days < 90:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='center', fontsize=8)


class Calibrator:
    @staticmethod
    def calibrate_heston(returns):
        dt = 1/252
        am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1, dist='Normal')
        res = am.fit(disp='off')
        conditional_vol = res.conditional_volatility / 100
        variance = conditional_vol**2
        variance = variance.values if hasattr(variance, 'values') else variance
        v_curr = variance[:-1]
        v_next = variance[1:]
        Y = (v_next - v_curr) / dt
        X = v_curr
        A = np.vstack([X, np.ones(len(X))]).T
        beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        kappa = -beta
        theta = alpha / kappa if kappa != 0 else np.mean(variance)
        kappa = max(kappa, 0.1)
        theta = max(theta, 0.01)
        residuals = Y - (alpha + beta * X)
        xi = np.std(residuals) * np.sqrt(dt) / np.mean(np.sqrt(v_curr))
        xi = max(xi, 0.1)
        rho = np.corrcoef(returns[1:], np.diff(variance))[0, 1]
        mu = np.mean(returns) / dt + 0.5 * np.mean(variance)
        return {
            'mu': mu, 'kappa': kappa, 'theta': theta,
            'xi': xi, 'rho': rho, 'v0': variance[-1], 'S0': 100.0
        }


class KalmanFilterReg:
    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta
        self.R = R
        self.trans_cov = delta / (1 - delta) * np.eye(2)

    def run_filter(self, y, x):
        n = len(y)
        state_mean = np.zeros((n, 2))
        state_cov = np.zeros((n, 2, 2))
        state_mean[0] = [0, 1]
        state_cov[0] = np.eye(2)
        for t in range(1, n):
            pred_state = state_mean[t-1]
            pred_cov = state_cov[t-1] + self.trans_cov
            obs_mat = np.array([[1.0, x[t]]])
            y_pred = np.dot(obs_mat, pred_state)
            error = y[t] - y_pred
            S = np.dot(np.dot(obs_mat, pred_cov), obs_mat.T) + self.R
            K = np.dot(pred_cov, obs_mat.T) / S
            state_mean[t] = pred_state + (K.flatten() * error)
            state_cov[t] = pred_cov - np.dot(np.dot(K, obs_mat), pred_cov)
        return state_mean, state_cov


class KalmanFilterTrend:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.Q = process_noise
        self.R = measurement_noise

    def filter(self, data):
        n = len(data)
        estimates = np.zeros(n)
        covariances = np.zeros(n)
        init_window = min(10, n // 10)
        x = np.mean(data[:init_window])
        P = np.var(data[:init_window]) if init_window > 1 else 1.0
        for t in range(n):
            x_pred = x
            P_pred = P + self.Q
            K = P_pred / (P_pred + self.R)
            x = x_pred + K * (data[t] - x_pred)
            P = (1 - K) * P_pred
            estimates[t] = x
            covariances[t] = P
        return estimates, covariances

    def smooth(self, data):
        n = len(data)
        filtered_means, filtered_covs = self.filter(data)
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
    dt = T/steps
    prices = np.zeros((steps + 1, paths))
    vols = np.zeros((steps + 1, paths))
    prices[0] = S0
    vols[0] = v0
    for t in range(1, steps + 1):
        Z1 = np.random.normal(size=paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=paths)
        v_prev = vols[t-1]
        dv = kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.abs(v_prev)) * np.sqrt(dt) * Z2
        v_curr = np.abs(v_prev + dv)
        vols[t] = v_curr
        dS = r * prices[t-1] * dt + np.sqrt(v_curr) * prices[t-1] * np.sqrt(dt) * Z1
        prices[t] = prices[t-1] + dS
    return prices, vols


def merton_jump_diffusion(S0, T, r, sigma, lam, mu_j, sigma_j, steps, paths):
    dt = T/steps
    prices = np.zeros((steps + 1, paths))
    prices[0] = S0
    drift = r - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    for t in range(1, steps + 1):
        z = np.random.normal(size=paths)
        N = np.random.poisson(lam * dt, size=paths)
        J = np.random.normal(mu_j, sigma_j, size=paths) * N
        prices[t] = prices[t-1] * np.exp(drift * dt + sigma * np.sqrt(dt) * z + J)
    return prices


class RealizedVolatility:
    @staticmethod
    def realized_variance(returns):
        return np.sum(returns**2)

    @staticmethod
    def bipower_variation(returns):
        abs_rets = np.abs(returns)
        if len(returns) < 2:
            return 0.0
        return (np.pi / 2) * np.sum(abs_rets[1:] * abs_rets[:-1])

    @staticmethod
    def jump_component(returns):
        rv = RealizedVolatility.realized_variance(returns)
        bv = RealizedVolatility.bipower_variation(returns)
        jump_var = max(rv - bv, 0)
        jump_ratio = jump_var / rv if rv > 0 else 0.0
        n = len(returns)
        if n < 10:
            return {'jump_ratio': 0.0, 'p_value': 1.0, 'z_score': 0.0}
        z_score = (jump_ratio - 0.05) * np.sqrt(n/2)
        p_value = 1 - stats.norm.cdf(z_score)
        return {'jump_ratio': jump_ratio, 'p_value': p_value, 'z_score': z_score}


class HawkesVolatility:
    def __init__(self):
        self.mu = 0.5
        self.alpha = 0.5
        self.beta = 2.0
        self.metrics = {}

    def fit(self, returns):
        vol_proxy = np.abs(returns)
        if len(vol_proxy) < 20:
            return self
        threshold = np.percentile(vol_proxy, 90)
        events = np.where(vol_proxy > threshold)[0]
        if len(events) < 5:
            return self

        def neg_log_likelihood(params):
            mu_p, alpha_p, beta_p = params
            if mu_p <= 0 or alpha_p < 0 or beta_p <= alpha_p:
                return 1e9
            t = events
            n = len(t)
            T_end = len(returns)
            R = np.zeros(n)
            for i in range(1, n):
                dt = t[i] - t[i-1]
                R[i] = np.exp(-beta_p * dt) * (1 + R[i-1])
            intensities = mu_p + alpha_p * R
            if np.any(intensities <= 0):
                return 1e9
            term1 = np.sum(np.log(intensities))
            term2 = mu_p * T_end + (alpha_p / beta_p) * np.sum(1 - np.exp(-beta_p * (T_end - t)))
            return -(term1 - term2)

        try:
            res = minimize(neg_log_likelihood, [0.1, 0.2, 1.0],
                           bounds=[(1e-4, 2.0), (1e-4, 5.0), (0.1, 10.0)], method='L-BFGS-B')
            self.mu, self.alpha, self.beta = res.x
        except:
            pass
        return self

    def branching_ratio(self):
        if self.beta == 0:
            return 0.0
        return self.alpha / self.beta

    def half_life(self):
        if self.beta == 0:
            return 0.0
        return np.log(2) / self.beta


class AdvancedRegimeDetector:
    def __init__(self, log_returns):
        self.data = log_returns.values.reshape(-1, 1) if hasattr(log_returns, 'values') else log_returns
        self.dates = log_returns.index if hasattr(log_returns, 'index') else np.arange(len(log_returns))
        self.metrics = {}
        self.regimes = {}
        self.regime_characteristics = []

    def fit_all(self, n_states=3):
        if SKLEARN_AVAILABLE:
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=42)
            model.fit(self.data)
            hidden_states = model.predict(self.data)
            probs = model.predict_proba(self.data)
            state_vars = []
            for i in range(n_states):
                mask = (hidden_states == i)
                if np.sum(mask) > 0:
                    state_vars.append(np.std(self.data[mask]))
                else:
                    state_vars.append(0)
            sorted_idx = np.argsort(state_vars)
            map_dict = {old: new for new, old in enumerate(sorted_idx)}
            sorted_states = np.vectorize(map_dict.get)(hidden_states)
            sorted_probs = probs[:, sorted_idx]
            self.regimes['hmm_states'] = sorted_states
            self.regimes['hmm_probs'] = sorted_probs
            self.metrics['hmm_aic'] = model.aic(self.data)
            self._calculate_characteristics(sorted_states)
        else:
            self.regimes['hmm_probs'] = np.zeros((len(self.data), n_states))
            self.metrics['hmm_aic'] = 0
        self.regimes['changepoint_probs'] = self._bayesian_changepoint_proxy(self.data.flatten())

    def _bayesian_changepoint_proxy(self, data):
        vol = pd.Series(data).rolling(window=22).std().fillna(method='bfill')
        vol_change = vol.diff().abs()
        mean_change = vol_change.rolling(252, min_periods=20).mean()
        std_change = vol_change.rolling(252, min_periods=20).std()
        z = (vol_change - mean_change) / (std_change + 1e-8)
        probs = 1 / (1 + np.exp(-(z - 2.0)))
        return probs.fillna(0).values

    def _calculate_characteristics(self, states):
        df = pd.DataFrame(self.data, columns=['ret'])
        df['state'] = states
        stats_df = df.groupby('state')['ret'].agg(['mean', 'std', 'count'])
        self.regime_characteristics = []
        labels = ['Bull/Calm', 'Normal/Transition', 'Bear/Crisis']
        for i in range(len(stats_df)):
            if i >= len(labels):
                cn = f"State {i}"
            else:
                cn = labels[i]
            s = stats_df.iloc[i]
            self.regime_characteristics.append({
                'label': cn,
                'mean_return': s['mean'] * 252,
                'volatility': s['std'] * np.sqrt(252),
                'frequency': s['count'] / len(df),
                'avg_duration': 0.0,
                'max_drawdown': 0.0
            })

    def get_trading_signal(self):
        if 'hmm_probs' not in self.regimes:
            return "N/A", {'label': 'No Model', 'confidence': 0.0}
        probs = self.regimes['hmm_probs'][-1]
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
    def __init__(self, prices, log_returns):
        self.prices = prices if isinstance(prices, pd.Series) else pd.Series(prices)
        self.returns = log_returns if isinstance(log_returns, pd.Series) else pd.Series(log_returns)
        self.features = None
        self.regimes = {}
        self.metrics = {}
        self.state_labels = {}

    def _prepare_features(self):
        f1 = self.returns.rolling(window=5).mean().fillna(0)
        vol = self.returns.rolling(window=20).std().fillna(method='bfill')
        v_mean = vol.rolling(252, min_periods=20).mean()
        v_std = vol.rolling(252, min_periods=20).std()
        f2 = (vol - v_mean) / (v_std + 1e-9)
        f2 = f2.fillna(0)
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
            model = GaussianMixture(n_components=n_states, covariance_type='full', random_state=123, max_iter=200)
            model.fit(X_scaled)
            states = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)
            state_stats = []
            for i in range(n_states):
                mask = (states == i)
                if np.sum(mask) > 0:
                    m_ret = np.mean(self.features[mask, 0])
                    m_vol = np.mean(self.features[mask, 1])
                    state_stats.append({'id': i, 'ret': m_ret, 'vol': m_vol})
                else:
                    state_stats.append({'id': i, 'ret': -999, 'vol': 999})
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
            self.metrics['aic'] = model.aic(X_scaled)
            self.metrics['bic'] = model.bic(X_scaled)
            self.metrics['n_states'] = n_states
        else:
            self.regimes['states'] = np.zeros(len(X))
            self.regimes['probs'] = np.ones((len(X), 1))

    def fit_optimized(self, state_choices=[2, 3, 4]):
        best_bic = float('inf')
        best_n = 4
        for n in state_choices:
            try:
                temp_model = ProRegimeDetector(self.prices, self.returns)
                temp_model.fit(n_states=n)
                if temp_model.metrics.get('bic', float('inf')) < best_bic:
                    best_bic = temp_model.metrics['bic']
                    best_n = n
            except:
                continue
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
        common_idx = self.r_asset.index.intersection(self.r_bench.index)
        y = self.r_asset.loc[common_idx] - self.rf_daily
        x = self.r_bench.loc[common_idx] - self.rf_daily
        df = pd.DataFrame({'asset_ex': y, 'mkt_ex': x}, index=common_idx)
        beta_arr = np.full(len(df), np.nan)
        alpha_arr = np.full(len(df), np.nan)
        for i in range(window, len(df)):
            window_slice = df.iloc[i-window:i]
            y_win = window_slice['asset_ex']
            x_win = sm.add_constant(window_slice['mkt_ex'])
            try:
                model = sm.OLS(y_win, x_win).fit(cov_type='HAC', cov_kwds={'maxlags': 1})
                alpha_arr[i] = model.params.get('const', np.nan)
                beta_arr[i] = model.params.get('mkt_ex', np.nan)
            except:
                pass
        df['Beta'] = beta_arr
        df['Alpha_Daily'] = alpha_arr
        rolling_mkt_ret_ann = df['mkt_ex'].rolling(window).mean() * 252
        df['SML_Exp_Return'] = self.rf_annual + (df['Beta'] * rolling_mkt_ret_ann)
        df['Actual_Return_Ann'] = (df['asset_ex'].rolling(window).mean() * 252) + self.rf_annual
        df['Mispricing_Spread'] = df['Actual_Return_Ann'] - df['SML_Exp_Return']
        return df.dropna()


class MADTrendModes:
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
        return series.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def alma(series, length, offset=0.85, sigma=6):
        m = offset * (length - 1)
        s = length / sigma
        weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * s * s))
        weights /= weights.sum()
        return series.rolling(window=length).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def lsma(series, length):
        def linreg_end(y):
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            return slope * (len(y) - 1) + intercept
        return series.rolling(window=length).apply(linreg_end, raw=True)

    @staticmethod
    def ehlers_supersmoother(series, period):
        """
        Ehlers SuperSmoother Filter — Original 2-pole version.
        From John Ehlers' 'Cybernetic Analysis for Stocks and Futures' (2004).

        Coefficients:
          a1 = exp(-1.414 * pi / period)
          b1 = 2 * a1 * cos(1.414 * 180 / period)  [degrees]
          c2 = b1
          c3 = -a1^2
          c1 = 1 - c2 - c3

        Recursive formula:
          SS[i] = c1 * (src[i] + src[i-1]) / 2 + c2 * SS[i-1] + c3 * SS[i-2]
        """
        import math
        period = max(period, 2)  # guard against period < 2
        a1 = math.exp(-1.414 * math.pi / period)
        b1 = 2.0 * a1 * math.cos(math.radians(1.414 * 180.0 / period))
        c2 = b1
        c3 = -(a1 ** 2)
        c1 = 1.0 - c2 - c3

        vals = series.values.astype(float)
        n = len(vals)
        ss = np.zeros(n)

        for i in range(n):
            src_avg = (vals[i] + vals[i - 1]) / 2.0 if i >= 1 else vals[i]
            ss_prev1 = ss[i - 1] if i >= 1 else src_avg
            ss_prev2 = ss[i - 2] if i >= 2 else src_avg
            ss[i] = c1 * src_avg + c2 * ss_prev1 + c3 * ss_prev2

        return pd.Series(ss, index=series.index)

    @staticmethod
    def ehlers_simple_decycler(series, hp_period, upper_thresh=0.5, lower_thresh=0.5):
        """
        Ehlers Simple Decycler — from John Ehlers' 'Cycle Analytics for Traders' (2013).

        Concept:
          The Decycler removes the dominant cycle component from price by
          subtracting a High-Pass (HP) filter output from price itself:
              Decycler[i] = Price[i] - HP[i]

          This leaves only the trend (low-frequency) component — like the
          orange line in TradingView's EhlersSimpleDecycler indicator.

        High-Pass Filter (1-pole):
          alpha = (cos(2*pi/hp_period) + sin(2*pi/hp_period) - 1) / cos(2*pi/hp_period)
          HP[i] = (1 - alpha/2)^2 * (Price[i] - 2*Price[i-1] + Price[i-2])
                + 2*(1-alpha)*HP[i-1] - (1-alpha)^2 * HP[i-2]

        Signals (mirroring TradingView indicator):
          BUY  when Decycler crosses ABOVE  (Price * (1 - upper_thresh/100))
          SELL when Decycler crosses BELOW  (Price * (1 - lower_thresh/100))

          In practice with thresh=0: BUY when price crosses above Decycler line,
          SELL when price crosses below Decycler line.
        """
        import math
        vals = series.values.astype(float)
        n = len(vals)

        # ── High-Pass filter coefficients ──────────────────────────────────
        hp_period = max(hp_period, 2)
        angle = 2.0 * math.pi / hp_period
        cos_a = math.cos(math.radians(360.0 / hp_period))
        sin_a = math.sin(math.radians(360.0 / hp_period))
        alpha = (cos_a + sin_a - 1.0) / cos_a   # Ehlers 1-pole HP alpha

        hp  = np.zeros(n)
        dec = np.zeros(n)

        for i in range(n):
            p0 = vals[i]
            p1 = vals[i-1] if i >= 1 else p0
            p2 = vals[i-2] if i >= 2 else p0
            hp1 = hp[i-1] if i >= 1 else 0.0
            hp2 = hp[i-2] if i >= 2 else 0.0

            hp[i] = ((1.0 - alpha / 2.0) ** 2) * (p0 - 2.0 * p1 + p2) \
                    + 2.0 * (1.0 - alpha) * hp1 \
                    - ((1.0 - alpha) ** 2) * hp2

            dec[i] = p0 - hp[i]   # Decycler = Price - HighPass

        decycler = pd.Series(dec, index=series.index)
        hp_series = pd.Series(hp, index=series.index)
        return decycler, hp_series

    @staticmethod
    def ma_switch(series, length, avg_type):
        if avg_type == "SMA": return MADTrendModes.sma(series, length)
        if avg_type == "EMA": return MADTrendModes.ema(series, length)
        if avg_type == "WMA": return MADTrendModes.wma(series, length)
        if avg_type == "HMA": return MADTrendModes.hma(series, length)
        if avg_type == "RMA": return MADTrendModes.rma(series, length)
        if avg_type == "ALMA": return MADTrendModes.alma(series, length)
        if avg_type == "LSMA": return MADTrendModes.lsma(series, length)
        return MADTrendModes.sma(series, length)

    @staticmethod
    def calculate_mad(series, benchmark, length):
        from numpy.lib.stride_tricks import sliding_window_view
        vals = series.values
        bench_vals = benchmark.values
        if len(vals) < length:
            return pd.Series(np.nan, index=series.index)
        windows = sliding_window_view(vals, length)
        diffs = np.abs(windows - bench_vals[length-1:, np.newaxis])
        res_vals = np.mean(diffs, axis=1)
        res = np.full(len(series), np.nan)
        res[length-1:] = res_vals
        return pd.Series(res, index=series.index)

    @staticmethod
    def system_score(series, a, b):
        total = pd.Series(0.0, index=series.index)
        for i in range(a, b + 1):
            shifted = series.shift(i)
            total += np.sign(series - shifted).fillna(0)
        return total

    @staticmethod
    def get_signals(df, params):
        src = df['Close']
        mode = params.get('signal_mode', 'Bollinger Bands')
        bb_ma_type = params.get('bb_ma_type', 'EMA')
        bb_len = params.get('bb_len', 25)
        bb_mult_p = params.get('bb_mult_p', 1.4)
        bb_mult_n = params.get('bb_mult_n', 1.0)
        fl_ma_type = params.get('fl_ma_type', 'ALMA')
        fl_len = params.get('fl_len', 10)
        fl_a = params.get('fl_a', 10)
        fl_b = params.get('fl_b', 60)
        fl_thresh_l = params.get('fl_thresh_l', 23)
        fl_thresh_s = params.get('fl_thresh_s', 3)
        c_thresh_l = params.get('c_thresh_l', 0.0)
        c_thresh_s = params.get('c_thresh_s', 0.0)

        avg_bb = MADTrendModes.ma_switch(src, bb_len, bb_ma_type)
        mad_bb = MADTrendModes.calculate_mad(src, avg_bb, bb_len)
        bb_up = avg_bb + (mad_bb * bb_mult_p)
        bb_dn = avg_bb - (mad_bb * bb_mult_n)

        avg_fl = MADTrendModes.ma_switch(src, fl_len, fl_ma_type)
        mad_fl_val = MADTrendModes.calculate_mad(src, avg_fl, fl_len)
        num = MADTrendModes.ma_switch(src * mad_fl_val, fl_len, fl_ma_type)
        den = MADTrendModes.ma_switch(mad_fl_val, fl_len, fl_ma_type)
        mad_w_src = num / den
        sys_score = MADTrendModes.system_score(mad_w_src, fl_a, fl_b)

        bb_long = (src > bb_up) & (src.shift(1) <= bb_up.shift(1))
        bb_short = (src < bb_dn) & (src.shift(1) >= bb_dn.shift(1))

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

        if mode == "Bollinger Bands":
            final_score = bb_score
        elif mode == "For Loop":
            final_score = fl_score
        else:
            final_score = combined_score

        return (final_score == 1).astype(int)


class BacktestEngine:
    @staticmethod
    def run_strategy(prices, signals, initial_capital=10000.0, trailing_stop_pct=0.0):
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]
        signals = signals.loc[common_idx]
        returns = prices.pct_change().fillna(0)
        equity_curve = [initial_capital]
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        max_price_since_entry = 0
        cash = initial_capital
        holdings = 0

        for date, price, signal in zip(prices.index, prices, signals):
            if position == 1:
                current_val = cash + holdings * price
                if trailing_stop_pct > 0:
                    max_price_since_entry = max(max_price_since_entry, price)
                    stop_price = max_price_since_entry * (1 - trailing_stop_pct)
                    if price < stop_price:
                        position = 0
                        exit_price = price
                        cash = holdings * exit_price
                        holdings = 0
                        pnl = (exit_price - entry_price) / entry_price
                        trades.append({
                            'Side': 'Long', 'Entry Date': entry_date, 'Exit Date': date,
                            'Buy Price': entry_price, 'Sell Price': exit_price,
                            'PnL (%)': pnl * 100, 'Status': 'Trailing Stop'
                        })
                        equity_curve.append(cash)
                        continue
            else:
                current_val = cash

            if position == 0 and signal == 1:
                position = 1
                entry_price = price
                entry_date = date
                max_price_since_entry = price
                holdings = cash / price
                cash = 0
            elif position == 1 and signal == 0:
                position = 0
                exit_price = price
                cash = holdings * exit_price
                holdings = 0
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'Side': 'Long', 'Entry Date': entry_date, 'Exit Date': date,
                    'Buy Price': entry_price, 'Sell Price': exit_price,
                    'PnL (%)': pnl * 100, 'Status': 'Closed'
                })

            equity_curve.append(current_val)

        if position == 1:
            current_price = prices.iloc[-1]
            current_val = holdings * current_price
            pnl = (current_price - entry_price) / entry_price
            trades.append({
                'Side': 'Long', 'Entry Date': entry_date, 'Exit Date': None,
                'Buy Price': entry_price, 'Sell Price': current_price,
                'PnL (%)': pnl * 100, 'Status': 'Open'
            })
            equity_curve[-1] = current_val

        equity_curve_series = pd.Series(equity_curve[1:], index=prices.index)
        benchmark_curve = initial_capital * (1 + returns).cumprod()
        strat_returns = equity_curve_series.pct_change().fillna(0)

        return {
            'equity_curve': equity_curve_series,
            'benchmark_curve': benchmark_curve,
            'trades': pd.DataFrame(trades),
            'returns': strat_returns
        }

    @staticmethod
    def calculate_metrics(returns, risk_free_rate=0.0):
        if len(returns) < 2:
            return {}
        ann_factor = 252
        excess_ret = returns - (risk_free_rate / 252)
        sharpe = np.sqrt(ann_factor) * excess_ret.mean() / (returns.std() + 1e-9)
        downside = returns[returns < 0]
        sortino = np.sqrt(ann_factor) * excess_ret.mean() / (downside.std() + 1e-9)
        cum_ret = (1 + returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_dd = drawdown.min()
        total_ret = (1 + returns).prod()
        n_years = len(returns) / 252
        cagr = (total_ret ** (1/n_years)) - 1 if n_years > 0 else 0
        return {
            'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino,
            'Max Drawdown': max_dd, 'CAGR': cagr
        }


@st.cache_data(ttl=3600, show_spinner=False)
def fit_regime_model(model_data, n_regimes, switch_vol, switch_trend, search_reps=20):
    if hasattr(model_data, 'values'):
        clean_values = model_data.values.flatten().astype(float)
        idx = model_data.index
    else:
        clean_values = np.array(model_data).flatten().astype(float)
        idx = pd.RangeIndex(len(clean_values))

    if np.any(np.isnan(clean_values)) or np.any(np.isinf(clean_values)):
        st.error("❌ Data contains NaNs or Infinite values. Cannot fit model.")
        return None
    if np.std(clean_values) < 1e-9:
        st.error("❌ Data is constant (no variance). Cannot fit model.")
        return None

    endog_series = pd.Series(clean_values, index=idx)
    try:
        mod_markov = MarkovRegression(
            endog_series, k_regimes=n_regimes, trend='c',
            switching_variance=switch_vol, switching_trend=switch_trend
        )
        res_markov = mod_markov.fit(search_reps=search_reps, disp=False)
        if isinstance(res_markov.params, np.ndarray):
            names = res_markov.model.param_names
            res_markov.params = pd.Series(res_markov.params, index=names)
            res_markov.bse = pd.Series(res_markov.bse, index=names)
            res_markov.pvalues = pd.Series(res_markov.pvalues, index=names)
        return res_markov
    except Exception as e:
        st.error(f"❌ Fit failed: {str(e)}")
        if "Singular matrix" in str(e):
            st.warning("Hint: underlying data might be too flat or collinear.")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_master_signal(ticker, df, n_regimes=4, freq='Daily', opt_goal='Robustness (BIC)', stability=0,
                      switch_vol=True, switch_trend=True, engine='Markov',
                      initial_cap=10000.0, trailing_stop=0.0):
    try:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if stability > 0:
            df['Returns'] = df['Returns'].ewm(span=stability, adjust=False).mean()
            df['Log_Returns'] = df['Log_Returns'].ewm(span=stability, adjust=False).mean()
            df['Close'] = df['Close'].ewm(span=stability, adjust=False).mean()

        if freq == 'Weekly':
            df = df.resample('W').last().replace([np.inf, -np.inf], np.nan).dropna()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Returns'] = df['Close'].pct_change()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 15:
            return None

        if engine == 'Markov':
            if n_regimes == 'Auto':
                best_n = 4
                best_score = -float('inf') if opt_goal == 'Performance (PnL)' else float('inf')
                best_r = None
                for n in [2, 3, 4]:
                    try:
                        r = fit_regime_model(df['Returns']*100, n, switch_vol, switch_trend, search_reps=5)
                        if r:
                            if opt_goal == 'Performance (PnL)':
                                p_df = r.filtered_marginal_probabilities
                                r_means = []
                                for i in range(n):
                                    m = r.params[f'const[{i}]'] if f'const[{i}]' in r.params else r.params.get('const', 0.0)
                                    r_means.append((i, m))
                                bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
                                dom = p_df.idxmax(axis=1)
                                sigs = (dom == bull_idx).astype(int)
                                bt_res = BacktestEngine.run_strategy(df['Close'], sigs, initial_cap, trailing_stop)
                                pnl = (bt_res['equity_curve'].iloc[-1] / initial_cap - 1)
                                if pnl > best_score:
                                    best_score = pnl
                                    best_n = n
                                    best_r = r
                            else:
                                score = r.bic
                                if score < best_score:
                                    best_score = score
                                    best_n = n
                                    best_r = r
                    except:
                        continue
                res_markov = best_r
            else:
                res_markov = fit_regime_model(df['Returns']*100, int(n_regimes), switch_vol, switch_trend)

            if not res_markov:
                return None

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
            p_detector = None
        else:
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
                        except:
                            continue
                    p_detector.fit(n_states=best_n)
                else:
                    p_detector.fit_optimized()
            else:
                p_detector.fit(n_states=int(n_regimes))
            regime_sig, regime_prob, regime_label = p_detector.get_latest_verdict()
            regime_data = {'label': regime_label, 'confidence': regime_prob, 'n_states': p_detector.metrics.get('n_states', 4)}

        k_filter = KalmanFilterTrend(process_noise=1e-4, measurement_noise=1e-2)
        trend_est, _ = k_filter.filter(df['Close'].values)
        last_price = df['Close'].iloc[-1]
        last_trend = trend_est[-1]
        trend_diff = (last_price - last_trend) / (last_trend + 1e-9)

        returns_scaled = df['Returns'] * 100
        returns_scaled = returns_scaled.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns_scaled) < 15:
            return None

        am = arch_model(returns_scaled, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off')
        curr_vol = res.conditional_volatility.iloc[-1]
        avg_vol = res.conditional_volatility.mean()
        vol_state = "HIGH" if curr_vol > avg_vol * 1.2 else "LOW" if curr_vol < avg_vol * 0.8 else "NORMAL"

        jump_res = RealizedVolatility.jump_component(df['Returns'].values)
        jump_detected = jump_res['p_value'] < 0.05

        sentiment_score = 0
        if "LONG" in regime_sig: sentiment_score += 2
        if "SHORT" in regime_sig: sentiment_score -= 2
        if trend_diff > 0.01: sentiment_score += 1
        if trend_diff < -0.01: sentiment_score -= 1
        if vol_state == "LOW": sentiment_score += 1
        if vol_state == "HIGH": sentiment_score -= 1
        if jump_detected: sentiment_score -= 1

        return {
            'regime_sig': regime_sig, 'regime_label': regime_label, 'regime_data': regime_data,
            'regime_prob': regime_prob, 'pro_detector': p_detector, 'trend_diff': trend_diff,
            'vol_state': vol_state, 'curr_vol': curr_vol, 'jump_detected': jump_detected,
            'sentiment_score': sentiment_score, 'garch_res': res
        }
    except Exception as e:
        st.error(f"Error in Decision Engine for {ticker}: {e}")
        return None


@st.cache_data(ttl=60)
def load_data(ticker, start, end, interval='1d'):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return None
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1, drop_level=True) if ticker in df.columns.get_level_values(1) else df
            if ticker in df.columns:
                df = df[ticker]
            elif 'Close' in df.columns and len(df.columns) > 1 and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None


@st.cache_data(ttl=3600)
def load_fred_data(series_id):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading FRED {series_id}: {e}")
        return None


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
    import requests
    import io
    headers = {'User-Agent': 'Mozilla/5.0'}
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
        pass


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
            "INTC", "SBUX", "AMD", "GILD", "VRTX", "MDLZ", "REGN", "ISRG", "ADI",
            "BKNG", "AMAT", "ADP", "PDD", "PYPL", "MU", "VRSK", "MELI", "KDP"
        ]


@st.cache_data(ttl=3600*24)
def get_total_us_stocks():
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
            other_url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
            try:
                other = robust_fetch_csv(other_url, sep="|")
                other = other[(other['Test Issue'] == 'N') & (other['ETF'] == 'N')]
                tickers += other['NASDAQ Symbol'].tolist()
            except:
                pass
            res = sorted(list(set([str(t).strip() for t in tickers if str(t).strip() and len(str(t)) < 6])))
            if len(res) > 500:
                return res
        except:
            continue
    st.warning("⚠️ Total Market Connection Issue. Using expanded internal universe.")
    return get_sp500_tickers() + ["AAPL", "TSLA", "NVDA", "AMD", "PLTR", "SQ", "PYPL", "COIN", "MARA", "RIOT"]


@st.cache_data(ttl=3600*24)
def get_total_us_etfs():
    try:
        url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
        data = robust_fetch_csv(url, sep="|")
        etfs = data[(data['Test Issue'] == 'N') & (data['ETF'] == 'Y')]
        res = sorted(list(set([str(t).strip() for t in etfs['NASDAQ Symbol'].tolist()])))
        if len(res) > 100:
            return res
    except:
        pass
    st.warning("⚠️ ETF Universe Connection Issue.")
    return []


def get_market_cap(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.info.get('marketCap', 0)
    except:
        return 0


def get_analyst_target(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        target = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('previousClose')
        if target and current:
            implied_return = np.log(target / current)
            return target, implied_return
        return None, None
    except:
        return None, None


def calculate_beta(ticker_returns, benchmark_ticker='SPY', lookback_years=2):
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_years*365)
        bench = yf.download(benchmark_ticker, start=start, end=end, progress=False)
        if isinstance(bench.columns, pd.MultiIndex):
            if benchmark_ticker in bench.columns.get_level_values(1):
                bench = bench.xs(benchmark_ticker, axis=1, level=1, drop_level=True)
            elif 'Close' in bench.columns:
                bench.columns = bench.columns.droplevel(1)
        if 'Close' not in bench.columns and 'Adj Close' in bench.columns:
            bench['Close'] = bench['Adj Close']
        bench_ret = bench['Close'].pct_change().dropna()
        common_idx = ticker_returns.index.intersection(bench_ret.index)
        if len(common_idx) < 30:
            return 1.0
        y = ticker_returns.loc[common_idx]
        x = bench_ret.loc[common_idx]
        cov = np.cov(y, x)[0, 1]
        var = np.var(x)
        return cov / var
    except:
        return 1.0


class ReportGenerator:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_store = {}
        self.plots = {}

    def add_data(self, key, df_or_dict):
        self.data_store[key] = df_or_dict

    def add_plot(self, key, fig):
        buf = io.BytesIO()
        if hasattr(fig, 'savefig'):
            fig.savefig(buf, format='png', bbox_inches='tight')
        elif hasattr(fig, 'write_image'):
            try:
                fig.write_image(buf, format='png', engine='kaleido')
            except Exception as e:
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
        pdf.set_font("Arial", 'B', 24)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(0, 20, f"Unified Quant Analysis Report", ln=True, align='C')
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"Asset: {self.ticker}", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, f"Analysis Period: {self.start_date} to {self.end_date}", ln=True, align='C')
        pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(10)
        printed_plots = set()
        for key, data in self.data_store.items():
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(31, 119, 180)
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
            if key in self.plots:
                pdf.ln(2)
                pdf.image(self.plots[key], x=15, w=180)
                printed_plots.add(key)
                pdf.ln(5)
            pdf.ln(10)
            if pdf.get_y() > 230:
                pdf.add_page()
        remaining_plots = [k for k in self.plots.keys() if k not in printed_plots]
        if remaining_plots:
            if pdf.get_y() > 100:
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
        pdf_raw = pdf.output()
        if isinstance(pdf_raw, bytearray):
            return bytes(pdf_raw)
        return pdf_raw


# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("Thesis Parameters")

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
        BENCHMARK = "GC=F"
        DEFAULT_RF = 4.0
        SUFFIX = ""
    else:
        CURRENCY = "$"
        BENCHMARK = "SPY"
        DEFAULT_RF = 4.0
        SUFFIX = ""

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        raw_ticker = st.text_input("Main Ticker", "RELIANCE" if market_region == "Indian Market (INR)" else "AAPL").upper()
    with col_t2:
        raw_pair = st.text_input("Pair Ticker", "").upper()

    TICKER = raw_ticker + SUFFIX if (SUFFIX and not raw_ticker.endswith(SUFFIX)) else raw_ticker
    PAIR_TICKER = raw_pair + SUFFIX if (SUFFIX and raw_pair and not raw_pair.endswith(SUFFIX)) else raw_pair

    st.caption(f"Active Ticker: {TICKER}")

    with st.expander("🛠️ Debug Info", expanded=False):
        st.write(f"Region: {market_region}")
        st.code(f"SUFFIX = '{SUFFIX}'")
        st.code(f"Raw Ticker = '{raw_ticker}'")
        st.code(f"Final TICKER = '{TICKER}'")

    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
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
                               index=0)
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
        trailing_stop = st.slider("Trailing Stop Loss (%)", 0.0, 20.0, 0.0, step=0.5) / 100

    st.divider()
    st.header("⚡ Live Decision Mode")
    live_mode = st.toggle("Enable Live Data", value=False)
    if live_mode:
        data_interval = st.selectbox("Live Interval", ["1m", "5m", "15m", "60m"], index=1)
        st.info("Live mode uses a shorter window and higher frequency data.")
        if st.button("🔄 Refresh Live Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    else:
        data_interval = '1d'

    st.subheader("Report Export")
    if not EXPORT_AVAILABLE:
        st.error("📥 Export libraries missing.")
        st.info("Run: `pip install fpdf2 xlsxwriter`")
    else:
        if 'report_gen' not in st.session_state:
            st.session_state.report_gen = None
        if st.session_state.report_gen:
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                try:
                    raw_pdf = st.session_state.report_gen.generate_pdf()
                    if isinstance(raw_pdf, bytes):
                        pdf_bytes = raw_pdf
                    elif isinstance(raw_pdf, bytearray):
                        pdf_bytes = bytes(raw_pdf)
                    elif isinstance(raw_pdf, str):
                        pdf_bytes = raw_pdf.encode('latin1')
                    else:
                        pdf_bytes = bytes(raw_pdf)
                    st.download_button(
                        label="📥 PDF Report", data=pdf_bytes,
                        file_name=f"Quant_Report_{TICKER}.pdf",
                        mime="application/pdf", key="pdf_download_btn"
                    )
                except Exception as e:
                    st.error(f"PDF Error: {str(e)}")
            with col_ex2:
                try:
                    excel_bytes = st.session_state.report_gen.generate_excel()
                    st.download_button(
                        label="📥 Excel Data", data=excel_bytes,
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
now_rounded = datetime.now().replace(second=0, microsecond=0)

if live_mode:
    lookback_days = 7 if data_interval == '1m' else 30
    df_main = load_data(TICKER, now_rounded - timedelta(days=lookback_days), now_rounded, interval=data_interval)
else:
    df_main = load_data(TICKER, start_date, end_date, interval='1d')

st.subheader("Asset & Macro Analysis Suite")

# ==========================================
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
    "🏦 FED Balance Sheet"
])

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = tabs

if df_main is not None:
    st.session_state.report_gen = ReportGenerator(TICKER, start_date, end_date)
    st.session_state.report_gen.add_data("Historical Data", df_main.tail(100))

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
                                 trailing_stop=trailing_stop)
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
        res_sum = analysis['garch_res']
        prog_bar.progress(100)
        prog_bar.empty()
    else:
        st.sidebar.error("Decision Engine Error.")
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
        st.info("💡 **Welcome**. Enter a ticker in the sidebar to begin.")
    else:
        st.write("### 🧠 Executive Decision Dashboard")
    st.markdown(f"**Unified Quant Signal for {TICKER}** | Interval: `{data_interval}` | Mode: {'Live' if live_mode else 'Historical'}")

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
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.write("#### Master Quant Score")
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
            st.error("🚨 **FAT TAIL RISK**: Significant price jumps detected.")
        else:
            st.success("✅ **SMOOTH DYNAMICS**: No significant jumps.")
        if vol_state == "HIGH":
            st.warning("⚠️ **VOL CLUSTERING**: Recent shocks are likely to trigger further volatility.")
        st.info(f"**Recommendation**: {regime_sig}. Target Exposure: {min(1.0, 0.5 + 0.1*sentiment_score):.0%}")

    st.divider()
    st.caption("This summary aggregates deep statistical models. Visit respective tabs for details.")

# ==========================================
# TAB 1: VOLATILITY (GARCH/Risk)
# ==========================================
with tab1:
    if df_main is None:
        st.warning("Please load a ticker to view Volatility models.")
    else:
        st.write("### 📉 Advanced Volatility Analysis")

    if res_sum is not None:
        latest_vol = res_sum.conditional_volatility.iloc[-1]
        vol_msg = f"Volatility is currently **{vol_state}** ({latest_vol:.2f}% daily)."
        if vol_state == "HIGH": st.error(f"🎯 **MODEL VERDICT**: {vol_msg} Defensive sizing recommended.")
        else: st.success(f"🎯 **MODEL VERDICT**: {vol_msg} Risk environment is stable.")

    if ARCH_AVAILABLE and df_main is not None:
        returns_pct = df_main['Returns'] * 100

        with st.expander("⚙️ Model Configuration", expanded=True):
            c_mdl1, c_mdl2, c_mdl3 = st.columns(3)
            with c_mdl1:
                vol_model_type = st.selectbox("Volatility Model", ["GARCH", "GJR-GARCH", "EGARCH"])
            with c_mdl2:
                dist_type = st.selectbox("Distribution", ["Normal", "Student's t", "Skewed Student's t"])
            with c_mdl3:
                vol_lag = st.slider("GARCH Lag (p, q)", 1, 3, 1)

        vol_map = {"GARCH": "Garch", "GJR-GARCH": "Garch", "EGARCH": "EGarch"}
        dist_map = {"Normal": "Normal", "Student's t": "t", "Skewed Student's t": "skewt"}
        o_param = 1 if vol_model_type == "GJR-GARCH" else 0

        try:
            am = arch_model(returns_pct, vol=vol_map[vol_model_type], p=vol_lag, o=o_param, q=vol_lag, dist=dist_map[dist_type])
            res = am.fit(disp='off')

            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                st.subheader("Conditional Volatility")
                fig_v, ax_v = plt.subplots(figsize=(10, 4))
                ax_v.plot(res.conditional_volatility, color='#2980b9', linewidth=1.5, label=f'{vol_model_type} Vol')
                ax_v.set_title(f"{vol_model_type} ({dist_type}) Conditional Volatility")
                ax_v.legend()
                format_plot_dates(ax_v, returns_pct.index)
                st.pyplot(fig_v)
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

                pers_val = np.nan
                if 'beta[1]' in res.params and 'alpha[1]' in res.params:
                    pers_val = res.params['alpha[1]'] + res.params['beta[1]']
                    if vol_model_type == 'GJR-GARCH' and 'gamma[1]' in res.params:
                        pers_val += res.params['gamma[1]'] / 2

                if not np.isnan(pers_val):
                    st.metric("Persistence", f"{pers_val:.4f}")
                    if pers_val < 1:
                        half_life = np.log(0.5) / np.log(pers_val)
                        st.metric("Half-Life (Days)", f"{half_life:.1f}")
                    else:
                        st.caption("Non-stationary (Persistence >= 1)")

                if 'gamma[1]' in res.params:
                    gamma_val = res.params['gamma[1]']
                    st.metric("Leverage (Gamma)", f"{gamma_val:.4f}")
                    if gamma_val > 0.05:
                        st.success("✅ Leverage Effect Confirmed.")
                    elif gamma_val < -0.05:
                        st.info("Inverse Leverage Structure.")
                    else:
                        st.caption("No significant asymmetry.")

                st.markdown("---")
                st.metric("AIC", f"{res.aic:.2f}")
                st.metric("BIC", f"{res.bic:.2f}")

            tab_diag, tab_cast, tab_risk = st.tabs(["🔍 Diagnostics", "🔮 Forecasting", "🛡️ Risk Management"])

            with tab_diag:
                d_col1, d_col2 = st.columns(2)
                std_resid = res.std_resid
                with d_col1:
                    st.markdown("**Standardized Residuals**")
                    fig_r, ax_r = plt.subplots(figsize=(8, 4))
                    ax_r.plot(std_resid, color='gray', alpha=0.7)
                    ax_r.axhline(0, color='black', linestyle='--')
                    format_plot_dates(ax_r, returns_pct.index)
                    st.pyplot(fig_r)
                with d_col2:
                    st.markdown("**Q-Q Plot (vs Normal)**")
                    fig_qq = plt.figure(figsize=(8, 4))
                    ax_qq = fig_qq.add_subplot(111)
                    stats.probplot(std_resid, dist="norm", plot=ax_qq)
                    st.pyplot(fig_qq)

                st.markdown("**Residual Diagnostics**")
                lb_test = acorr_ljungbox(std_resid, lags=[10], return_df=True)
                arch_test = het_arch(std_resid)
                diag_data = {
                    "Test": ["Ljung-Box (No Serial Corr)", "ARCH-LM (No ARCH Effect)"],
                    "p-value": [lb_test['lb_pvalue'].iloc[0], arch_test[1]],
                    "Conclusion": [
                        "Fail to Reject H0 (Good)" if lb_test['lb_pvalue'].iloc[0] > 0.05 else "Reject H0 (Bad)",
                        "Fail to Reject H0 (Good)" if arch_test[1] > 0.05 else "Reject H0 (Bad)"
                    ]
                }
                st.table(pd.DataFrame(diag_data).set_index("Test"))

            with tab_cast:
                f_horizon = st.slider("Forecast Horizon (Days)", 1, 63, 21)
                try:
                    forecasts = res.forecast(horizon=f_horizon, reindex=False)
                except ValueError:
                    forecasts = res.forecast(horizon=f_horizon, method='simulation', simulations=1000, reindex=False)

                var_forecast = forecasts.variance.iloc[-1]
                vol_forecast = np.sqrt(var_forecast)

                fig_f, ax_f = plt.subplots(figsize=(10, 4))
                last_days = 60
                hist_dates = returns_pct.index[-last_days:]
                hist_vol = res.conditional_volatility[-last_days:]
                ax_f.plot(hist_dates, hist_vol, color='black', alpha=0.5, label='Historical Vol')
                fut_dates = [returns_pct.index[-1] + timedelta(days=i) for i in range(1, f_horizon+1)]
                ax_f.plot(fut_dates, vol_forecast, color='red', marker='o', linestyle='--', label='Forecast Vol')
                ax_f.set_title("Volatility Term Structure Forecast")
                format_plot_dates(ax_f, hist_dates)
                ax_f.legend()
                st.pyplot(fig_f)

                current_vol_v = res.conditional_volatility[-1]
                if vol_forecast.iloc[-1] < current_vol_v:
                    st.success("Mean Reversion: Volatility expected to DECLINE.")
                else:
                    st.warning("Mean Reversion: Volatility expected to RISE.")

            with tab_risk:
                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    acc_size = st.number_input("Portfolio Value", 1000, 10000000, 100000)
                    conf_level = st.selectbox("Confidence Level", [0.95, 0.99])
                    try:
                        forecasts_risk = res.forecast(horizon=1, reindex=False)
                    except ValueError:
                        forecasts_risk = res.forecast(horizon=1, method='simulation', simulations=1000, reindex=False)
                    next_vol = np.sqrt(forecasts_risk.variance.iloc[-1].iloc[0]) / 100
                    if dist_type == "Normal":
                        q = stats.norm.ppf(1-conf_level)
                    elif dist_type == "Student's t":
                        nu = res.params.get('nu')
                        q = stats.t.ppf(1-conf_level, df=nu)
                    else:
                        nu = res.params.get('nu')
                        lam = res.params.get('lambda')
                        if nu is not None and lam is not None:
                            dist_inst = am.distribution
                            q = dist_inst.ppf(1-conf_level, [nu, lam])
                        else:
                            q = stats.norm.ppf(1-conf_level)
                    var_pct = -q * next_vol
                    var_val = var_pct * acc_size
                    st.metric(f"1-Day VaR ({conf_level:.0%})", f"{CURRENCY}{var_val:,.2f}", f"-{var_pct*100:.2f}%")

                with r_col2:
                    target_vol = st.slider("Target Annual Volatility (%)", 5, 50, 15) / 100
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

    elif not ARCH_AVAILABLE:
        st.warning("⚠️ 'arch' library not found. Run `pip install arch`.")

# ==========================================
# TAB 2: REGIME SWITCHING
# ==========================================
with tab2:
    if df_main is None:
        st.warning("Please load a ticker to view Regime Switching models.")
    else:
        st.write("### Markov Regime Switching Model")

    if "LONG" in regime_sig: st.success(f"🎯 **MODEL VERDICT**: Confirmed **{regime_sig}** in {regime_data['label']}.")
    elif "SHORT" in regime_sig: st.error(f"🎯 **MODEL VERDICT**: Confirmed **{regime_sig}**. Market risk is elevated.")
    else: st.info(f"🎯 **MODEL VERDICT**: {regime_sig}. Await confirmation.")

    if df_main is not None:
        col_config1, col_config2, col_config3 = st.columns(3)
        with col_config1:
            regime_freq = st.selectbox("Data Frequency", ["Daily", "Weekly"], index=1)
        with col_config2:
            lookback_years = st.slider("Lookback Period (Years)", 1, 10, 2)
        with col_config3:
            n_regimes = st.slider("Number of Regimes", 2, 4, 2)

        stability = st.slider("Signal Stability (Pre-Smoothing)", 0, 10, 4)
        conviction_thresh = st.slider("High-Conviction Threshold", 0.5, 0.95, 0.7, step=0.05)

        col_sw1, col_sw2 = st.columns(2)
        with col_sw1:
            switch_trend = st.checkbox("Switching Mean (Trend)", value=True)
        with col_sw2:
            switch_vol = st.checkbox("Switching Volatility", value=True)

        warnings_list = []
        if lookback_years <= 1:
            warnings_list.append("⚠️ Very short history - consider 3+ years for stable regimes")
            if regime_freq == "Weekly":
                warnings_list.append("❌ Cannot use Weekly with <1 year. Switch to Daily.")
                regime_freq = "Daily"
        if regime_freq == "Daily" and switch_trend and lookback_years < 3:
            warnings_list.append("⚠️ Daily + Switching Trend needs 3+ years. Disabling...")
            switch_trend = False
        for w in warnings_list:
            st.warning(w)

        start_dt_regime = datetime.now() - timedelta(days=lookback_years*365)
        df_regime = load_data(TICKER, start_dt_regime, end_date)

        if df_regime is not None:
            if regime_freq == "Weekly":
                returns = df_regime['Returns'].resample('W').sum()
            else:
                returns = df_regime['Returns']

            if stability > 0:
                returns = returns.ewm(span=stability, adjust=False).mean()
                st.caption(f"ℹ️ Applied EWMA Smoothing (Span={stability})")

            try:
                model_data = returns.dropna() * 100
                if len(model_data) < 10:
                    st.error("Insufficient data points (>10 required).")
                    st.stop()
                model_data = pd.Series(model_data.values.flatten().astype(float), index=model_data.index)
            except Exception as e:
                st.error(f"Data Prep Error: {e}")
                st.stop()

            st.caption(f"Modeling {len(model_data)} {regime_freq.lower()} returns from {start_dt_regime.date()}")

            with st.spinner(f"Fitting {n_regimes}-regime model..."):
                res_markov = fit_regime_model(model_data, n_regimes, switch_vol, switch_trend)

            if res_markov is None:
                st.error("Model fitting failed.")
                st.stop()

            if not res_markov.mle_retvals['converged']:
                st.error("⛔ Model did not converge.")
                st.stop()

            trans_matrix = np.squeeze(res_markov.regime_transition)
            if trans_matrix.ndim < 2:
                trans_matrix = np.atleast_2d(trans_matrix)

            if np.any(trans_matrix > 0.99):
                st.warning("⚠️ Near-permanent regimes detected - consider fewer regimes")

            regime_stats = []
            for i in range(n_regimes):
                if f'const[{i}]' in res_markov.params:
                    mean_val = res_markov.params[f'const[{i}]']
                else:
                    mean_val = res_markov.params.get('const', 0.0)
                if f'sigma2[{i}]' in res_markov.params:
                    vol_val = np.sqrt(res_markov.params[f'sigma2[{i}]'])
                else:
                    vol_val = np.sqrt(res_markov.params.get('sigma2', 1.0))
                regime_stats.append({
                    'regime': i, 'mean': float(mean_val),
                    'vol': float(vol_val), 'persistence': float(trans_matrix[i, i])
                })

            regime_stats = sorted(regime_stats, key=lambda x: x['mean'], reverse=True)

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
                    st.caption(f"Avg duration: {avg_duration:.1f} periods")

            st.session_state.report_gen.add_data("Regime Statistics", pd.DataFrame(regime_stats))

            last_probs = res_markov.filtered_marginal_probabilities.iloc[-1]
            current_regime = np.argmax(last_probs)
            current_prob = last_probs.iloc[current_regime]
            regime_label_tab2 = labels[[r['regime'] for r in regime_stats].index(current_regime)]
            is_conviction = current_prob >= conviction_thresh
            stability_score = np.mean([r['persistence'] for r in regime_stats])

            st.divider()
            c_dash1, c_dash2, c_dash3 = st.columns(3)
            with c_dash1:
                st.caption("Current State")
                if is_conviction:
                    st.subheader(f"{regime_label_tab2}")
                    st.success(f"High Conviction ({current_prob:.1%})")
                else:
                    st.subheader("⚪ Mixed / Uncertain")
                    st.warning(f"Low Conviction ({current_prob:.1%})")
            with c_dash2:
                sorted_probs = sorted(last_probs.values, reverse=True)
                spread = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0)
                st.metric("Probability Spread", f"{spread:.1%}")
                st.progress(max(0.0, min(1.0, float(spread))))
            with c_dash3:
                st.metric("Avg Persistence", f"{stability_score:.1%}")
                expected_switches = (1 - stability_score) * (52 if regime_freq == "Weekly" else 252)
                st.caption(f"Exp. Switches/Year: ~{expected_switches:.1f}")

            st.write(f"**As of:** {model_data.index[-1].date()}")
            st.divider()

            fig_m, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(model_data.index, model_data, color='black', alpha=0.6, linewidth=1)
            for i, regime in enumerate(regime_stats):
                probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
                color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
                color = plt.cm.RdYlGn(color_idx)
                axes[0].fill_between(model_data.index, model_data.min(), model_data.max(),
                                     where=(probs > 0.6), alpha=0.15, color=color, label=labels[i])
            axes[0].set_title(f"{TICKER} Returns with Regime Periods")
            axes[0].legend(loc='upper left')
            axes[0].set_ylabel("Return (%)")

            smooth_probs = st.checkbox("Smooth Probabilities (4-period Rolling)", value=True, key="smooth_probs_check")
            for i, regime in enumerate(regime_stats):
                color_idx = 1 - (i / (n_regimes - 1)) if n_regimes > 1 else 1.0
                color = plt.cm.RdYlGn(color_idx)
                raw_probs = res_markov.filtered_marginal_probabilities.iloc[:, regime['regime']]
                plot_probs = raw_probs.rolling(window=4, min_periods=1).mean() if smooth_probs else raw_probs
                axes[1].fill_between(model_data.index, 0, plot_probs, color=color, alpha=0.3, label=labels[i])
                axes[1].plot(model_data.index, plot_probs, color=color, linewidth=1.5)
            axes[1].axhline(1/n_regimes, color='gray', linestyle='--', alpha=0.4)
            axes[1].set_title("Regime Probabilities (Filtered)")
            axes[1].set_ylabel("Probability")
            axes[1].set_ylim([0, 1])
            axes[1].legend()

            def get_const(i):
                if f'const[{i}]' in res_markov.params:
                    return float(res_markov.params[f'const[{i}]'])
                return float(res_markov.params.get('const', 0.0))

            expected_ret = pd.Series(0.0, index=model_data.index)
            for i in range(n_regimes):
                prob = res_markov.filtered_marginal_probabilities.iloc[:, i]
                expected_ret += prob * get_const(i)

            axes[2].plot(model_data.index, expected_ret, color='darkblue', linewidth=2)
            axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
            axes[2].fill_between(model_data.index, 0, expected_ret, where=(expected_ret > 0), color='green', alpha=0.3)
            axes[2].fill_between(model_data.index, 0, expected_ret, where=(expected_ret < 0), color='red', alpha=0.3)
            axes[2].set_title("Regime-Weighted Expected Return")
            axes[2].set_ylabel("Expected Return (%)")

            format_plot_dates(axes[-1], model_data.index)
            axes[-1].tick_params(labelbottom=True)
            for ax in axes[:-1]:
                ax.tick_params(labelbottom=False)

            st.pyplot(fig_m)
            st.session_state.report_gen.add_plot("Regime Switching Analysis", fig_m)

            with st.expander("📋 Technical Parameters"):
                summary_data = {
                    "Parameter": res_markov.params.index,
                    "Value": res_markov.params.values.astype(float),
                    "Std Error": res_markov.bse.values.astype(float),
                    "P-Value": res_markov.pvalues.values.astype(float)
                }
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary.style.format({"Value": "{:.4f}", "Std Error": "{:.4f}", "P-Value": "{:.4f}"}))
                st.caption("AIC: {:.2f} | BIC: {:.2f}".format(res_markov.aic, res_markov.bic))

# ==========================================
# TAB 3: STOCHASTIC MODELS
# ==========================================
with tab3:
    if df_main is None:
        st.warning("Please load a ticker to view Stochastic/Jump models.")
    else:
        st.write("### Advanced Stochastic Simulations")

    if df_main is not None:
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

        if drift_type == "Risk-Neutral (Risk-Free Rate)":
            mu_drift = rf_rate
        elif drift_type == "Historical Mean (Real World)":
            hist_mu = df_main['Log_Returns'].mean() * 252
            mu_drift = hist_mu
        elif drift_type == "CAPM (Expected Return)":
            beta = calculate_beta(df_main['Returns'], benchmark_ticker=BENCHMARK)
            mkt_return = 0.08
            capm_ret = rf_rate + beta * (mkt_return - rf_rate)
            mu_drift = np.log(1 + capm_ret)
            st.metric("CAPM Beta", f"{beta:.2f}")
        elif drift_type == "Analyst Consensus (1Y Target)":
            target, implied_ret = get_analyst_target(TICKER)
            if target:
                mu_drift = implied_ret
                st.metric("Analyst Target", f"{CURRENCY}{target:.2f}")
            else:
                st.warning("No Analyst Target found. Reverting to Historical.")
                mu_drift = df_main['Log_Returns'].mean() * 252
        else:
            custom_ret = st.number_input("Expected Annual Return (%)", -50.0, 100.0, 10.0) / 100
            mu_drift = np.log(1 + custom_ret)

        last_date = df_main.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(253)]

        import plotly.graph_objects as go

        seed = st.number_input("Random Seed", 1, 10000, 42)
        np.random.seed(seed)

        if sim_type == "Merton Jump Diffusion":
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Parameters**")
                mj_lam = st.slider("Jump Intensity (Lambda)", 0.1, 10.0, 1.0)
                mj_mu = st.slider("Jump Mean Size", -0.5, 0.5, -0.1)
                mj_sigma = st.slider("Jump Std Dev", 0.01, 0.5, 0.1)
                mj_vol = st.slider("Diffusive Volatility", 0.05, 1.0, 0.2)
            with col2:
                current_price = df_main['Close'].iloc[-1]
                paths = merton_jump_diffusion(current_price, 1.0, mu_drift, mj_vol, mj_lam, mj_mu, mj_sigma, 252, 50)
                mean_path = paths.mean(axis=1)
                median_path = np.median(paths, axis=1)
                p05_path = np.percentile(paths, 5, axis=1)
                p95_path = np.percentile(paths, 95, axis=1)
                final_mean = mean_path[-1]
                final_median = median_path[-1]
                m1, m2 = st.columns(2)
                m1.metric("Projected Mean", f"{CURRENCY}{final_mean:,.2f}")
                m2.metric("Projected Median", f"{CURRENCY}{final_median:,.2f}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=np.concatenate([p95_path, p05_path[::-1]]),
                    fill='toself', fillcolor='rgba(100,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='90% CI'
                ))
                for i in range(min(20, paths.shape[1])):
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=paths[:, i], mode='lines',
                        line=dict(color='rgba(100,100,255,0.05)', width=1),
                        showlegend=False, hoverinfo='skip'
                    ))
                fig.add_trace(go.Scatter(x=future_dates, y=mean_path, mode='lines', name='Mean Path', line=dict(color='orange', width=3)))
                fig.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines', name='Median Path', line=dict(color='white', width=3, dash='dash')))
                fig.update_layout(title=f"Merton Jump Diffusion: {TICKER}", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.report_gen.add_data("Merton Metrics", {"Mean": final_mean, "Median": final_median})

        elif sim_type == "Heston Stochastic Volatility":
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("**Heston Params**")
                if 'h_kappa' not in st.session_state: st.session_state.h_kappa = 2.0
                if 'h_theta' not in st.session_state: st.session_state.h_theta = 0.04
                if 'h_xi' not in st.session_state: st.session_state.h_xi = 0.3
                if 'h_rho' not in st.session_state: st.session_state.h_rho = -0.7
                if 'h_v0' not in st.session_state: st.session_state.h_v0 = 0.04
                st.caption("Methodology: Historical Proxy Calibration (GARCH-based)")
                if st.button("Calibrate from History"):
                    with st.spinner("Calibrating..."):
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
                h_kappa = st.number_input("Kappa (Mean Rev Speed)", 0.01, 1000.0, key='h_kappa', format="%.4f")
                h_theta = st.number_input("Theta (Long Term Vol)", 0.0, 5.0, key='h_theta', format="%.6f")
                h_xi = st.number_input("Xi (Vol of Vol)", 0.01, 100.0, key='h_xi', format="%.4f")
                h_rho = st.slider("Rho (Correlation)", -0.99, 0.99, key='h_rho')
                h_v0 = st.number_input("Initial Variance", 0.0, 5.0, key='h_v0', format="%.6f")
            with col2:
                current_price = df_main['Close'].iloc[-1]
                sim_prices, sim_vols = simulate_heston(current_price, 1.0, mu_drift, h_kappa, h_theta, h_xi, h_rho, h_v0, 252, 50)
                mean_path = sim_prices.mean(axis=1)
                median_path = np.median(sim_prices, axis=1)
                p05_path = np.percentile(sim_prices, 5, axis=1)
                p95_path = np.percentile(sim_prices, 95, axis=1)
                final_mean = mean_path[-1]
                final_median = median_path[-1]
                m1, m2 = st.columns(2)
                m1.metric("Projected Mean", f"{CURRENCY}{final_mean:,.2f}")
                m2.metric("Projected Median", f"{CURRENCY}{final_median:,.2f}")
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=np.concatenate([p95_path, p05_path[::-1]]),
                    fill='toself', fillcolor='rgba(100,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'), name='90% CI'
                ))
                for i in range(min(20, sim_prices.shape[1])):
                    fig_h.add_trace(go.Scatter(
                        x=future_dates, y=sim_prices[:, i], mode='lines',
                        line=dict(color='rgba(100,100,255,0.05)', width=1),
                        showlegend=False, hoverinfo='skip'
                    ))
                fig_h.add_trace(go.Scatter(x=future_dates, y=mean_path, mode='lines', name='Mean Path', line=dict(color='orange', width=3)))
                fig_h.add_trace(go.Scatter(x=future_dates, y=median_path, mode='lines', name='Median Path', line=dict(color='white', width=3, dash='dash')))
                fig_h.update_layout(title=f"Heston Price Paths ({TICKER})", template="plotly_dark", hovermode="x unified")
                st.plotly_chart(fig_h, use_container_width=True)
                st.session_state.report_gen.add_data("Heston Metrics", {"Mean": final_mean, "Median": final_median})
                st.write("**Stochastic Volatility Paths**")
                fig_v = go.Figure()
                for i in range(min(20, sim_vols.shape[1])):
                    fig_v.add_trace(go.Scatter(
                        x=future_dates, y=np.sqrt(sim_vols[:, i]), mode='lines',
                        line=dict(color='rgba(255,165,0,0.3)', width=1), showlegend=False
                    ))
                fig_v.update_layout(title="Volatility Process (Sigma)", template="plotly_dark", height=300)
                st.plotly_chart(fig_v, use_container_width=True)

# ==========================================
# TAB 4: KALMAN FILTER
# ==========================================
with tab4:
    if df_main is None:
        st.warning("Please load a ticker to view Kalman Filter dynamics.")
    else:
        st.write("### Kalman Filter Analysis")

    if df_main is not None:
        if trend_diff > 0.03: st.success(f"🎯 **MODEL VERDICT**: Price is **{trend_diff:.1%} ABOVE** the Kalman Trend.")
        elif trend_diff < -0.03: st.error(f"🎯 **MODEL VERDICT**: Price is **{abs(trend_diff):.1%} BELOW** the Kalman Trend.")
        else: st.info(f"🎯 **MODEL VERDICT**: Price is within **{abs(trend_diff):.1%}** of the Kalman Trend.")

        kf_mode = st.radio("Analysis Mode", ["Pairs Trading (Relative Value)", "Single Asset (Trend)"])

        if kf_mode == "Pairs Trading (Relative Value)":
            st.write(f"**{TICKER} vs {PAIR_TICKER}**")
            df_pair = load_data(PAIR_TICKER, start_date, end_date)
            if df_pair is not None:
                common_idx = df_main.index.intersection(df_pair.index)
                y = df_main.loc[common_idx, 'Close'].values
                x = df_pair.loc[common_idx, 'Close'].values
                if len(y) > 10:
                    kf = KalmanFilterReg(delta=1e-4, R=1e-3)
                    state_means, state_covs = kf.run_filter(y, x)
                    alpha = state_means[:, 0]
                    beta = state_means[:, 1]
                    fig_k, (ax_k1, ax_k2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                    dates = common_idx
                    ax_k1.plot(dates, beta, color='darkblue', label=f"Dynamic Beta ({TICKER}/{PAIR_TICKER})")
                    ax_k1.set_title("Kalman Estimated Hedge Ratio (Beta)")
                    ax_k1.legend()
                    format_plot_dates(ax_k1, dates)
                    spread_series = y - (alpha + beta * x)
                    z_score = (spread_series - np.mean(spread_series)) / np.std(spread_series)
                    ax_k2.plot(dates, z_score, color='purple', label="Spread Z-Score")
                    ax_k2.axhline(2.0, color='red', linestyle='--')
                    ax_k2.axhline(-2.0, color='green', linestyle='--')
                    ax_k2.set_title("Kalman Residual Z-Score")
                    ax_k2.legend()
                    format_plot_dates(ax_k2, dates)
                    st.pyplot(fig_k)
                    st.session_state.report_gen.add_plot("Kalman Pairs Analysis", fig_k)
                    st.write(f"Current Hedge Ratio: **{beta[-1]:.4f}**")
                else:
                    st.error("Not enough overlapping data.")
            else:
                st.error(f"Could not load data for {PAIR_TICKER}")

        elif kf_mode == "Single Asset (Trend)":
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                proc_noise = st.select_slider("Trend Flexibility", options=[1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)
            with col_k2:
                meas_noise = st.select_slider("Noise Tolerance", options=[1e-3, 1e-2, 1e-1, 1.0], value=1e-2)
            with col_k3:
                model_mode = st.radio("Model Type", ["Smoothed (New)", "Standard (Old)", "Compare Both"])

            prices = df_main['Close'].values
            kf_trend = KalmanFilterTrend(process_noise=proc_noise, measurement_noise=meas_noise)

            if model_mode == "Standard (Old)":
                est_trend, _ = kf_trend.filter(prices)
                label_trend = "Kalman Trend (Standard)"
                color_trend = "blue"
            elif model_mode == "Smoothed (New)":
                est_trend, _ = kf_trend.smooth(prices)
                label_trend = "Kalman Trend (Smoothed)"
                color_trend = "purple"
            else:
                est_trend_smooth, _ = kf_trend.smooth(prices)
                est_trend_std, _ = kf_trend.filter(prices)

            fig_kt, ax_kt = plt.subplots(figsize=(12, 6))
            ax_kt.plot(df_main.index, prices, color='gray', alpha=0.5, label='Actual Price')
            if model_mode == "Compare Both":
                ax_kt.plot(df_main.index, est_trend_std, color='blue', linewidth=1.5, linestyle='--', label='Standard (Causal)')
                ax_kt.plot(df_main.index, est_trend_smooth, color='purple', linewidth=2, label='Smoothed (RTS)')
                current_trend = est_trend_smooth[-1]
            else:
                ax_kt.plot(df_main.index, est_trend, color=color_trend, linewidth=2, label=label_trend)
                current_trend = est_trend[-1]

            ax_kt.set_title(f"Kalman Filter Trend: {TICKER}")
            ax_kt.legend()
            format_plot_dates(ax_kt, df_main.index)
            from matplotlib.ticker import MaxNLocator
            ax_kt.yaxis.set_major_locator(MaxNLocator(nbins=15))
            ax_kt.grid(True, which='major', linestyle='--', alpha=0.5)
            st.pyplot(fig_kt)
            st.session_state.report_gen.add_plot("Kalman Trend Analysis", fig_kt)

            current_price = prices[-1]
            diff_pct = (current_price - current_trend) / current_trend * 100
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Current Price", f"{CURRENCY}{current_price:.2f}")
            with c2: st.metric("Current Trend", f"{CURRENCY}{current_trend:.2f}")
            with c3: st.metric("Deviation", f"{diff_pct:.2f}%", delta=f"{diff_pct:.2f}%", delta_color="inverse")
            if diff_pct > 5.0: st.warning("Price significantly ABOVE Trend (Potential Overbought)")
            elif diff_pct < -5.0: st.success("Price significantly BELOW Trend (Potential Oversold)")
            else: st.info("Price near Trend (Neutral)")

# ==========================================
# TAB 5: MACRO FACTORS
# ==========================================
with tab5:
    if df_main is None:
        st.warning("Please load a ticker to view Factor analysis.")
    else:
        st.write("### Macro Factor Sensitivity")

    if df_main is not None:
        macro_tickers = {
            'Crude Oil': 'CL=F', 'Gold': 'GC=F', '10Y Yield': '^TNX',
            'US Dollar': 'DX-Y.NYB', 'S&P 500': '^GSPC'
        }
        macro_data = {}
        for name, sym in macro_tickers.items():
            m_df = load_data(sym, start_date, end_date)
            if m_df is not None:
                macro_data[name] = m_df['Returns']
        macro_data[TICKER] = df_main['Returns']
        df_macro = pd.DataFrame(macro_data).dropna()
        if not df_macro.empty:
            corr_matrix = df_macro.corr()
            fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
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
            st.session_state.report_gen.add_plot("Macro Correlations", fig_hm)
            st.session_state.report_gen.add_data("Correlation Matrix", corr_matrix)
            oil_corr = corr_matrix.loc[TICKER, 'Crude Oil']
            rate_corr = corr_matrix.loc[TICKER, '10Y Yield']
            if oil_corr > 0.3: st.success(f"High correlation with Energy ({oil_corr:.2f}).")
            elif oil_corr < -0.3: st.info(f"Inverse correlation with Energy ({oil_corr:.2f}).")
            else: st.warning(f"Low sensitivity to Energy prices ({oil_corr:.2f}).")

# ==========================================
# TAB 6: STRUCTURAL
# ==========================================
with tab6:
    if df_main is None:
        st.warning("Please load a ticker to view Structural Decomposition.")
    else:
        st.write("### Structural Decomposition")

    if df_main is not None:
        period = st.selectbox("Seasonality Period", [5, 21, 63, 252], index=1)
        if len(df_main) > period * 2:
            decomp = seasonal_decompose(df_main['Close'], model='multiplicative', period=period)
            fig_dec, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            decomp.trend.plot(ax=ax1, title='Trend')
            decomp.seasonal.plot(ax=ax2, title='Seasonal Component')
            decomp.resid.plot(ax=ax3, title='Residuals')
            format_plot_dates(ax3, df_main.index)
            st.pyplot(fig_dec)
            st.session_state.report_gen.add_plot("Structural Decomposition", fig_dec)
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

    if df_main is not None:
        strategy_type = st.radio(
            "Select Strategy",
            [
                "Regime Switching (Trend Following)",
                "Kalman Filter (Trend Crossover)",
                "Momentum Hedge (EMA/SMA Cross)",
                "MAD Trend Modes",
                "Dual MA Cross",
                "Ehlers SuperSmoother",
                "Ehlers Simple Decycler"
            ],
            horizontal=True
        )

        col_b3 = st.container()
        with col_b3:
            default_start = datetime.now() - timedelta(days=365)
            bt_start_date = st.date_input("Backtest Start", default_start)
            bt_end_date = st.date_input("Backtest End", datetime.now())

        if live_mode:
            df_bt = df_main
        else:
            if bt_start_date >= bt_end_date:
                st.error("Start date must be before end date.")
                st.stop()
            df_bt = load_data(TICKER, bt_start_date, bt_end_date, interval='1d')

        if df_bt is None or df_bt.empty:
            st.error("Could not load data for backtest.")
            st.stop()

        returns_bt = df_bt['Returns']
        prices_bt = df_bt['Close']
        model_data_bt = returns_bt.dropna() * 100
        signals = None
        strat_prices = prices_bt

        # ─────────────────────────────────────────────
        # STRATEGY: Regime Switching
        # ─────────────────────────────────────────────
        if strategy_type == "Regime Switching (Trend Following)":
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

            signal_method = st.radio("Signal Method", [
                "Regime Weighted Expected Return",
                "Regime Probability",
                "Regime Switching Period"
            ], horizontal=True)

            st.info("💡 **Pro Tip**: Blue-chips often favor **2 states**. High-beta tech often favors **3 states**.")

            if st.button("📊 Compare Regime Fitness (N=2,3,4)", use_container_width=True):
                with st.spinner("Analyzing model complexity..."):
                    comp_results = []
                    loc_prices = prices_bt.resample('W').last().dropna() if bt_freq == "Weekly" else prices_bt
                    loc_returns = loc_prices.pct_change().dropna()
                    if bt_stability > 0:
                        loc_model_data = loc_returns.ewm(span=bt_stability, adjust=False).mean().dropna() * 100
                    else:
                        loc_model_data = loc_returns.dropna() * 100

                    for n in [2, 3, 4]:
                        r = fit_regime_model(loc_model_data, n, bt_switch_vol, bt_switch_trend)
                        if r:
                            r_means = []
                            for i in range(n):
                                m_val = r.params[f'const[{i}]'] if f'const[{i}]' in r.params else r.params.get('const', 0.0)
                                r_means.append((i, m_val))
                            bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
                            p_df = r.filtered_marginal_probabilities
                            if signal_method == "Regime Weighted Expected Return":
                                e_ret = pd.Series(0.0, index=loc_model_data.index)
                                for i in range(n):
                                    m = r.params[f'const[{i}]'] if f'const[{i}]' in r.params else r.params.get('const', 0.0)
                                    e_ret += p_df.iloc[:, i] * m
                                sigs = (e_ret > 0).astype(int)
                            else:
                                dom = p_df.idxmax(axis=1)
                                sigs = (dom == bull_idx).astype(int)
                            common_idx = loc_prices.index.intersection(sigs.index)
                            bt_res = BacktestEngine.run_strategy(loc_prices.loc[common_idx], sigs.loc[common_idx], initial_cap, trailing_stop)
                            comp_results.append({
                                "Regimes": n, "AIC": r.aic, "BIC": r.bic,
                                "Total Return %": (bt_res['equity_curve'].iloc[-1] / initial_cap - 1) * 100
                            })

                    if comp_results:
                        comp_df = pd.DataFrame(comp_results)
                        best_bic = comp_df.loc[comp_df['BIC'].idxmin(), 'Regimes']
                        best_pnl = comp_df.loc[comp_df['Total Return %'].idxmax(), 'Regimes']
                        st.table(comp_df.style.highlight_min(subset=['AIC', 'BIC'], color='lightgreen')
                                            .highlight_max(subset=['Total Return %'], color='lightgreen'))
                        c_fit, c_perf = st.columns(2)
                        with c_fit: st.success(f"⚖️ **Robustness**: {best_bic} Regimes (Best BIC)")
                        with c_perf: st.success(f"🚀 **Performance**: {best_pnl} Regimes (Best PnL)")

            if bt_freq == "Weekly":
                prices_bt_resampled = prices_bt.resample('W').last().dropna()
                returns_bt_resampled = prices_bt_resampled.pct_change().dropna()
                strat_prices = prices_bt_resampled
            else:
                prices_bt_resampled = prices_bt
                returns_bt_resampled = returns_bt

            if bt_stability > 0:
                model_data_bt = returns_bt_resampled.ewm(span=bt_stability, adjust=False).mean().dropna() * 100
            else:
                model_data_bt = returns_bt_resampled.dropna() * 100

            if len(model_data_bt) > 5:
                model_data_bt = pd.Series(model_data_bt.values.flatten().astype(float), index=model_data_bt.index)

            if len(model_data_bt) < 10:
                st.error(f"❌ Insufficient data ({len(model_data_bt)} points). Increase date range.")
            else:
                with st.spinner("Fitting Regime Model..."):
                    res_bt = fit_regime_model(model_data_bt, bt_n_regimes, bt_switch_vol, bt_switch_trend)

                if res_bt:
                    fit_col1, fit_col2 = st.columns(2)
                    with fit_col1: st.caption(f"Model Fitness (AIC): **{res_bt.aic:.1f}**")
                    with fit_col2: st.caption(f"Model Fitness (BIC): **{res_bt.bic:.1f}**")

                    regime_means = []
                    for i in range(bt_n_regimes):
                        if f'const[{i}]' in res_bt.params:
                            mean_val = res_bt.params[f'const[{i}]']
                        else:
                            mean_val = res_bt.params.get('const', 0.0)
                        regime_means.append((i, mean_val))
                    sorted_regimes = sorted(regime_means, key=lambda x: x[1], reverse=True)
                    bull_regime_idx = sorted_regimes[0][0]
                    probs_df = res_bt.filtered_marginal_probabilities

                    if signal_method == "Regime Weighted Expected Return":
                        expected_ret = pd.Series(0.0, index=model_data_bt.index)
                        for i in range(bt_n_regimes):
                            if f'const[{i}]' in res_bt.params:
                                mean_val = res_bt.params[f'const[{i}]']
                            else:
                                mean_val = res_bt.params.get('const', 0.0)
                            prob = probs_df.iloc[:, i]
                            expected_ret += prob * mean_val
                        common_idx = strat_prices.index.intersection(expected_ret.index)
                        expected_ret = expected_ret.loc[common_idx]
                        signals = (expected_ret > 0).astype(int)
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            ax_ctx.plot(expected_ret.index, expected_ret, color='purple', label='Expected Return')
                            ax_ctx.axhline(0, color='black', linestyle='--', linewidth=1)
                            ax_ctx.fill_between(expected_ret.index, 0, expected_ret, where=(expected_ret>0), color='green', alpha=0.3)
                            ax_ctx.fill_between(expected_ret.index, 0, expected_ret, where=(expected_ret<0), color='red', alpha=0.3)
                            format_plot_dates(ax_ctx, expected_ret.index)
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    elif signal_method == "Regime Probability":
                        bull_probs = probs_df.iloc[:, bull_regime_idx]
                        dominant_regime = probs_df.idxmax(axis=1)
                        common_idx = strat_prices.index.intersection(dominant_regime.index)
                        dominant_regime = dominant_regime.loc[common_idx]
                        bull_probs = bull_probs.loc[common_idx]
                        signals = (dominant_regime == bull_regime_idx).astype(int)
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            ax_ctx.plot(bull_probs.index, bull_probs, color='green', label='Bull Probability')
                            ax_ctx.fill_between(bull_probs.index, 0, 1, where=(signals==1), color='green', alpha=0.1)
                            format_plot_dates(ax_ctx, bull_probs.index)
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    else:
                        dominant_regime = probs_df.idxmax(axis=1)
                        common_idx = strat_prices.index.intersection(dominant_regime.index)
                        dominant_regime = dominant_regime.loc[common_idx]
                        signals = (dominant_regime == bull_regime_idx).astype(int)
                        with st.expander("See Strategy Context"):
                            fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                            strat_prices_aligned = strat_prices.loc[common_idx]
                            ax_ctx.plot(strat_prices_aligned.index, strat_prices_aligned, color='gray', alpha=0.5)
                            ax_ctx.fill_between(strat_prices_aligned.index, strat_prices_aligned.min(), strat_prices_aligned.max(),
                                                where=(signals==1), color='green', alpha=0.2, label='Bull Period')
                            format_plot_dates(ax_ctx, strat_prices_aligned.index)
                            ax_ctx.legend()
                            st.pyplot(fig_ctx)

                    with st.expander("🔍 Debug: Signal Details"):
                        debug_df = pd.DataFrame({"Price": strat_prices, "Signal": signals}).dropna()
                        st.dataframe(debug_df.style.format({"Price": "{:.2f}", "Signal": "{:.0f}"}), use_container_width=True)
                else:
                    st.error("Regime model fitting failed.")

        # ─────────────────────────────────────────────
        # STRATEGY: Kalman Filter
        # ─────────────────────────────────────────────
        elif strategy_type == "Kalman Filter (Trend Crossover)":
            st.markdown("**Strategy:** Long when Price crosses **ABOVE** Kalman Trend. Sell when Price crosses **BELOW**.")
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                kf_noise = st.select_slider("Trend Sensitivity", options=[1e-5, 1e-4, 1e-3], value=1e-4)
            with col_k2:
                confirm_days = st.slider("Signal Confirmation (Days)", 1, 5, 1)

            with st.spinner("Running Kalman Filter..."):
                kf = KalmanFilterTrend(process_noise=kf_noise, measurement_noise=1e-2)
                trend_est, _ = kf.filter(prices_bt.values)
                trend_series = pd.Series(trend_est, index=prices_bt.index)
                sig_list = []
                position = 0
                days_above = 0
                days_below = 0
                for price, trend in zip(prices_bt, trend_series):
                    if price > trend:
                        days_above += 1
                        days_below = 0
                    else:
                        days_below += 1
                        days_above = 0
                    if position == 0:
                        if days_above >= confirm_days:
                            position = 1
                    elif position == 1:
                        if days_below >= confirm_days:
                            position = 0
                    sig_list.append(position)
                signals = pd.Series(sig_list, index=prices_bt.index)
                with st.expander("See Strategy Context"):
                    fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                    ax_ctx.plot(prices_bt.index, prices_bt, color='gray', alpha=0.5, label='Price')
                    ax_ctx.plot(trend_series.index, trend_series, color='blue', label='Kalman Trend')
                    ax_ctx.fill_between(trend_series.index, prices_bt.min(), prices_bt.max(),
                                        where=(signals==1), color='green', alpha=0.1, label='Long Zone')
                    format_plot_dates(ax_ctx, prices_bt.index)
                    ax_ctx.legend()
                    st.pyplot(fig_ctx)

        # ─────────────────────────────────────────────
        # STRATEGY: Momentum Hedge
        # ─────────────────────────────────────────────
        elif strategy_type == "Momentum Hedge (EMA/SMA Cross)":
            st.markdown("**Strategy:** Long when **Short EMA > Med SMA**. Cash/Hedge otherwise.")
            c_h1, c_h2 = st.columns(2)
            with c_h1:
                short_len = st.slider("Short EMA Length", 5, 50, 20)
            with c_h2:
                med_len = st.slider("Medium SMA Length", 20, 200, 60)

            with st.spinner("Calculating Momentum Hedge Signals..."):
                short_ema = prices_bt.ewm(span=short_len, adjust=False).mean()
                med_sma = prices_bt.rolling(window=med_len).mean()
                signals = (short_ema >= med_sma).astype(int)
                with st.expander("See Strategy Context"):
                    fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                    ax_ctx.plot(prices_bt.index, prices_bt, color='gray', alpha=0.5, label='Price')
                    ax_ctx.plot(short_ema.index, short_ema, color='orange', linewidth=1.5, label=f'Short EMA ({short_len})')
                    ax_ctx.plot(med_sma.index, med_sma, color='blue', linewidth=1.5, label=f'Med SMA ({med_len})')
                    ax_ctx.fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                        where=(signals==1), color='green', alpha=0.1, label='Long Zone')
                    ax_ctx.fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                        where=(signals==0), color='red', alpha=0.1, label='Hedge Zone')
                    format_plot_dates(ax_ctx, prices_bt.index)
                    ax_ctx.legend()
                    st.pyplot(fig_ctx)

        # ─────────────────────────────────────────────
        # STRATEGY: MAD Trend Modes
        # ─────────────────────────────────────────────
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
                with col_bb1: mult_p = st.number_input("+ Multiplier", 0.1, 5.0, 1.4)
                with col_bb2: mult_n = st.number_input("- Multiplier", 0.1, 5.0, 1.0)
                mad_params.update({'bb_mult_p': mult_p, 'bb_mult_n': mult_n})
            elif sig_mode == "For Loop":
                col_fl1, col_fl2, col_fl3 = st.columns(3)
                with col_fl1: fl_a = st.number_input("From", 1, 100, 10)
                with col_fl2: fl_b = st.number_input("To", 1, 200, 60)
                with col_fl3: fl_len = st.number_input("Loop MA Length", 1, 50, 10)
                col_fl4, col_fl5 = st.columns(2)
                with col_fl4: thresh_l = st.number_input("Threshold Long", 1, 100, 23)
                with col_fl5: thresh_s = st.number_input("Threshold Short", -100, 100, 3)
                mad_params.update({
                    'fl_ma_type': ma_type, 'fl_len': fl_len,
                    'fl_a': fl_a, 'fl_b': fl_b,
                    'fl_thresh_l': thresh_l, 'fl_thresh_s': thresh_s
                })
            else:
                col_c1, col_c2 = st.columns(2)
                with col_c1: c_thresh_l = st.number_input("Threshold Long (Combined)", -1.0, 1.0, 0.0, step=0.01)
                with col_c2: c_thresh_s = st.number_input("Threshold Short (Combined)", -1.0, 1.0, 0.0, step=0.01)
                mad_params.update({'c_thresh_l': c_thresh_l, 'c_thresh_s': c_thresh_s})

            if st.button("🚀 Run MAD Backtest", use_container_width=True):
                with st.spinner("Generating MAD Trend signals..."):
                    signals = MADTrendModes.get_signals(df_bt, mad_params)
                    with st.expander("See Strategy Context", expanded=True):
                        fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                        ax_ctx.plot(prices_bt.index, prices_bt, color='gray', alpha=0.5, label='Price')
                        ax_ctx.fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                            where=(signals==1), color='green', alpha=0.1, label='Long Zone')
                        format_plot_dates(ax_ctx, prices_bt.index)
                        ax_ctx.set_title(f"MAD Trend Modes Signal ({sig_mode})")
                        ax_ctx.legend()
                        st.pyplot(fig_ctx)

        # ─────────────────────────────────────────────
        # STRATEGY: Dual MA Cross
        # ─────────────────────────────────────────────
        elif strategy_type == "Dual MA Cross":
            st.markdown("### 🔀 Dual Moving Average Cross Settings")
            ma_options = ["SMA", "EMA", "WMA", "HMA", "RMA", "ALMA", "LSMA"]
            c_ma1, c_ma2 = st.columns(2)
            with c_ma1:
                st.subheader("Fast MA (Short-term)")
                f_ma_type = st.selectbox("Fast MA Type", ma_options, index=1)
                f_ma_len = st.number_input("Fast MA Length", 1, 250, 20)
            with c_ma2:
                st.subheader("Slow MA (Long-term)")
                s_ma_type = st.selectbox("Slow MA Type", ma_options, index=0)
                s_ma_len = st.number_input("Slow MA Length", 1, 250, 50)

            if st.button("🚀 Run Dual MA Backtest", use_container_width=True):
                if f_ma_len >= s_ma_len:
                    st.warning("Fast MA length is typically shorter than Slow MA length.")
                with st.spinner("Calculating Moving Average Crossovers..."):
                    fast_ma = MADTrendModes.ma_switch(prices_bt, f_ma_len, f_ma_type)
                    slow_ma = MADTrendModes.ma_switch(prices_bt, s_ma_len, s_ma_type)
                    long_cond = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
                    short_cond = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

                    def get_stateful_ma_signal(l_cond, s_cond, index):
                        sig = pd.Series(np.nan, index=index)
                        sig.loc[l_cond] = 1
                        sig.loc[s_cond] = 0
                        return sig.ffill().fillna(0)

                    signals = get_stateful_ma_signal(long_cond, short_cond, prices_bt.index)
                    with st.expander("See Strategy Context", expanded=True):
                        fig_ctx, ax_ctx = plt.subplots(figsize=(10, 4))
                        ax_ctx.plot(prices_bt.index, prices_bt, color='gray', alpha=0.5, label='Price')
                        ax_ctx.plot(fast_ma.index, fast_ma, label=f'Fast {f_ma_type} ({f_ma_len})', color='orange', alpha=0.8)
                        ax_ctx.plot(slow_ma.index, slow_ma, label=f'Slow {s_ma_type} ({s_ma_len})', color='blue', alpha=0.8)
                        ax_ctx.fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                            where=(signals==1), color='green', alpha=0.1, label='Long Zone')
                        format_plot_dates(ax_ctx, prices_bt.index)
                        ax_ctx.set_title(f"Dual MA Cross: {f_ma_type}({f_ma_len}) / {s_ma_type}({s_ma_len})")
                        ax_ctx.legend()
                        st.pyplot(fig_ctx)

        # ─────────────────────────────────────────────
        # STRATEGY: Ehlers SuperSmoother ← NEW
        # ─────────────────────────────────────────────
        elif strategy_type == "Ehlers SuperSmoother":
            st.markdown("### 🌀 Ehlers SuperSmoother Filter (Original 2-Pole)")
            st.caption(
                "John Ehlers' original SuperSmoother from *Cybernetic Analysis for Stocks and Futures* (2004). "
                "Eliminates aliasing noise below the Nyquist frequency using a 2-pole recursive IIR filter. "
                "Unlike simple MAs, it has minimal lag while aggressively attenuating high-frequency noise."
            )

            col_ss1, col_ss2 = st.columns(2)
            with col_ss1:
                ss_period_choice = st.selectbox(
                    "SuperSmoother Period (Preset)",
                    [20, 50, 100, 150, 200, 250],
                    index=1,
                    help="Cutoff period. Higher = smoother but more lag. Common: 50, 100, 200."
                )
                ss_custom_period = st.number_input(
                    "Custom Period Override (overrides preset if changed)",
                    min_value=5, max_value=500, value=int(ss_period_choice),
                    help="Enter any integer period. This value takes priority over the preset above."
                )
                final_ss_period = int(ss_custom_period)

            with col_ss2:
                ss_signal_type = st.selectbox(
                    "Signal Generation Method",
                    [
                        "Price vs SuperSmoother (Crossover)",
                        "SuperSmoother Slope (Direction)",
                        "Dual SuperSmoother Cross (Fast/Slow)"
                    ],
                    help=(
                        "Crossover: Long when price is above the filter. "
                        "Slope: Long when filter is rising. "
                        "Dual: Long when fast SS is above slow SS."
                    )
                )

            # Extra controls for dual SS
            ss_fast_period = None
            ss_slow_period = None
            if ss_signal_type == "Dual SuperSmoother Cross (Fast/Slow)":
                col_ss3, col_ss4 = st.columns(2)
                with col_ss3:
                    ss_fast_period = st.number_input(
                        "Fast SS Period", min_value=5, max_value=200,
                        value=max(5, final_ss_period // 2),
                        help="Short-term SuperSmoother (triggers first)"
                    )
                with col_ss4:
                    ss_slow_period = st.number_input(
                        "Slow SS Period", min_value=10, max_value=500,
                        value=final_ss_period,
                        help="Long-term SuperSmoother (acts as trend filter)"
                    )

            # Info box about the math
            with st.expander("📐 How the Ehlers SuperSmoother Works"):
                st.markdown("""
**2-Pole Recursive IIR Filter Coefficients:**
```
a1 = exp(-1.414 × π / Period)
b1 = 2 × a1 × cos(1.414 × 180° / Period)
c2 = b1
c3 = -a1²
c1 = 1 - c2 - c3
```
**Recursive Formula:**
```
SS[i] = c1 × (Price[i] + Price[i-1]) / 2  +  c2 × SS[i-1]  +  c3 × SS[i-2]
```
- **Period** controls the cutoff frequency — frequencies above Nyquist (Period/2) are eliminated
- **Two-pole** design provides steeper roll-off than a single-pole (EMA-like) filter
- The filter is **causal** (no look-ahead bias) — safe for backtesting
                """)

            if st.button("🚀 Run SuperSmoother Backtest", use_container_width=True, type="primary"):
                with st.spinner(f"Applying Ehlers SuperSmoother (Period={final_ss_period})..."):

                    # Compute main SS
                    ss_main = MADTrendModes.ehlers_supersmoother(prices_bt, final_ss_period)

                    if ss_signal_type == "Price vs SuperSmoother (Crossover)":
                        # Stateful crossover: Long when price crosses above SS, cash when below
                        long_cond  = (prices_bt > ss_main) & (prices_bt.shift(1) <= ss_main.shift(1))
                        short_cond = (prices_bt < ss_main) & (prices_bt.shift(1) >= ss_main.shift(1))
                        sig_raw = pd.Series(np.nan, index=prices_bt.index)
                        sig_raw.loc[long_cond]  = 1
                        sig_raw.loc[short_cond] = 0
                        signals = sig_raw.ffill().fillna(0).astype(int)

                    elif ss_signal_type == "SuperSmoother Slope (Direction)":
                        # Long when SS is rising (positive slope), cash when falling
                        slope = ss_main.diff()
                        signals = (slope > 0).astype(int)

                    else:  # Dual SuperSmoother Cross
                        ss_fast_series = MADTrendModes.ehlers_supersmoother(prices_bt, int(ss_fast_period))
                        ss_slow_series = MADTrendModes.ehlers_supersmoother(prices_bt, int(ss_slow_period))
                        long_cond  = (ss_fast_series > ss_slow_series) & (ss_fast_series.shift(1) <= ss_slow_series.shift(1))
                        short_cond = (ss_fast_series < ss_slow_series) & (ss_fast_series.shift(1) >= ss_slow_series.shift(1))
                        sig_raw = pd.Series(np.nan, index=prices_bt.index)
                        sig_raw.loc[long_cond]  = 1
                        sig_raw.loc[short_cond] = 0
                        signals = sig_raw.ffill().fillna(0).astype(int)

                    # ── Strategy Context Chart ──────────────────────────────
                    with st.expander("📊 See Strategy Context", expanded=True):
                        fig_ctx, axes_ss = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                                         gridspec_kw={'height_ratios': [3, 1]})

                        # Panel 1: Price + SS lines
                        axes_ss[0].plot(prices_bt.index, prices_bt, color='gray', alpha=0.55,
                                        linewidth=1, label='Price')
                        axes_ss[0].plot(ss_main.index, ss_main, color='royalblue', linewidth=2.2,
                                        label=f'SuperSmoother ({final_ss_period})')

                        if ss_signal_type == "Dual SuperSmoother Cross (Fast/Slow)":
                            axes_ss[0].plot(ss_fast_series.index, ss_fast_series,
                                            color='darkorange', linewidth=1.6, linestyle='--',
                                            label=f'Fast SS ({int(ss_fast_period)})')
                            axes_ss[0].plot(ss_slow_series.index, ss_slow_series,
                                            color='darkgreen', linewidth=1.6, linestyle=':',
                                            label=f'Slow SS ({int(ss_slow_period)})')

                        axes_ss[0].fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                                where=(signals == 1), color='limegreen', alpha=0.12, label='Long Zone')
                        axes_ss[0].fill_between(prices_bt.index, prices_bt.min(), prices_bt.max(),
                                                where=(signals == 0), color='tomato', alpha=0.07, label='Cash Zone')
                        axes_ss[0].set_title(f"Ehlers SuperSmoother — {ss_signal_type} | {TICKER}")
                        axes_ss[0].legend(fontsize='small', loc='upper left')
                        axes_ss[0].set_ylabel("Price")

                        # Panel 2: Signal bar
                        axes_ss[1].fill_between(prices_bt.index, 0, signals,
                                                color='limegreen', alpha=0.5, step='pre', label='Long=1 / Cash=0')
                        axes_ss[1].set_ylim(-0.1, 1.4)
                        axes_ss[1].set_yticks([0, 1])
                        axes_ss[1].set_yticklabels(['Cash', 'Long'])
                        axes_ss[1].set_title("Signal")
                        axes_ss[1].legend(fontsize='small')

                        format_plot_dates(axes_ss[1], prices_bt.index)
                        plt.tight_layout()
                        st.pyplot(fig_ctx)

                    # Filter comparison chart (optional insight)
                    with st.expander("🔬 Filter Comparison: SS vs EMA vs SMA"):
                        fig_cmp, ax_cmp = plt.subplots(figsize=(13, 5))
                        ax_cmp.plot(prices_bt.index, prices_bt, color='gray', alpha=0.4, linewidth=0.8, label='Price')
                        ax_cmp.plot(ss_main.index, ss_main, color='royalblue', linewidth=2, label=f'SuperSmoother ({final_ss_period})')
                        ema_cmp = prices_bt.ewm(span=final_ss_period, adjust=False).mean()
                        sma_cmp = prices_bt.rolling(window=final_ss_period).mean()
                        ax_cmp.plot(ema_cmp.index, ema_cmp, color='orange', linewidth=1.5, linestyle='--', label=f'EMA ({final_ss_period})')
                        ax_cmp.plot(sma_cmp.index, sma_cmp, color='green', linewidth=1.5, linestyle=':', label=f'SMA ({final_ss_period})')
                        ax_cmp.set_title(f"SuperSmoother vs Traditional MAs (Period={final_ss_period})")
                        ax_cmp.legend(fontsize='small')
                        format_plot_dates(ax_cmp, prices_bt.index)
                        st.pyplot(fig_cmp)
                        st.caption(
                            "SuperSmoother typically tracks the trend more closely than SMA and has less lag than SMA "
                            "while still suppressing noise more aggressively than EMA."
                        )

        # ─────────────────────────────────────────────
        # STRATEGY: Ehlers Simple Decycler ← NEW
        # ─────────────────────────────────────────────
        elif strategy_type == "Ehlers Simple Decycler":
            st.markdown("### 〰️ Ehlers Simple Decycler")
            st.caption(
                "From John Ehlers' *Cycle Analytics for Traders* (2013). "
                "Isolates the trend by subtracting a High-Pass filter from price: **Decycler = Price − HP**. "
                "The orange line in the reference chart above is exactly this output."
            )

            col_dc1, col_dc2 = st.columns(2)
            with col_dc1:
                dc_hp_period = st.selectbox(
                    "High-Pass Period (Cycle Cutoff)",
                    [20, 30, 50, 60, 89, 100, 125, 150, 200],
                    index=6,   # default 125 — matches the TradingView screenshot
                    help="Controls which cycles are removed. "
                         "Higher = smoother, removes slower cycles too. "
                         "125 matches the TradingView reference chart."
                )
                dc_custom = st.number_input(
                    "Custom HP Period (overrides preset)",
                    min_value=5, max_value=500, value=int(dc_hp_period)
                )
                final_dc_period = int(dc_custom)

            with col_dc2:
                dc_signal_mode = st.selectbox(
                    "Signal Generation Mode",
                    [
                        "Price vs Decycler (Crossover)",
                        "Decycler Slope (Direction)",
                        "Dual Decycler Cross (Fast/Slow)",
                        "Price Band (Upper/Lower Threshold)"
                    ],
                    help=(
                        "Crossover: Long when price crosses above Decycler line (matches TV indicator logic). "
                        "Slope: Long when Decycler is rising. "
                        "Dual: Fast vs Slow Decycler crossover. "
                        "Band: Uses % thresholds above/below Decycler."
                    )
                )

            # Extra controls per mode
            dc_fast_period = None
            dc_slow_period = None
            dc_upper_thresh = 0.5
            dc_lower_thresh = 0.5

            if dc_signal_mode == "Dual Decycler Cross (Fast/Slow)":
                col_dc3, col_dc4 = st.columns(2)
                with col_dc3:
                    dc_fast_period = st.number_input(
                        "Fast Decycler HP Period", min_value=5, max_value=200,
                        value=max(5, final_dc_period // 2),
                        help="Shorter period = reacts faster to cycle removal"
                    )
                with col_dc4:
                    dc_slow_period = st.number_input(
                        "Slow Decycler HP Period", min_value=10, max_value=500,
                        value=final_dc_period,
                        help="Longer period = smoother trend baseline"
                    )

            elif dc_signal_mode == "Price Band (Upper/Lower Threshold)":
                col_dc5, col_dc6 = st.columns(2)
                with col_dc5:
                    dc_upper_thresh = st.slider(
                        "Upper Band Threshold (%)", 0.0, 5.0, 0.5, step=0.1,
                        help="Buy when Decycler crosses above Price*(1 - upper%/100)"
                    )
                with col_dc6:
                    dc_lower_thresh = st.slider(
                        "Lower Band Threshold (%)", 0.0, 5.0, 0.5, step=0.1,
                        help="Sell when Decycler crosses below Price*(1 - lower%/100)"
                    )

            # How it works expander
            with st.expander("📐 How the Ehlers Simple Decycler Works"):
                st.markdown(f"""
**Step 1 — 1-Pole High-Pass Filter:**
```
angle  = 360 / HP_Period  (degrees)
alpha  = (cos(angle) + sin(angle) - 1) / cos(angle)

HP[i]  = (1 - alpha/2)² × (Price[i] - 2×Price[i-1] + Price[i-2])
        + 2×(1-alpha)×HP[i-1]
        - (1-alpha)²×HP[i-2]
```

**Step 2 — Subtract to get Decycler (trend):**
```
Decycler[i] = Price[i] - HP[i]
```

**Interpretation:**
- The HP filter extracts **short-cycle (high-frequency) noise**
- Subtracting it leaves the **trend (low-frequency) component**
- At HP Period = **{final_dc_period}**: cycles shorter than ~{final_dc_period} bars are removed
- The orange line in the TradingView chart is exactly this output
- **Buy** when price is above/crosses above the Decycler line
- **Sell** when price is below/crosses below the Decycler line
                """)

            if st.button("🚀 Run Decycler Backtest", use_container_width=True, type="primary"):
                with st.spinner(f"Computing Ehlers Simple Decycler (HP Period={final_dc_period})..."):

                    # ── Compute main decycler ────────────────────────────────────
                    dec_main, hp_main = MADTrendModes.ehlers_simple_decycler(
                        prices_bt, final_dc_period, dc_upper_thresh, dc_lower_thresh
                    )

                    # ── Generate signals based on mode ───────────────────────────
                    if dc_signal_mode == "Price vs Decycler (Crossover)":
                        # Long when price crosses ABOVE decycler; cash when crosses BELOW
                        long_cond  = (prices_bt > dec_main) & (prices_bt.shift(1) <= dec_main.shift(1))
                        short_cond = (prices_bt < dec_main) & (prices_bt.shift(1) >= dec_main.shift(1))
                        sig_raw = pd.Series(np.nan, index=prices_bt.index)
                        sig_raw.loc[long_cond]  = 1
                        sig_raw.loc[short_cond] = 0
                        signals = sig_raw.ffill().fillna(0).astype(int)

                    elif dc_signal_mode == "Decycler Slope (Direction)":
                        # Long when decycler is rising
                        slope = dec_main.diff()
                        signals = (slope > 0).astype(int)

                    elif dc_signal_mode == "Dual Decycler Cross (Fast/Slow)":
                        dec_fast, _ = MADTrendModes.ehlers_simple_decycler(prices_bt, int(dc_fast_period))
                        dec_slow, _ = MADTrendModes.ehlers_simple_decycler(prices_bt, int(dc_slow_period))
                        long_cond  = (dec_fast > dec_slow) & (dec_fast.shift(1) <= dec_slow.shift(1))
                        short_cond = (dec_fast < dec_slow) & (dec_fast.shift(1) >= dec_slow.shift(1))
                        sig_raw = pd.Series(np.nan, index=prices_bt.index)
                        sig_raw.loc[long_cond]  = 1
                        sig_raw.loc[short_cond] = 0
                        signals = sig_raw.ffill().fillna(0).astype(int)

                    else:  # Price Band
                        upper_band = prices_bt * (1.0 - dc_upper_thresh / 100.0)
                        lower_band = prices_bt * (1.0 - dc_lower_thresh / 100.0)
                        long_cond  = (dec_main > upper_band) & (dec_main.shift(1) <= upper_band.shift(1))
                        short_cond = (dec_main < lower_band) & (dec_main.shift(1) >= lower_band.shift(1))
                        sig_raw = pd.Series(np.nan, index=prices_bt.index)
                        sig_raw.loc[long_cond]  = 1
                        sig_raw.loc[short_cond] = 0
                        signals = sig_raw.ffill().fillna(0).astype(int)

                    # ── Context Chart (mirrors TV screenshot) ───────────────────
                    with st.expander("📊 Decycler Chart (TV-style)", expanded=True):
                        fig_dc, axes_dc = plt.subplots(
                            3, 1, figsize=(14, 9), sharex=True,
                            gridspec_kw={'height_ratios': [3.5, 1.2, 0.8]}
                        )

                        # ── Panel 1: Price + Decycler line ──────────────────────
                        # Candlestick-style: colour bars green/red by daily return
                        ret_colours = ['#26a69a' if r >= 0 else '#ef5350'
                                       for r in prices_bt.pct_change().fillna(0)]
                        axes_dc[0].bar(prices_bt.index, prices_bt, color=ret_colours,
                                       alpha=0.55, width=0.8, label='Price')
                        axes_dc[0].plot(dec_main.index, dec_main,
                                        color='#f5a623', linewidth=2.2,
                                        label=f'Decycler (HP={final_dc_period})')

                        if dc_signal_mode == "Dual Decycler Cross (Fast/Slow)":
                            axes_dc[0].plot(dec_fast.index, dec_fast,
                                            color='#00bcd4', linewidth=1.5, linestyle='--',
                                            label=f'Fast Decycler (HP={int(dc_fast_period)})')
                            axes_dc[0].plot(dec_slow.index, dec_slow,
                                            color='#ab47bc', linewidth=1.5, linestyle=':',
                                            label=f'Slow Decycler (HP={int(dc_slow_period)})')

                        if dc_signal_mode == "Price Band (Upper/Lower Threshold)":
                            axes_dc[0].plot(upper_band.index, upper_band,
                                            color='#4caf50', linewidth=1, linestyle='--', alpha=0.7, label='Upper Band')
                            axes_dc[0].plot(lower_band.index, lower_band,
                                            color='#f44336', linewidth=1, linestyle='--', alpha=0.7, label='Lower Band')

                        # Shade long/cash zones
                        axes_dc[0].fill_between(
                            prices_bt.index, prices_bt.min() * 0.98, prices_bt.max() * 1.02,
                            where=(signals == 1), color='#26a69a', alpha=0.10, label='Long Zone'
                        )
                        axes_dc[0].fill_between(
                            prices_bt.index, prices_bt.min() * 0.98, prices_bt.max() * 1.02,
                            where=(signals == 0), color='#ef5350', alpha=0.06, label='Cash Zone'
                        )

                        # Buy/Sell markers
                        sig_changes = signals.diff().fillna(0)
                        buy_dates  = prices_bt.index[sig_changes == 1]
                        sell_dates = prices_bt.index[sig_changes == -1]
                        axes_dc[0].scatter(buy_dates,  prices_bt.loc[buy_dates]  * 0.985,
                                           marker='^', color='#26a69a', s=80, zorder=5, label='Buy')
                        axes_dc[0].scatter(sell_dates, prices_bt.loc[sell_dates] * 1.015,
                                           marker='v', color='#ef5350',  s=80, zorder=5, label='Sell')

                        axes_dc[0].set_title(
                            f"Ehlers Simple Decycler — {dc_signal_mode} | {TICKER}",
                            fontsize=11
                        )
                        axes_dc[0].legend(fontsize='x-small', loc='upper left', ncol=3)
                        axes_dc[0].set_ylabel("Price")

                        # ── Panel 2: HP Filter (cycle component) ───────────────
                        axes_dc[1].plot(hp_main.index, hp_main,
                                        color='#7e57c2', linewidth=1.2, label='High-Pass (Cycle)')
                        axes_dc[1].axhline(0, color='white', linewidth=0.6, alpha=0.4)
                        axes_dc[1].fill_between(hp_main.index, 0, hp_main,
                                                where=(hp_main >= 0), color='#26a69a', alpha=0.25)
                        axes_dc[1].fill_between(hp_main.index, 0, hp_main,
                                                where=(hp_main < 0),  color='#ef5350',  alpha=0.25)
                        axes_dc[1].set_title("High-Pass Filter Output (Removed Cycle Component)", fontsize=9)
                        axes_dc[1].set_ylabel("HP Value")
                        axes_dc[1].legend(fontsize='x-small')

                        # ── Panel 3: Signal bar ─────────────────────────────────
                        axes_dc[2].fill_between(prices_bt.index, 0, signals,
                                                color='#26a69a', alpha=0.6, step='pre')
                        axes_dc[2].set_ylim(-0.1, 1.4)
                        axes_dc[2].set_yticks([0, 1])
                        axes_dc[2].set_yticklabels(['Cash', 'Long'], fontsize=8)
                        axes_dc[2].set_title("Strategy Signal", fontsize=9)

                        format_plot_dates(axes_dc[2], prices_bt.index)
                        plt.tight_layout(h_pad=0.5)
                        st.pyplot(fig_dc)

                    # ── Decycler vs SuperSmoother comparison ─────────────────────
                    with st.expander("🔬 Decycler vs SuperSmoother vs EMA Comparison"):
                        ss_cmp = MADTrendModes.ehlers_supersmoother(prices_bt, final_dc_period)
                        ema_cmp = prices_bt.ewm(span=final_dc_period, adjust=False).mean()

                        fig_cmp2, ax_cmp2 = plt.subplots(figsize=(13, 5))
                        ax_cmp2.plot(prices_bt.index, prices_bt, color='gray', alpha=0.35,
                                     linewidth=0.8, label='Price')
                        ax_cmp2.plot(dec_main.index, dec_main, color='#f5a623', linewidth=2.2,
                                     label=f'Decycler HP={final_dc_period} (this strategy)')
                        ax_cmp2.plot(ss_cmp.index, ss_cmp, color='royalblue', linewidth=1.8,
                                     linestyle='--', label=f'SuperSmoother ({final_dc_period})')
                        ax_cmp2.plot(ema_cmp.index, ema_cmp, color='#66bb6a', linewidth=1.5,
                                     linestyle=':', label=f'EMA ({final_dc_period})')
                        ax_cmp2.set_title(f"Filter Comparison at Period = {final_dc_period}")
                        ax_cmp2.legend(fontsize='small')
                        format_plot_dates(ax_cmp2, prices_bt.index)
                        st.pyplot(fig_cmp2)
                        st.caption(
                            "The Decycler and SuperSmoother both track the trend but via different mechanisms. "
                            "Decycler subtracts the cycle; SuperSmoother applies a 2-pole low-pass filter. "
                            "In practice both are similar in smoothness but Decycler reacts slightly faster to turns."
                        )

        # ─────────────────────────────────────────────
        # BACKTEST ENGINE — runs for all strategies
        # ─────────────────────────────────────────────
        if signals is not None:
            bt_results = BacktestEngine.run_strategy(strat_prices, signals, initial_cap, trailing_stop)

            last_sig = signals.iloc[-1]
            last_dt = signals.index[-1]
            st.divider()
            if last_sig == 1:
                st.success(f"🚀 **STRATEGY SIGNAL (LONG)** | Last Update: {last_dt} | Action: **HOLD LONG**")
            else:
                st.error(f"🛑 **STRATEGY SIGNAL (CASH)** | Last Update: {last_dt} | Action: **STAY IN CASH / HEDGE**")

            strat_metrics = BacktestEngine.calculate_metrics(bt_results['returns'], rf_rate)
            bench_metrics = BacktestEngine.calculate_metrics(strat_prices.pct_change().dropna(), rf_rate)

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

            st.write("#### 📈 Equity Curve")
            fig_bt, ax_bt = plt.subplots(figsize=(12, 6))
            ax_bt.plot(bt_results['equity_curve'], label=f'Strategy ({strategy_type})', color='green', linewidth=2)
            ax_bt.plot(bt_results['benchmark_curve'], label='Buy & Hold (Benchmark)', color='gray', linestyle='--', alpha=0.7)
            ax_bt.set_title(f"Strategy Performance: {TICKER}")
            ax_bt.legend()
            format_plot_dates(ax_bt, bt_results['equity_curve'].index)
            st.pyplot(fig_bt)
            st.session_state.report_gen.add_plot("Backtest Performance", fig_bt)
            st.session_state.report_gen.add_data("Backtest Metrics", strat_metrics)

            st.write("#### 📝 Trade Log")
            trades_df = bt_results['trades']
            if not trades_df.empty:
                trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date']).dt.date
                trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date']).apply(
                    lambda x: x.date() if pd.notnull(x) else "Open"
                )
                st.dataframe(trades_df.style.format({
                    "Buy Price": "{:.2f}", "Sell Price": "{:.2f}", "PnL (%)": "{:.2f}%"
                }), use_container_width=True)
            else:
                st.info("No closed trades generated by the strategy.")

# ==========================================
# TAB 8: VOLATILITY CLUSTERING
# ==========================================
with tab8:
    if df_main is None:
        st.warning("Please load a ticker to view Volatility Clustering.")
    else:
        st.write("### 🌩️ Volatility Clustering & Jump Analysis")

    if df_main is not None:
        returns_arr = df_main['Returns'].values
        rv = RealizedVolatility.realized_variance(returns_arr)
        hawkes = HawkesVolatility().fit(returns_arr)
        br = hawkes.branching_ratio()
        latest_rv = np.sqrt(rv)*np.sqrt(252)

        if jump_detected: st.error(f"🎯 **MODEL VERDICT**: Significant **JUMPS** detected.")
        elif br > 0.8: st.warning(f"🎯 **MODEL VERDICT**: High **Volatility Clustering** (Branching Ratio: {br:.2f}).")
        else: st.success(f"🎯 **MODEL VERDICT**: Volatility is **Stable**.")

        bv = RealizedVolatility.bipower_variation(returns_arr)
        jump_res = RealizedVolatility.jump_component(returns_arr)

        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1: st.metric("Total Volatility (RV)", f"{np.sqrt(rv)*np.sqrt(252):.2%}")
        with col_v2: st.metric("Continuous Vol (BV)", f"{np.sqrt(bv)*np.sqrt(252):.2%}")
        with col_v3:
            st.metric("Jump Ratio", f"{jump_res['jump_ratio']:.1%}")
            if jump_res['p_value'] < 0.05: st.error("Significant Jumps Detected")
            else: st.success("No Significant Jumps")

        st.divider()
        st.write("#### Self-Exciting Volatility (Hawkes Process)")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1: st.metric("Branching Ratio", f"{br:.2f}")
        with h_col2:
            hl = hawkes.half_life()
            st.metric("Vol Cluster Half-Life", f"{hl:.1f} days")
        with h_col3: st.metric("Baseline Intensity", f"{hawkes.mu:.4f}")

        if br > 0.9: st.warning("⚠️ Critical Instability: Volatility is self-reinforcing.")
        elif br > 0.5: st.info("Moderate Clustering: Recent shocks affect near-term future.")
        else: st.success("Stable: Volatility mean-reverts quickly.")

        st.session_state.report_gen.add_data("Volatility Clustering Metrics", {
            "RV": rv, "BV": bv, "Jump Ratio": jump_res['jump_ratio'],
            "Branching Ratio": br, "Half-Life": hl
        })

        fig_vol, ax_vol = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax_vol[0].plot(df_main.index, df_main['Returns'], color='gray', alpha=0.6, linewidth=0.8, label="Returns")
        ax_vol[0].set_title(f"{TICKER} Returns Series")
        ax_vol[0].legend(loc='upper right')
        squared_rets = df_main['Returns']**2
        ax_vol[1].plot(df_main.index, squared_rets, color='orange', alpha=0.8, linewidth=0.8, label="Squared Returns")
        threshold = squared_rets.mean() + 2 * squared_rets.std()
        ax_vol[1].axhline(threshold, color='red', linestyle='--', linewidth=0.8, label="2-Sigma Threshold")
        ax_vol[1].set_title("Volatility Clustering (Squared Returns)")
        ax_vol[1].legend(loc='upper right')
        format_plot_dates(ax_vol[1], df_main.index)
        st.pyplot(fig_vol)
        st.session_state.report_gen.add_plot("Volatility Clustering Visuals", fig_vol)

# ==========================================
# TAB 9: ADVANCED REGIME
# ==========================================
with tab9:
    if df_main is None:
        st.warning("Please load a ticker to view Advanced Regime diagnostics.")
    else:
        st.write("### 🧠 Pro Regime Detection (Multi-Factor)")

    if df_main is not None:
        if not SKLEARN_AVAILABLE:
            st.error("⚠️ `scikit-learn` library is missing.")
        elif pro_detector is None:
            st.info("🏛️ **Active Engine**: Markov Switching Model (High Accuracy)")
            st.success(f"Current State: **{regime_label}** ({regime_prob:.1%} confidence)")
            st.caption(f"Number of States: {regime_data.get('n_states', 'Unknown')}")
        else:
            m_col1, m_col2 = st.columns(2)
            with m_col1: st.metric("Regime Label", regime_label, f"{regime_prob:.1%} Confidence")
            with m_col2: st.metric("Model AIC", f"{pro_detector.metrics.get('aic', 0):.0f}")

            fig_pro, ax_pro = plt.subplots(figsize=(10, 4))
            probs = pro_detector.regimes['probs']
            labels_pro = [pro_detector.state_labels.get(i, f"State {i}") for i in range(probs.shape[1])]
            ax_pro.stackplot(df_main.index, probs.T, labels=labels_pro, alpha=0.6)
            ax_pro.legend(loc='upper left', fontsize='x-small')
            ax_pro.set_title("Multi-Factor Regime Probabilities")
            format_plot_dates(ax_pro, df_main.index)
            st.pyplot(fig_pro)
            st.session_state.report_gen.add_plot("Institutional Regime Probabilities", fig_pro)

            feat_df = pd.DataFrame(pro_detector.features, columns=['Momentum', 'Vol_Z', 'Trend_Dev'], index=df_main.index)
            fig_feat, ax_feat = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            feat_df['Momentum'].plot(ax=ax_feat[0], title='Feature 1: Momentum', color='blue', alpha=0.7)
            feat_df['Vol_Z'].plot(ax=ax_feat[1], title='Feature 2: Volatility (Z-Score)', color='orange', alpha=0.7)
            feat_df['Trend_Dev'].plot(ax=ax_feat[2], title='Feature 3: Structural Dev', color='green', alpha=0.7)
            for a in ax_feat: a.axhline(0, color='black', lw=0.5)
            format_plot_dates(ax_feat[2], df_main.index)
            plt.tight_layout()
            st.pyplot(fig_feat)

# ==========================================
# TAB 10: SML & ALPHA
# ==========================================
with tab10:
    if df_main is None:
        st.warning("Please load a ticker to view SML/Alpha analysis.")
    else:
        st.write("### 📐 Securities Market Line (SML) & Alpha Analysis")

    if df_main is not None:
        col_sml1, col_sml2 = st.columns(2)
        with col_sml1:
            bench_ticker = st.selectbox("Benchmark Index", ["SPY", "QQQ", "IWM", "VT", "^NSEI"] if market_region != "Indian Market (INR)" else ["^NSEI", "^NSEBANK", "SPY"])
        with col_sml2:
            roll_win = st.slider("Rolling Window (Days)", 30, 252, 90)

        if st.button("Run Alpha Analysis"):
            with st.spinner(f"Calibrating CAPM against {bench_ticker}..."):
                df_bench = load_data(bench_ticker, start_date, end_date)
                if df_bench is not None:
                    analyzer = SMLAnalyzer(df_main['Returns'], df_bench['Returns'], rf_annual=rf_rate)
                    res_sml = analyzer.calculate_metrics(window=roll_win)
                    last_row = res_sml.iloc[-1]
                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    with m_c1: st.metric("Current Beta", f"{last_row['Beta']:.2f}")
                    with m_c2: st.metric("Jensen's Alpha", f"{last_row['Alpha_Daily']*252:.2%}")
                    with m_c3: st.metric("SML Exp Return", f"{last_row['SML_Exp_Return']:.2%}")
                    with m_c4:
                        mispricing = last_row['Mispricing_Spread']
                        st.metric("Mispricing Spread", f"{mispricing*100:.2f}%",
                                  delta="Undervalued" if mispricing > 0 else "Overvalued")

                    st.divider()
                    fig_sml, ax_sml = plt.subplots(figsize=(10, 6))
                    avg_mkt_excess = res_sml['mkt_ex'].mean() * 252
                    betas_line = np.linspace(0, max(res_sml['Beta'].max(), 2.0), 100)
                    sml_y = rf_rate + betas_line * avg_mkt_excess
                    ax_sml.plot(betas_line, sml_y, color='black', linestyle='--', linewidth=2, label='SML')
                    curr_beta = last_row['Beta']
                    curr_ret = last_row['Actual_Return_Ann']
                    ax_sml.scatter(curr_beta, curr_ret, color='blue', s=100, zorder=5, label=f'{TICKER} (Current)')
                    ax_sml.scatter(res_sml['Beta'], res_sml['Actual_Return_Ann'],
                                   c=range(len(res_sml)), cmap='Blues', alpha=0.3, s=20, label='Historical Path')
                    mkt_ret_tot = (res_sml['mkt_ex'].mean() * 252) + rf_rate
                    ax_sml.scatter(1.0, mkt_ret_tot, color='red', marker='D', s=80, label='Market')
                    ax_sml.set_xlabel("Systematic Risk (Beta)")
                    ax_sml.set_ylabel("Annualized Expected Return")
                    ax_sml.set_title("Risk-Reward Profile vs Equilibrium")
                    ax_sml.legend()
                    ax_sml.grid(True, alpha=0.3)
                    st.pyplot(fig_sml)

                    fig_dyn, ax_dyn = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    ax_dyn[0].plot(res_sml.index, res_sml['Beta'], color='purple', label='Rolling Beta')
                    ax_dyn[0].axhline(1.0, color='gray', linestyle='--')
                    ax_dyn[0].set_title(f"Beta - {roll_win} Day Window")
                    ax_dyn[0].legend()
                    ax_dyn[1].plot(res_sml.index, res_sml['Alpha_Daily'] * 252, color='green', label='Annualized Alpha')
                    ax_dyn[1].axhline(0, color='gray', linestyle='--')
                    ax_dyn[1].fill_between(res_sml.index, 0, res_sml['Alpha_Daily'] * 252, where=(res_sml['Alpha_Daily']>0), color='green', alpha=0.1)
                    ax_dyn[1].fill_between(res_sml.index, 0, res_sml['Alpha_Daily'] * 252, where=(res_sml['Alpha_Daily']<0), color='red', alpha=0.1)
                    ax_dyn[1].set_title("Manager Skill / Alpha")
                    ax_dyn[1].legend()
                    format_plot_dates(ax_dyn[1], res_sml.index)
                    st.pyplot(fig_dyn)
                    st.session_state.report_gen.add_plot("SML Factor Dynamics", fig_dyn)
                    st.session_state.report_gen.add_data("SML Analysis Results", res_sml.tail(100))
                else:
                    st.error(f"Could not load data for Benchmark: {bench_ticker}")

# ==========================================
# TAB 11: MULTI-ASSET SCAN
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
        mcap_map = {"All": 0, "$500M (Micro+)": 5e8, "$2B (Small+)": 2e9,
                    "$10B (Mid+)": 1e10, "$50B (Large+)": 5e10, "$200B (Mega+)": 2e11}
    with scan_col3:
        scan_depth = st.number_input("Scan Limit (Depth)", min_value=1, max_value=10000, value=50)
        if scan_depth > 500:
            st.warning("⚠️ High Depth: Scanning thousands of assets can take 30+ minutes.")

    scan_col1b, scan_col2b, scan_col3b = st.columns(3)
    with scan_col1b:
        scan_regime_mode = st.selectbox("Scanner Regime Mode",
                                        ["Fixed: 4 States", "Fixed: 2 States", "Fixed: 3 States", "Auto: Best Fit"], index=0)
        scan_reg_map = {"Fixed: 4 States": 4, "Fixed: 2 States": 2, "Fixed: 3 States": 3, "Auto: Best Fit": "Auto"}
        scan_reg_param = scan_reg_map[scan_regime_mode]
    with scan_col2b:
        scan_opt_goal = st.selectbox("Optimization Goal", ["Robustness (BIC)", "Performance (PnL)"], index=0)
    with scan_col3b:
        scan_freq = st.selectbox("Scanner Frequency", ["Daily", "Weekly"], index=0)

    with st.expander("🛠️ Advanced Model Sync", expanded=False):
        async_col1, async_col2, async_col3 = st.columns(3)
        with async_col1:
            scan_engine = st.selectbox("Model Engine", ["Markov (High Accuracy)", "GMM (Fast)"], index=1)
            scan_engine_param = "Markov" if "Markov" in scan_engine else "GMM"
            scan_initial_cap = st.number_input("Backtest Capital ($)", 1000, 1000000, 10000)
        with async_col2:
            scan_stability = st.slider("Signal Stability (Smoothing)", 0, 10, 4)
            scan_trailing_stop = st.slider("Trailing Stop (%)", 0.0, 20.0, 0.0, step=0.5) / 100
        with async_col3:
            scan_switch_vol = st.toggle("Switching Volatility", value=True)
            scan_switch_trend = st.toggle("Switching Mean", value=True)

    custom_input = ""
    if universe_type == "Custom Watchlist":
        custom_input = st.text_area("Ticker List (Comma separated)", "AAPL, TSLA, BTC-USD, GC=F")

    st.divider()

    if st.button("🚀 EXECUTE TOTAL MARKET SCAN", use_container_width=True, type="primary"):
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

        tickers_to_scan = full_list[:scan_depth]
        long_list = []
        cash_list = []
        st.session_state.scanner_results = None
        scan_prog = st.progress(0)
        status_text = st.empty()

        def process_ticker_worker(tick):
            try:
                mcap = get_market_cap(tick)
                if mcap_filter != "All" and mcap < mcap_map[mcap_filter]:
                    return None
                s_df = load_data(tick, start_date, end_date, interval=data_interval if live_mode else '1d')
                if s_df is None or s_df.empty:
                    return None
                s_analysis = get_master_signal(tick, s_df,
                                               n_regimes=scan_reg_param, freq=scan_freq,
                                               opt_goal=scan_opt_goal, stability=scan_stability,
                                               switch_vol=scan_switch_vol, switch_trend=scan_switch_trend,
                                               engine=scan_engine_param, initial_cap=scan_initial_cap,
                                               trailing_stop=scan_trailing_stop)
                if not s_analysis:
                    return None
                s_price = s_df['Close'].iloc[-1]
                return {
                    'Ticker': tick, 'Price': round(s_price, 2),
                    'Mkt Cap ($B)': round(mcap / 1e9, 2),
                    'Regimes (N)': s_analysis['regime_data'].get('n_states', 4),
                    'Score': s_analysis['sentiment_score'],
                    'Regime': s_analysis['regime_label'],
                    'Trend': f"{s_analysis['trend_diff']:+.2%}",
                    'Action': s_analysis['regime_sig']
                }
            except:
                return None

        from concurrent.futures import as_completed
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

        st.session_state.scanner_results = {'long': long_list, 'cash': cash_list,
                                             'universe': universe_type, 'count': len(tickers_to_scan)}

    if 'scanner_results' in st.session_state and st.session_state.scanner_results:
        res = st.session_state.scanner_results
        long_list = res['long']
        cash_list = res['cash']
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader(f"🚀 LONG / OPEN ({len(long_list)})")
            if long_list:
                ldf = pd.DataFrame(long_list).sort_values(by='Score', ascending=False)
                st.dataframe(ldf.style.background_gradient(subset=['Score'], cmap='Greens'), use_container_width=True)
            else:
                st.info("No bullish signals found.")
        with res_col2:
            st.subheader(f"🛑 CLOSED / CASH ({len(cash_list)})")
            if cash_list:
                cdf = pd.DataFrame(cash_list).sort_values(by='Score', ascending=True)
                st.dataframe(cdf.style.background_gradient(subset=['Score'], cmap='Reds'), use_container_width=True)
            else:
                st.info("No bearish/neutral signals found.")
        st.divider()
        st.success(f"✅ **Scan Complete**: Analyzed {res['count']} assets from `{res['universe']}`.")

# ==========================================
# TAB 12: FED BALANCE SHEET
# ==========================================
with tab12:
    st.write("### 🏦 Federal Reserve Balance Sheet (Assets & Liabilities)")
    st.caption("Macroeconomic dashboard tracking FED liquidity and monetary policy shifts via FRED.")

    fed_date_col1, _ = st.columns(2)
    with fed_date_col1:
        fed_start_date = st.date_input("FED History Start", datetime(2010, 1, 1))

    @st.fragment
    def render_fed_dashboard():
        with st.status("Fetching FRED Macro Data...", expanded=True) as status:
            asset_dfs = {}
            for sid, name in FED_ASSETS.items():
                status.update(label=f"Loading Asset: {name}...")
                df = load_fred_data(sid)
                if df is not None:
                    asset_dfs[name] = df[sid]

            liab_dfs = {}
            for sid, name in FED_LIABILITIES.items():
                status.update(label=f"Loading Liability: {name}...")
                df = load_fred_data(sid)
                if df is not None:
                    liab_dfs[name] = df[sid]

            status.update(label="All Macro data synchronized!", state="complete", expanded=False)

        if asset_dfs:
            assets_master = pd.DataFrame(asset_dfs).fillna(0)
            assets_master = assets_master[assets_master.index >= pd.Timestamp(fed_start_date)]
            st.subheader("Federal Reserve Assets (Stacked)")
            fig_assets, ax_assets = plt.subplots(figsize=(12, 6))
            (assets_master / 1e3).plot.area(ax=ax_assets, alpha=0.7, stacked=True, cmap='tab20')
            ax_assets.set_ylabel("Amount (Billions $)")
            ax_assets.set_title("FED Assets: Detailed Breakdown")
            ax_assets.legend(loc='upper left', fontsize='x-small', ncol=2)
            st.pyplot(fig_assets)

            st.subheader("Weekly Change in Total Assets (WALCL)")
            walcl = load_fred_data("WALCL")
            if walcl is not None:
                walcl = walcl[walcl.index >= pd.Timestamp(fed_start_date)]
                walcl_diff = walcl.diff().dropna() / 1e3
                fig_diff, ax_diff = plt.subplots(figsize=(12, 4))
                colors = ['green' if x >= 0 else 'red' for x in walcl_diff['WALCL']]
                ax_diff.bar(walcl_diff.index, walcl_diff['WALCL'], color=colors, width=7, alpha=0.6)
                ax_diff.set_ylabel("Change (Billions $)")
                ax_diff.set_title("FED Balance Sheet: Weekly Expansion/Contraction")
                st.pyplot(fig_diff)

        if liab_dfs:
            liabs_master = pd.DataFrame(liab_dfs).fillna(0)
            liabs_master = liabs_master[liabs_master.index >= pd.Timestamp(fed_start_date)]
            st.subheader("Federal Reserve Liabilities (Stacked)")
            fig_liabs, ax_liabs = plt.subplots(figsize=(12, 6))
            (liabs_master / 1e3).plot.area(ax=ax_liabs, alpha=0.7, stacked=True, cmap='Set3')
            ax_liabs.set_ylabel("Amount (Billions $)")
            ax_liabs.set_title("FED Liabilities & Capital Accounts")
            ax_liabs.legend(loc='upper left', fontsize='x-small', ncol=2)
            st.pyplot(fig_liabs)

    render_fed_dashboard()

# Footer
st.markdown("---")
st.caption("Quant Thesis Dashboard | Ehlers SuperSmoother + Multi-Model Suite")
