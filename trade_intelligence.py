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
import io, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
 
try:
    from fpdf import FPDF
    import xlsxwriter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
 
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
 
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
 
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
 
st.set_page_config(page_title="Unified Quant Suite", layout="wide", page_icon="📊")
plt.style.use("ggplot")
 
st.title("📊 Unified Quant Suite — Thesis + IV Scanner")
st.markdown("""
**Robust Financial Modeling Dashboard** incorporating:
GARCH/EGARCH | Regime Switching | Jump Diffusion | Heston | Kalman Filter | Macro Factors | **Institutional IV Scanner**
""")
 
if not ARCH_AVAILABLE:
    st.error("⚠️ 'arch' library not installed. Run: pip install arch")
 
# ==========================================
# HELPER FUNCTIONS & CLASSES
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
 
def highlight_plotly_zones(fig, mask, color, opacity=0.15, row=None, col=None):
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
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity,
                              layer="below", line_width=0, row=row, col=col)
            else:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=opacity,
                              layer="below", line_width=0)
 
 
# ── CALIBRATOR ──────────────────────────────────────────────────────────────
class Calibrator:
    @staticmethod
    def calibrate_heston(returns):
        dt = 1/252
        am = arch_model(returns * 100, vol='Garch', p=1, o=0, q=1, dist='Normal')
        res = am.fit(disp='off')
        conditional_vol = res.conditional_volatility / 100
        variance = conditional_vol**2
        variance = variance.values if hasattr(variance, 'values') else variance
        v_curr = variance[:-1]; v_next = variance[1:]
        Y = (v_next - v_curr) / dt; X = v_curr
        A = np.vstack([X, np.ones(len(X))]).T
        beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        kappa = max(-beta, 0.1)
        theta = max(alpha / kappa if kappa != 0 else np.mean(variance), 0.01)
        residuals = Y - (alpha + beta * X)
        xi = max(np.std(residuals) * np.sqrt(dt) / np.mean(np.sqrt(v_curr)), 0.1)
        rho = np.corrcoef(returns[1:], np.diff(variance))[0, 1]
        mu = np.mean(returns) / dt + 0.5 * np.mean(variance)
        return {'mu': mu, 'kappa': kappa, 'theta': theta, 'xi': xi,
                'rho': rho, 'v0': variance[-1], 'S0': 100.0}
 
 
# ── KALMAN FILTERS ───────────────────────────────────────────────────────────
class KalmanFilterReg:
    def __init__(self, delta=1e-4, R=1e-3):
        self.delta = delta; self.R = R
        self.trans_cov = delta / (1 - delta) * np.eye(2)
 
    def run_filter(self, y, x):
        n = len(y)
        state_mean = np.zeros((n, 2)); state_cov = np.zeros((n, 2, 2))
        state_mean[0] = [0, 1]; state_cov[0] = np.eye(2)
        for t in range(1, n):
            pred_state = state_mean[t-1]
            pred_cov = state_cov[t-1] + self.trans_cov
            obs_mat = np.array([[1.0, x[t]]])
            error = y[t] - np.dot(obs_mat, pred_state)
            S = np.dot(np.dot(obs_mat, pred_cov), obs_mat.T) + self.R
            K = np.dot(pred_cov, obs_mat.T) / S
            state_mean[t] = pred_state + K.flatten() * error
            state_cov[t] = pred_cov - np.dot(np.dot(K, obs_mat), pred_cov)
        return state_mean, state_cov
 
 
class KalmanFilterTrend:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-3):
        self.Q = process_noise; self.R = measurement_noise
 
    def filter(self, data):
        n = len(data)
        estimates = np.zeros(n); covariances = np.zeros(n)
        init_window = min(10, max(1, n // 10))
        x = np.mean(data[:init_window])
        P = np.var(data[:init_window]) if init_window > 1 else 1.0
        for t in range(n):
            P_pred = P + self.Q
            K = P_pred / (P_pred + self.R)
            x = x + K * (data[t] - x)
            P = (1 - K) * P_pred
            estimates[t] = x; covariances[t] = P
        return estimates, covariances
 
    def smooth(self, data):
        n = len(data)
        filtered_means, filtered_covs = self.filter(data)
        smoothed_means = np.zeros(n); smoothed_covs = np.zeros(n)
        smoothed_means[-1] = filtered_means[-1]; smoothed_covs[-1] = filtered_covs[-1]
        for t in range(n - 2, -1, -1):
            P_pred = filtered_covs[t] + self.Q
            J = filtered_covs[t] / P_pred
            smoothed_means[t] = filtered_means[t] + J * (smoothed_means[t+1] - filtered_means[t])
            smoothed_covs[t] = filtered_covs[t] + J**2 * (smoothed_covs[t+1] - P_pred)
        return smoothed_means, smoothed_covs
 
 
# ── STOCHASTIC MODELS ─────────────────────────────────────────────────────────
def simulate_heston(S0, T, r, kappa, theta, sigma, rho, v0, steps, paths):
    dt = T / steps
    prices = np.zeros((steps + 1, paths)); vols = np.zeros((steps + 1, paths))
    prices[0] = S0; vols[0] = v0
    for t in range(1, steps + 1):
        Z1 = np.random.normal(size=paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(size=paths)
        v_prev = vols[t-1]
        v_curr = np.abs(v_prev + kappa * (theta - v_prev) * dt +
                        sigma * np.sqrt(np.abs(v_prev)) * np.sqrt(dt) * Z2)
        vols[t] = v_curr
        prices[t] = prices[t-1] + r * prices[t-1] * dt + \
                    np.sqrt(v_curr) * prices[t-1] * np.sqrt(dt) * Z1
    return prices, vols
 
 
def merton_jump_diffusion(S0, T, r, sigma, lam, mu_j, sigma_j, steps, paths):
    dt = T / steps
    prices = np.zeros((steps + 1, paths)); prices[0] = S0
    drift = r - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    for t in range(1, steps + 1):
        z = np.random.normal(size=paths)
        N = np.random.poisson(lam * dt, size=paths)
        J = np.random.normal(mu_j, sigma_j, size=paths) * N
        prices[t] = prices[t-1] * np.exp(drift * dt + sigma * np.sqrt(dt) * z + J)
    return prices
 
 
# ── REALIZED VOLATILITY ───────────────────────────────────────────────────────
class RealizedVolatility:
    @staticmethod
    def realized_variance(returns):
        return np.sum(returns**2)
 
    @staticmethod
    def bipower_variation(returns):
        if len(returns) < 2: return 0.0
        abs_rets = np.abs(returns)
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
        z_score = (jump_ratio - 0.05) * np.sqrt(n / 2)
        p_value = 1 - stats.norm.cdf(z_score)
        return {'jump_ratio': jump_ratio, 'p_value': p_value, 'z_score': z_score}
 
 
# ── HAWKES VOLATILITY ─────────────────────────────────────────────────────────
class HawkesVolatility:
    def __init__(self):
        self.mu = 0.5; self.alpha = 0.5; self.beta = 2.0
 
    def fit(self, returns):
        vol_proxy = np.abs(returns)
        if len(vol_proxy) < 20: return self
        threshold = np.percentile(vol_proxy, 90)
        events = np.where(vol_proxy > threshold)[0]
        if len(events) < 5: return self
 
        def neg_ll(params):
            mu_p, alpha_p, beta_p = params
            if mu_p <= 0 or alpha_p < 0 or beta_p <= alpha_p: return 1e9
            t = events; T_end = len(returns)
            R = np.zeros(len(t))
            for i in range(1, len(t)):
                R[i] = np.exp(-beta_p * (t[i] - t[i-1])) * (1 + R[i-1])
            intensities = mu_p + alpha_p * R
            if np.any(intensities <= 0): return 1e9
            term1 = np.sum(np.log(intensities))
            term2 = mu_p * T_end + (alpha_p / beta_p) * np.sum(1 - np.exp(-beta_p * (T_end - t)))
            return -(term1 - term2)
 
        try:
            res = minimize(neg_ll, [0.1, 0.2, 1.0],
                           bounds=[(1e-4, 2.0), (1e-4, 5.0), (0.1, 10.0)], method='L-BFGS-B')
            self.mu, self.alpha, self.beta = res.x
        except:
            pass
        return self
 
    def branching_ratio(self):
        return self.alpha / self.beta if self.beta != 0 else 0.0
 
    def half_life(self):
        return np.log(2) / self.beta if self.beta != 0 else 0.0
 
 
# ── PRO REGIME DETECTOR ───────────────────────────────────────────────────────
class ProRegimeDetector:
    def __init__(self, prices, log_returns):
        self.prices = prices if isinstance(prices, pd.Series) else pd.Series(prices)
        self.returns = log_returns if isinstance(log_returns, pd.Series) else pd.Series(log_returns)
        self.features = None; self.regimes = {}; self.metrics = {}; self.state_labels = {}
 
    def _prepare_features(self):
        f1 = self.returns.rolling(window=5).mean().fillna(0)
        vol = self.returns.rolling(window=20).std().bfill()
        v_mean = vol.rolling(252, min_periods=20).mean()
        v_std = vol.rolling(252, min_periods=20).std()
        f2 = ((vol - v_mean) / (v_std + 1e-9)).fillna(0)
        ema = self.prices.ewm(span=20).mean()
        f3 = ((self.prices - ema) / (ema + 1e-9)).fillna(0)
        self.features = np.column_stack([f1.values, f2.values, f3.values])
        return self.features
 
    def fit(self, n_states=4):
        X = self._prepare_features()
        if not SKLEARN_AVAILABLE:
            self.regimes['states'] = np.zeros(len(X))
            self.regimes['probs'] = np.ones((len(X), 1))
            return
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = GaussianMixture(n_components=n_states, covariance_type='full',
                                random_state=123, max_iter=200)
        model.fit(X_scaled)
        states = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        state_stats = []
        for i in range(n_states):
            mask = (states == i)
            if np.sum(mask) > 0:
                state_stats.append({'id': i,
                                    'ret': np.mean(self.features[mask, 0]),
                                    'vol': np.mean(self.features[mask, 1])})
            else:
                state_stats.append({'id': i, 'ret': -999, 'vol': 999})
        sorted_stats = sorted(state_stats, key=lambda x: x['ret'], reverse=True)
        if n_states == 4:
            bulls = sorted_stats[:2]; bears = sorted_stats[2:]
            bl = min(bulls, key=lambda x: x['vol']); bh = max(bulls, key=lambda x: x['vol'])
            dl = min(bears, key=lambda x: x['vol']); dh = max(bears, key=lambda x: x['vol'])
            self.state_labels = {
                bl['id']: "BULL / QUIET (Conviction)", bh['id']: "BULL / VOLATILE (Exhaustion)",
                dl['id']: "BEAR / QUIET (Distribution)", dh['id']: "BEAR / VOLATILE (Panic/Crisis)"}
        elif n_states == 2:
            self.state_labels = {sorted_stats[0]['id']: "BULL REGIME (Accumulation)",
                                  sorted_stats[1]['id']: "BEAR REGIME (Distribution)"}
        elif n_states == 3:
            self.state_labels = {sorted_stats[0]['id']: "BULL REGIME (Conviction)",
                                  sorted_stats[1]['id']: "NEUTRAL / TRANSITION",
                                  sorted_stats[2]['id']: "BEAR REGIME (Panic)"}
        self.regimes['states'] = states; self.regimes['probs'] = probs
        self.metrics['aic'] = model.aic(X_scaled); self.metrics['bic'] = model.bic(X_scaled)
        self.metrics['n_states'] = n_states
 
    def fit_optimized(self, state_choices=[2, 3, 4]):
        best_bic = float('inf'); best_n = 4
        for n in state_choices:
            try:
                tmp = ProRegimeDetector(self.prices, self.returns); tmp.fit(n_states=n)
                if tmp.metrics.get('bic', float('inf')) < best_bic:
                    best_bic = tmp.metrics['bic']; best_n = n
            except: continue
        self.fit(n_states=best_n); return best_n
 
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
 
 
# ── SML ANALYZER ──────────────────────────────────────────────────────────────
class SMLAnalyzer:
    def __init__(self, ticker_returns, benchmark_returns, rf_annual=0.04):
        self.r_asset = ticker_returns; self.r_bench = benchmark_returns
        self.rf_annual = rf_annual; self.rf_daily = rf_annual / 252
 
    def calculate_metrics(self, window=90):
        common_idx = self.r_asset.index.intersection(self.r_bench.index)
        y = self.r_asset.loc[common_idx] - self.rf_daily
        x = self.r_bench.loc[common_idx] - self.rf_daily
        df = pd.DataFrame({'asset_ex': y, 'mkt_ex': x}, index=common_idx)
        beta_arr = np.full(len(df), np.nan); alpha_arr = np.full(len(df), np.nan)
        for i in range(window, len(df)):
            ws = df.iloc[i-window:i]
            try:
                model = sm.OLS(ws['asset_ex'], sm.add_constant(ws['mkt_ex'])).fit(
                    cov_type='HAC', cov_kwds={'maxlags': 1})
                alpha_arr[i] = model.params.get('const', np.nan)
                beta_arr[i] = model.params.get('mkt_ex', np.nan)
            except: pass
        df['Beta'] = beta_arr; df['Alpha_Daily'] = alpha_arr
        rolling_mkt_ret_ann = df['mkt_ex'].rolling(window).mean() * 252
        df['SML_Exp_Return'] = self.rf_annual + df['Beta'] * rolling_mkt_ret_ann
        df['Actual_Return_Ann'] = (df['asset_ex'].rolling(window).mean() * 252) + self.rf_annual
        df['Mispricing_Spread'] = df['Actual_Return_Ann'] - df['SML_Exp_Return']
        return df.dropna()
 
 
# ── MAD TREND MODES ───────────────────────────────────────────────────────────
class MADTrendModes:
    @staticmethod
    def sma(s, l): return s.rolling(window=l).mean()
    @staticmethod
    def ema(s, l): return s.ewm(span=l, adjust=False).mean()
    @staticmethod
    def wma(s, l):
        w = np.arange(1, l+1)
        return s.rolling(l).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    @staticmethod
    def hma(s, l):
        hl = int(l/2); sl = int(np.sqrt(l))
        combined = 2*MADTrendModes.wma(s, hl) - MADTrendModes.wma(s, l)
        return MADTrendModes.wma(combined.dropna(), sl).reindex_like(s)
    @staticmethod
    def rma(s, l): return s.ewm(alpha=1/l, adjust=False).mean()
    @staticmethod
    def alma(s, l, offset=0.85, sigma=6):
        m = offset*(l-1); si = l/sigma
        w = np.exp(-((np.arange(l)-m)**2)/(2*si*si)); w /= w.sum()
        return s.rolling(l).apply(lambda x: np.dot(x, w), raw=True)
    @staticmethod
    def lsma(s, l):
        def lr(y):
            x = np.arange(len(y)); sl, ic = np.polyfit(x, y, 1)
            return sl*(len(y)-1)+ic
        return s.rolling(l).apply(lr, raw=True)
    @staticmethod
    def ma_switch(s, l, t):
        m = {'SMA': MADTrendModes.sma, 'EMA': MADTrendModes.ema, 'WMA': MADTrendModes.wma,
             'HMA': MADTrendModes.hma, 'RMA': MADTrendModes.rma, 'ALMA': MADTrendModes.alma,
             'LSMA': MADTrendModes.lsma}
        return m.get(t, MADTrendModes.sma)(s, l)
    @staticmethod
    def calculate_mad(series, benchmark, length):
        from numpy.lib.stride_tricks import sliding_window_view
        vals = series.values; bench_vals = benchmark.values
        if len(vals) < length: return pd.Series(np.nan, index=series.index)
        windows = sliding_window_view(vals, length)
        diffs = np.abs(windows - bench_vals[length-1:, np.newaxis])
        res = np.full(len(series), np.nan); res[length-1:] = np.mean(diffs, axis=1)
        return pd.Series(res, index=series.index)
    @staticmethod
    def system_score(series, a, b):
        total = pd.Series(0.0, index=series.index)
        for i in range(a, b+1):
            total += np.sign(series - series.shift(i)).fillna(0)
        return total
    @staticmethod
    def get_signals(df, params):
        src = df['Close']
        mode = params.get('signal_mode', 'Bollinger Bands')
        bb_ma = params.get('bb_ma_type', 'EMA'); bb_len = params.get('bb_len', 25)
        bb_mp = params.get('bb_mult_p', 1.4); bb_mn = params.get('bb_mult_n', 1.0)
        fl_ma = params.get('fl_ma_type', 'ALMA'); fl_len = params.get('fl_len', 10)
        fl_a = params.get('fl_a', 10); fl_b = params.get('fl_b', 60)
        fl_tl = params.get('fl_thresh_l', 23); fl_ts = params.get('fl_thresh_s', 3)
        c_tl = params.get('c_thresh_l', 0.0); c_ts = params.get('c_thresh_s', 0.0)
        avg_bb = MADTrendModes.ma_switch(src, bb_len, bb_ma)
        mad_bb = MADTrendModes.calculate_mad(src, avg_bb, bb_len)
        bb_up = avg_bb + mad_bb*bb_mp; bb_dn = avg_bb - mad_bb*bb_mn
        avg_fl = MADTrendModes.ma_switch(src, fl_len, fl_ma)
        mad_fl = MADTrendModes.calculate_mad(src, avg_fl, fl_len)
        num = MADTrendModes.ma_switch(src*mad_fl, fl_len, fl_ma)
        den = MADTrendModes.ma_switch(mad_fl, fl_len, fl_ma)
        mad_w = num/den
        sys_sc = MADTrendModes.system_score(mad_w, fl_a, fl_b)
        def stateful(lc, sc, idx):
            sig = pd.Series(np.nan, index=idx)
            sig.loc[lc] = 1; sig.loc[sc] = -1
            return sig.ffill().fillna(0)
        bb_sc = stateful((src>bb_up)&(src.shift(1)<=bb_up.shift(1)),
                          (src<bb_dn)&(src.shift(1)>=bb_dn.shift(1)), src.index)
        fl_sc = stateful((sys_sc>fl_tl)&(sys_sc.shift(1)<=fl_tl),
                          (sys_sc<fl_ts)&(sys_sc.shift(1)>=fl_ts), src.index)
        c_sig = (bb_sc+fl_sc)/2
        c_sc = stateful((c_sig>c_tl)&(c_sig.shift(1)<=c_tl),
                         (c_sig<c_ts)&(c_sig.shift(1)>=c_ts), src.index)
        fs = bb_sc if mode=="Bollinger Bands" else (fl_sc if mode=="For Loop" else c_sc)
        return (fs == 1).astype(int)
 
 
# ── EHLERS FILTERS ────────────────────────────────────────────────────────────
class EhlersFilters:
    @staticmethod
    def super_smoother(prices, period=15):
        a1 = np.exp(-1.414*np.pi/period); b1 = 2*a1*np.cos(1.414*np.pi/period)
        c2 = b1; c3 = -a1*a1; c1 = 1-c2-c3
        filt = np.zeros(len(prices)); pv = prices.values
        for i in range(len(prices)):
            filt[i] = pv[i] if i < 2 else c1*(pv[i]+pv[i-1])/2+c2*filt[i-1]+c3*filt[i-2]
        return pd.Series(filt, index=prices.index)
 
    @staticmethod
    def simple_decycler(prices, period=60):
        a1 = (np.cos(0.707*2*np.pi/period)+np.sin(0.707*2*np.pi/period)-1)/np.cos(0.707*2*np.pi/period)
        hp = np.zeros(len(prices)); pv = prices.values
        for i in range(len(prices)):
            hp[i] = 0 if i < 2 else ((1-a1/2)**2)*(pv[i]-2*pv[i-1]+pv[i-2]) + \
                    2*(1-a1)*hp[i-1]-((1-a1)**2)*hp[i-2]
        return pd.Series(pv-hp, index=prices.index)
 
 
# ── VOL-TARGETED SIZING (FIX 1) ───────────────────────────────────────────────
def vol_targeted_signal(raw_signal, garch_res, target_vol_annual=0.15):
    """Converts binary 0/1 signal to fractional 0.0-1.0 using vol targeting."""
    if garch_res is None: return raw_signal.astype(float)
    cond_vol = garch_res.conditional_volatility / 100.0 * np.sqrt(252)
    common = raw_signal.index.intersection(cond_vol.index)
    if len(common) == 0: return raw_signal.astype(float)
    sig = raw_signal.loc[common].astype(float)
    vol = cond_vol.loc[common]
    size = np.minimum(1.0, target_vol_annual / (vol + 1e-9))
    return (sig * size).fillna(0.0)
 
 
# ── IMPROVED HURST SIGNAL (FIX 2) ────────────────────────────────────────────
def rolling_hurst(prices, window=100, max_lag=20):
    log_prices = np.log(prices)
    def hurst_val(x):
        lags = range(2, max_lag)
        tau = [max(np.std(x[lag:]-x[:-lag]), 1e-8) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return log_prices.rolling(window).apply(hurst_val, raw=True)
 
 
def hurst_confirmed_signal(prices_bt, hurst_window=100, trend_thresh=0.55,
                            mr_thresh=0.45, confirm_bars=5):
    """
    Dead zone 0.45-0.55 + 5-bar confirmation to eliminate whipsaw.
    H>0.55 confirmed -> momentum, H<0.45 confirmed -> mean reversion, else -> cash.
    """
    hurst_series = rolling_hurst(prices_bt, window=hurst_window)
 
    def confirmed(h, threshold, above=True):
        raw = (h > threshold).astype(int) if above else (h < threshold).astype(int)
        return raw.rolling(confirm_bars).min().fillna(0).astype(int)
 
    is_trending = confirmed(hurst_series, trend_thresh, above=True)
    is_mr = confirmed(hurst_series, mr_thresh, above=False)
 
    ema_fast = prices_bt.ewm(span=20, adjust=False).mean()
    ema_slow = prices_bt.ewm(span=50, adjust=False).mean()
    trend_sig = (ema_fast > ema_slow).astype(int)
 
    bb_ma = prices_bt.rolling(20).mean()
    bb_std = prices_bt.rolling(20).std()
    bb_lower = bb_ma - 2*bb_std
    mr_raw = pd.Series(np.nan, index=prices_bt.index)
    mr_raw.loc[prices_bt < bb_lower] = 1
    mr_raw.loc[prices_bt > bb_ma] = 0
    mr_sig = mr_raw.ffill().fillna(0)
 
    signals = pd.Series(0, index=prices_bt.index)
    signals[is_trending == 1] = trend_sig[is_trending == 1]
    signals[is_mr == 1] = mr_sig[is_mr == 1]
    return signals.fillna(0), hurst_series, is_trending, is_mr
 
 
# ── GARCH VOL FILTER (FIX 3) ─────────────────────────────────────────────────
def garch_vol_filter_signal(prices_bt, returns_bt, base_signal, target_vol_pct=15.0):
    """
    Replaces VIX proxy with stock's own GARCH vol state.
    Returns fractional signal sized by vol target instead of binary on/off.
    """
    if not ARCH_AVAILABLE: return base_signal.astype(float)
    try:
        rs = (returns_bt * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if len(rs) < 30: return base_signal.astype(float)
        am = arch_model(rs, vol='Garch', p=1, q=1, dist='Normal')
        res = am.fit(disp='off')
        return vol_targeted_signal(base_signal, res, target_vol_annual=target_vol_pct/100.0)
    except:
        return base_signal.astype(float)
 
 
# ── BACKTEST ENGINE ───────────────────────────────────────────────────────────
class BacktestEngine:
    @staticmethod
    def run_strategy(prices, signals, initial_capital=10000.0,
                     trailing_stop_pct=0.0, stop_loss_pct=0.0):
        common_idx = prices.index.intersection(signals.index)
        prices = prices.loc[common_idx]; signals = signals.loc[common_idx]
        returns = prices.pct_change().fillna(0)
        equity_curve = [initial_capital]
        trades = []; position = 0; entry_price = 0; entry_date = None
        max_price = 0; cash = initial_capital; holdings = 0
        cooldown = 0; prev_sig = 0.0
 
        for date, price, signal in zip(prices.index, prices, signals):
            signal = float(signal)
            if cooldown > 0:
                cooldown -= 1
                if signal == 0: cooldown = 0
 
            if position == 1:
                stop_out = False; smsg = ""
                if stop_loss_pct > 0 and price <= entry_price*(1-stop_loss_pct):
                    stop_out = True; smsg = 'Stop Loss'
                if not stop_out and trailing_stop_pct > 0:
                    max_price = max(max_price, price)
                    if price <= max_price*(1-trailing_stop_pct):
                        stop_out = True; smsg = 'Trailing Stop'
                if stop_out:
                    position = 0; cash = holdings*price; holdings = 0
                    pnl = (price-entry_price)/entry_price
                    trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':date,
                                   'Buy Price':entry_price,'Sell Price':price,
                                   'PnL (%)':pnl*100,'Status':smsg})
                    equity_curve.append(cash); cooldown = 5; prev_sig = 0.0; continue
 
            if position == 0 and signal > 0 and cooldown == 0:
                position = 1; entry_price = price; entry_date = date; max_price = price
                invest = cash * signal; holdings = invest/price; cash -= invest
            elif position == 1 and signal == 0:
                position = 0; cash += holdings*price
                pnl = (price-entry_price)/entry_price; holdings = 0
                trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':date,
                               'Buy Price':entry_price,'Sell Price':price,
                               'PnL (%)':pnl*100,'Status':'Closed'})
            elif position == 1 and signal != prev_sig and signal > 0:
                total = cash + holdings*price
                holdings = (total*signal)/price; cash = total - holdings*price
            prev_sig = signal
            equity_curve.append((cash+holdings*price) if position==1 else cash)
 
        if position == 1:
            cp = prices.iloc[-1]; pnl = (cp-entry_price)/entry_price
            trades.append({'Side':'Long','Entry Date':entry_date,'Exit Date':None,
                           'Buy Price':entry_price,'Sell Price':cp,
                           'PnL (%)':pnl*100,'Status':'Open'})
            equity_curve[-1] = cash+holdings*cp
 
        eq = pd.Series(equity_curve[1:], index=prices.index)
        bm = initial_capital*(1+returns).cumprod()
        return {'equity_curve':eq,'benchmark_curve':bm,
                'trades':pd.DataFrame(trades),'returns':eq.pct_change().fillna(0)}
 
    @staticmethod
    def calculate_metrics(returns, risk_free_rate=0.0):
        if len(returns) < 2: return {}
        excess = returns - risk_free_rate/252
        sharpe = np.sqrt(252)*excess.mean()/(returns.std()+1e-9)
        down = returns[returns<0]
        sortino = np.sqrt(252)*excess.mean()/(down.std()+1e-9)
        cum = (1+returns).cumprod()
        max_dd = ((cum-cum.cummax())/cum.cummax()).min()
        n_years = len(returns)/252
        cagr = ((1+returns).prod()**(1/n_years)-1) if n_years>0 else 0
        return {'Sharpe Ratio':sharpe,'Sortino Ratio':sortino,'Max Drawdown':max_dd,'CAGR':cagr}
 
# ── REGIME MODEL FIT (CACHED) ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fit_regime_model(model_data, n_regimes, switch_vol, switch_trend, search_reps=20):
    if hasattr(model_data, 'values'):
        clean = model_data.values.flatten().astype(float); idx = model_data.index
    else:
        clean = np.array(model_data).flatten().astype(float); idx = pd.RangeIndex(len(clean))
    if np.any(np.isnan(clean)) or np.any(np.isinf(clean)):
        st.error("Data contains NaN/Inf. Cannot fit model."); return None
    if np.std(clean) < 1e-9:
        st.error("Data is constant. Cannot fit model."); return None
    endog = pd.Series(clean, index=idx)
    try:
        mod = MarkovRegression(endog, k_regimes=n_regimes, trend='c',
                               switching_variance=switch_vol, switching_trend=switch_trend)
        res = mod.fit(search_reps=search_reps, disp=False)
        if isinstance(res.params, np.ndarray):
            names = res.model.param_names
            res.params = pd.Series(res.params, index=names)
            res.bse = pd.Series(res.bse, index=names)
            res.pvalues = pd.Series(res.pvalues, index=names)
        return res
    except Exception as e:
        st.error(f"Fit failed: {e}"); return None
 
 
# ── MASTER SIGNAL ENGINE ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_master_signal(ticker, df, n_regimes=4, freq='Daily', opt_goal='Robustness (BIC)',
                       stability=0, switch_vol=True, switch_trend=True, engine='Markov',
                       initial_cap=10000.0, trailing_stop=0.0, stop_loss=0.0):
    try:
        df = df.replace([np.inf,-np.inf], np.nan).dropna()
        if stability > 0:
            for col in ['Returns','Log_Returns','Close']:
                if col in df.columns:
                    df[col] = df[col].ewm(span=stability, adjust=False).mean()
        if freq == 'Weekly':
            df = df.resample('W').last().replace([np.inf,-np.inf], np.nan).dropna()
            df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
            df['Returns'] = df['Close'].pct_change()
            df = df.replace([np.inf,-np.inf], np.nan).dropna()
        if len(df) < 15: return None
 
        if engine == 'Markov':
            if n_regimes == 'Auto':
                best_n=4; best_sc=-float('inf') if opt_goal=='Performance (PnL)' else float('inf')
                best_r=None
                for n in [2,3,4]:
                    try:
                        r = fit_regime_model(df['Returns']*100, n, switch_vol, switch_trend, search_reps=5)
                        if r:
                            sc = r.bic if opt_goal != 'Performance (PnL)' else 0
                            if (opt_goal=='Robustness (BIC)' and sc < best_sc) or \
                               (opt_goal=='Performance (PnL)' and sc > best_sc):
                                best_sc=sc; best_n=n; best_r=r
                    except: continue
                res_markov = best_r
            else:
                res_markov = fit_regime_model(df['Returns']*100, int(n_regimes), switch_vol, switch_trend)
            if not res_markov: return None
            p_df = res_markov.filtered_marginal_probabilities
            n_st = res_markov.k_regimes
            r_means = []
            for i in range(n_st):
                m = res_markov.params[f'const[{i}]'] if f'const[{i}]' in res_markov.params \
                    else res_markov.params.get('const', 0.0)
                r_means.append((i, m))
            bull_idx = sorted(r_means, key=lambda x: x[1], reverse=True)[0][0]
            bear_idx = sorted(r_means, key=lambda x: x[1])[0][0]
            curr = p_df.iloc[-1].idxmax(); prob = p_df.iloc[-1].max()
            if curr == bull_idx: sig, lbl = "LONG", "BULL"
            elif curr == bear_idx: sig, lbl = "SHORT", "BEAR"
            else: sig, lbl = "CASH", "NEUTRAL"
            regime_data = {'label': lbl, 'confidence': prob, 'n_states': n_st}
            p_detector = None
        else:
            p_detector = ProRegimeDetector(df['Close'], df['Log_Returns'])
            if n_regimes == 'Auto': p_detector.fit_optimized()
            else: p_detector.fit(n_states=int(n_regimes))
            sig, prob, lbl = p_detector.get_latest_verdict()
            regime_data = {'label': lbl, 'confidence': prob,
                           'n_states': p_detector.metrics.get('n_states', 4)}
 
        kf = KalmanFilterTrend(process_noise=1e-4, measurement_noise=1e-2)
        trend_est, _ = kf.filter(df['Close'].values)
        last_price = df['Close'].iloc[-1]; last_trend = trend_est[-1]
        trend_diff = (last_price - last_trend) / (last_trend + 1e-9)
 
        rs = (df['Returns']*100).replace([np.inf,-np.inf], np.nan).dropna()
        if len(rs) < 15: return None
        am = arch_model(rs, vol='Garch', p=1, q=1, dist='Normal')
        garch_res = am.fit(disp='off')
        curr_vol = garch_res.conditional_volatility.iloc[-1]
        avg_vol = garch_res.conditional_volatility.mean()
        vol_state = "HIGH" if curr_vol > avg_vol*1.2 else "LOW" if curr_vol < avg_vol*0.8 else "NORMAL"
 
        jump_res = RealizedVolatility.jump_component(df['Returns'].values)
        jump_detected = jump_res['p_value'] < 0.05
 
        score = 0
        if "LONG" in sig: score += 2
        if "SHORT" in sig: score -= 2
        if trend_diff > 0.01: score += 1
        if trend_diff < -0.01: score -= 1
        if vol_state == "LOW": score += 1
        if vol_state == "HIGH": score -= 1
        if jump_detected: score -= 1
 
        return {'regime_sig': sig, 'regime_label': lbl, 'regime_data': regime_data,
                'regime_prob': prob, 'pro_detector': p_detector, 'trend_diff': trend_diff,
                'vol_state': vol_state, 'curr_vol': curr_vol, 'jump_detected': jump_detected,
                'sentiment_score': score, 'garch_res': garch_res}
    except Exception as e:
        st.error(f"Error in Decision Engine for {ticker}: {e}"); return None
 
 
# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_data(ticker, start, end, interval='1d'):
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex):
            try: df = df.xs(ticker, axis=1, level=1, drop_level=True)
            except: df.columns = df.columns.droplevel(1)
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
        return df.replace([np.inf,-np.inf], np.nan).dropna()
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}"); return None
 
 
@st.cache_data(ttl=3600)
def load_fred_data(series_id):
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url)
        df['DATE'] = pd.to_datetime(df['DATE']); df.set_index('DATE', inplace=True)
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df
    except: return None
 
 
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    try:
        return pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","BRK-B","UNH","JNJ"]
 
 
@st.cache_data(ttl=86400)
def get_nasdaq100_tickers():
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for t in tables:
            if 'Ticker' in t.columns: return t['Ticker'].tolist()
        return tables[4].iloc[:,1].tolist()
    except:
        return ["AAPL","MSFT","AMZN","GOOG","NVDA","META","TSLA","PEP","AVGO","COST"]
 
 
@st.cache_data(ttl=86400)
def get_total_us_stocks():
    import requests
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers={'User-Agent':'QuantApp/1.0 (admin@q.local)'}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            tickers = sorted(list(set([v['ticker'] for v in data.values()
                                       if len(str(v['ticker'])) <= 5 and '-' not in str(v['ticker'])])))
            if len(tickers) > 5000: return tickers
    except: pass
    return get_sp500_tickers()
 
 
def get_market_cap(ticker):
    try: return yf.Ticker(ticker).info.get('marketCap', 0)
    except: return 0
 
 
def get_analyst_target(ticker):
    try:
        info = yf.Ticker(ticker).info
        target = info.get('targetMeanPrice')
        current = info.get('currentPrice') or info.get('previousClose')
        if target and current: return target, np.log(target/current)
        return None, None
    except: return None, None
 
 
def calculate_beta(ticker_returns, benchmark_ticker='SPY', lookback_years=2):
    try:
        end = datetime.now(); start = end - timedelta(days=lookback_years*365)
        bench = yf.download(benchmark_ticker, start=start, end=end, progress=False)
        if isinstance(bench.columns, pd.MultiIndex): bench.columns = bench.columns.droplevel(1)
        if 'Close' not in bench.columns and 'Adj Close' in bench.columns:
            bench['Close'] = bench['Adj Close']
        bench_ret = bench['Close'].pct_change().dropna()
        common = ticker_returns.index.intersection(bench_ret.index)
        if len(common) < 30: return 1.0
        y = ticker_returns.loc[common]; x = bench_ret.loc[common]
        return np.cov(y, x)[0,1] / np.var(x)
    except: return 1.0
 
 
# ── REPORT GENERATOR ──────────────────────────────────────────────────────────
class ReportGenerator:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker; self.start_date = start_date
        self.end_date = end_date; self.data_store = {}; self.plots = {}
 
    def add_data(self, key, df_or_dict): self.data_store[key] = df_or_dict
 
    def add_plot(self, key, fig):
        buf = io.BytesIO()
        if hasattr(fig, 'savefig'): fig.savefig(buf, format='png', bbox_inches='tight')
        elif hasattr(fig, 'write_image'):
            try: fig.write_image(buf, format='png', engine='kaleido')
            except: return
        else: return
        buf.seek(0); self.plots[key] = buf
 
    def generate_excel(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for key, data in self.data_store.items():
                sn = "".join([c for c in key if c.isalnum() or c in (" ","_")])[:31]
                if isinstance(data, pd.DataFrame): data.to_excel(writer, sheet_name=sn)
                elif isinstance(data, dict):
                    pd.DataFrame(list(data.items()), columns=['Metric','Value']).to_excel(
                        writer, sheet_name=sn, index=False)
        return output.getvalue()
 
    def generate_pdf(self):
        pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
        pdf.set_font("Arial",'B',24); pdf.set_text_color(44,62,80)
        pdf.cell(0,20,f"Unified Quant Analysis Report",ln=True,align='C')
        pdf.set_font("Arial",'B',16); pdf.cell(0,10,f"Asset: {self.ticker}",ln=True,align='C')
        pdf.set_font("Arial",size=10); pdf.set_text_color(100,100,100)
        pdf.cell(0,10,f"Period: {self.start_date} to {self.end_date}",ln=True,align='C')
        pdf.cell(0,10,f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",ln=True,align='C')
        pdf.ln(10); printed = set()
        for key, data in self.data_store.items():
            pdf.set_font("Arial",'B',14); pdf.set_text_color(31,119,180)
            pdf.cell(0,10,f"SECTION: {key}",ln=True)
            pdf.set_font("Arial",size=10); pdf.set_text_color(0,0,0)
            if isinstance(data, dict):
                for k,v in data.items():
                    vs = f"{v:.4f}" if isinstance(v,(float,np.float64,np.float32)) else str(v)
                    pdf.cell(70,6,f"{k}:",border=0); pdf.cell(0,6,vs,ln=True,border=0)
            elif isinstance(data, pd.DataFrame):
                pdf.cell(0,7,f"Table: {len(data)} rows (see Excel)",ln=True)
            if key in self.plots:
                pdf.ln(2); pdf.image(self.plots[key],x=15,w=180); printed.add(key); pdf.ln(5)
            pdf.ln(8)
            if pdf.get_y() > 230: pdf.add_page()
        for key in [k for k in self.plots if k not in printed]:
            if pdf.get_y() > 100: pdf.add_page()
            pdf.set_font("Arial",'B',12); pdf.cell(0,10,key,ln=True)
            pdf.image(self.plots[key],x=15,w=180); pdf.ln(10)
            if pdf.get_y() > 230: pdf.add_page()
        raw = pdf.output()
        return bytes(raw) if isinstance(raw, bytearray) else raw
 
 
# ── FED DATA ──────────────────────────────────────────────────────────────────
FED_ASSETS = {
    "WGCAL":"Gold Certificate Account","SDRACL":"SDR Certificate Account",
    "WCOINL":"Coin","WSHONBLL":"Treasury Bills","WSHONBNL":"Treasury Notes and Bonds",
    "WSHONBIIL":"Treasury Tips","WSHOMCB":"Mortgage-Backed Securities",
    "WUDSHO":"Unamortized Premiums","WLCFOCEL":"Other Credit Extensions","WOTHAL":"Other Assets"
}
FED_LIABILITIES = {
    "WCURCIR":"Currency in Circulation","WDTPGCAS":"Treasury General Account (TGA)",
    "RRPONTSYD":"Overnight Reverse Repo (RRP)","WLRRAL":"Reverse Repos (Total)",
    "WLFN":"Federal Reserve Notes","WDFOL":"Foreign Official Deposits",
    "WDLTCL":"Term Deposits"
}
 
 
# ==========================================
# TRADE INTELLIGENCE MODULE (7-Layer System)
# Embedded inline — no external file needed
# ==========================================
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
try:
    from fpdf import FPDF
    import xlsxwriter
    EXPORT_AVAILABLE = True
except ImportError:
    EXPORT_AVAILABLE = False
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
 
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
# TRADE INTELLIGENCE MODULE (7-Layer System)
# ==========================================
# Embedded directly — no separate file needed
 
SECTOR_ETF_MAP = {
    "AAPL":"XLK","MSFT":"XLK","NVDA":"SOXX","AMD":"SOXX","INTC":"SOXX",
    "AVGO":"SOXX","QCOM":"SOXX","MU":"SOXX","AMAT":"SOXX","LRCX":"SOXX",
    "GOOGL":"XLC","GOOG":"XLC","META":"XLC","NFLX":"XLC","DIS":"XLC",
    "AMZN":"XLY","TSLA":"XLY","HD":"XLY","NKE":"XLY","MCD":"XLY",
    "JPM":"XLF","BAC":"XLF","WFC":"XLF","GS":"XLF","MS":"XLF","V":"XLF","MA":"XLF",
    "JNJ":"XLV","UNH":"XLV","PFE":"XLV","MRK":"XLV","ABBV":"XLV","LLY":"XLV",
    "XOM":"XLE","CVX":"XLE","COP":"XLE","SLB":"XLE","EOG":"XLE",
    "LIN":"XLB","APD":"XLB","NEM":"XLB","FCX":"XLB",
    "NEE":"XLU","DUK":"XLU","SO":"XLU","AEP":"XLU",
    "AMT":"XLRE","PLD":"XLRE","EQIX":"XLRE","SPG":"XLRE",
    "CAT":"XLI","BA":"XLI","HON":"XLI","GE":"XLI","RTX":"XLI","UPS":"XLI",
    "PG":"XLP","KO":"XLP","PEP":"XLP","WMT":"XLP","COST":"XLP","PM":"XLP",
}
SECTOR_NAMES = {
    "XLK":"Technology","SOXX":"Semiconductors","XLC":"Communication",
    "XLY":"Consumer Discret.","XLF":"Financials","XLV":"Healthcare",
    "XLE":"Energy","XLB":"Materials","XLU":"Utilities",
    "XLRE":"Real Estate","XLI":"Industrials","XLP":"Consumer Staples",
}
 
@st.cache_data(ttl=300, show_spinner=False)
def _ti_fetch_history(ticker: str, period: str = "1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty: return None
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df[['Open','High','Low','Close','Volume']].dropna()
    except: return None
 
@st.cache_data(ttl=3600, show_spinner=False)
def _ti_fetch_earnings(ticker: str):
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None: return None
        if isinstance(cal, pd.DataFrame):
            if 'Earnings Date' in cal.index:
                return pd.Timestamp(cal.loc['Earnings Date'].iloc[0]).date()
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date', [])
            if isinstance(ed, list) and len(ed) > 0:
                return pd.Timestamp(ed[0]).date()
        return None
    except: return None
 
def _ti_market_condition() -> dict:
    result = {'spy_trend':'Unknown','qqq_trend':'Unknown','iwm_trend':'Unknown',
               'vix_level':None,'vix_signal':'Unknown','breadth_score':0,
               'market_score':0,'market_verdict':'Unknown','details':{}}
    index_score = 0
    for sym, key in [("SPY","spy"),("QQQ","qqq"),("IWM","iwm")]:
        df = _ti_fetch_history(sym, "6mo")
        if df is None: continue
        close = df['Close']
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ema200 = close.rolling(200).mean()
        last = float(close.iloc[-1])
        a20 = last > float(ema20.iloc[-1])
        a50 = last > float(ema50.iloc[-1])
        a200 = last > float(ema200.iloc[-1]) if len(close) >= 200 else True
        slope20 = (float(ema20.iloc[-1]) - float(ema20.iloc[-5])) / float(ema20.iloc[-5]) * 100
        if a20 and a50 and a200 and slope20 > 0: trend = "STRONG UPTREND"; sc = 2
        elif a50 and a200: trend = "UPTREND"; sc = 1
        elif not a50 and not a200: trend = "DOWNTREND"; sc = -2
        else: trend = "MIXED/PULLBACK"; sc = 0
        result[f'{key}_trend'] = trend
        result['details'][sym] = {'last':round(last,2),'ema20':round(float(ema20.iloc[-1]),2),
                                   'ema50':round(float(ema50.iloc[-1]),2),
                                   'above20':a20,'above50':a50,'above200':a200,
                                   'slope20_pct':round(slope20,3)}
        index_score += sc
    vix_df = _ti_fetch_history("^VIX","6mo")
    if vix_df is not None:
        vix = float(vix_df['Close'].iloc[-1])
        vix_ma20 = float(vix_df['Close'].ewm(span=20).mean().iloc[-1])
        vix_spike = vix > vix_ma20 * 1.3
        result['vix_level'] = round(vix,2)
        result['details']['VIX'] = {'level':vix,'ma20':round(vix_ma20,2),'spike':vix_spike}
        if vix < 15: result['vix_signal'] = "CALM"; index_score += 1
        elif vix < 20: result['vix_signal'] = "NORMAL"
        elif vix < 30: result['vix_signal'] = "ELEVATED ⚠️"; index_score -= 1
        else: result['vix_signal'] = "EXTREME FEAR 🚨"; index_score -= 2
        if vix_spike: result['vix_signal'] += " ⚡SPIKE"; index_score -= 1
    sample = ["AAPL","MSFT","GS","JPM","HD","MCD","V","DIS","BA","CAT"]
    above50 = 0; checked = 0
    for sym in sample[:10]:
        df = _ti_fetch_history(sym,"3mo")
        if df is None: continue
        ema50 = df['Close'].ewm(span=50).mean()
        if float(df['Close'].iloc[-1]) > float(ema50.iloc[-1]): above50 += 1
        checked += 1
    breadth = (above50/checked*100) if checked > 0 else 50
    result['breadth_score'] = round(breadth,1)
    result['details']['Breadth'] = {'pct_above_50ema':breadth,'checked':checked}
    if breadth > 70: index_score += 1
    elif breadth < 40: index_score -= 1
    result['market_score'] = index_score
    if index_score >= 5: result['market_verdict'] = "STRONG BULL — High probability environment for longs"; result['market_color'] = "#00ff88"
    elif index_score >= 2: result['market_verdict'] = "BULL BIAS — Favorable for longs, stay selective"; result['market_color'] = "#44cc66"
    elif index_score >= 0: result['market_verdict'] = "NEUTRAL — Mixed signals, reduce size"; result['market_color'] = "#ffcc00"
    elif index_score >= -2: result['market_verdict'] = "BEAR BIAS — Avoid new longs, tight stops"; result['market_color'] = "#ff8844"
    else: result['market_verdict'] = "STRONG BEAR — Cash is king, no new longs"; result['market_color'] = "#ff4444"
    return result
 
def _ti_sector_strength(ticker: str, stock_df: pd.DataFrame) -> dict:
    sector_etf = SECTOR_ETF_MAP.get(ticker.upper())
    result = {'sector_etf':sector_etf,
               'sector_name':SECTOR_NAMES.get(sector_etf,"Unknown") if sector_etf else "Unknown",
               'sector_trend':'Unknown','stock_vs_sector':None,'stock_vs_spy':None,
               'sector_score':0,'sector_verdict':'Unknown','details':{}}
    if sector_etf is None:
        try:
            info = yf.Ticker(ticker).info
            result['sector_name'] = info.get('sector','Unknown')
        except: pass
        result['sector_verdict'] = "Sector ETF not mapped — manual check needed"
        result['sector_color'] = "#888888"
        return result
    sec_df = _ti_fetch_history(sector_etf,"6mo")
    spy_df = _ti_fetch_history("SPY","6mo")
    sc = 0
    if sec_df is not None:
        close = sec_df['Close']
        ema20 = close.ewm(span=20).mean(); ema50 = close.ewm(span=50).mean()
        last = float(close.iloc[-1])
        a20 = last > float(ema20.iloc[-1]); a50 = last > float(ema50.iloc[-1])
        mom4w = (last/float(close.iloc[-20])-1)*100 if len(close)>=20 else 0
        if a20 and a50 and mom4w>0: result['sector_trend']="STRONG"; sc+=2
        elif a50: result['sector_trend']="MODERATE"; sc+=1
        elif not a20 and not a50: result['sector_trend']="WEAK"; sc-=2
        else: result['sector_trend']="MIXED"
        result['details']['sector_etf']={'symbol':sector_etf,'last':round(last,2),
                                          'momentum_4w_pct':round(mom4w,2)}
    if sec_df is not None and len(stock_df)>20 and len(sec_df)>20:
        common = stock_df.index.intersection(sec_df.index)
        if len(common)>=20:
            stk = stock_df['Close'].loc[common]; sec = sec_df['Close'].loc[common]
            rs = (stk/stk.iloc[0])/(sec/sec.iloc[0])
            rs_slope = (float(rs.iloc[-1])-float(rs.iloc[-10]))/float(rs.iloc[-10])*100
            result['stock_vs_sector'] = round(rs_slope,2)
            if rs_slope>2: sc+=1
            elif rs_slope<-2: sc-=1
    if spy_df is not None and len(stock_df)>20:
        common = stock_df.index.intersection(spy_df.index)
        if len(common)>=20:
            stk=stock_df['Close'].loc[common]; spy=spy_df['Close'].loc[common]
            rs_spy=(stk/stk.iloc[0])/(spy/spy.iloc[0])
            rs_spy_slope=(float(rs_spy.iloc[-1])-float(rs_spy.iloc[-10]))/float(rs_spy.iloc[-10])*100
            result['stock_vs_spy']=round(rs_spy_slope,2)
            if rs_spy_slope>3: sc+=1
            elif rs_spy_slope<-3: sc-=1
    result['sector_score']=sc
    if sc>=3: result['sector_verdict']=f"STRONG TAILWIND ({result['sector_name']} bullish + outperforming)"; result['sector_color']="#00ff88"
    elif sc>=1: result['sector_verdict']=f"MODERATE TAILWIND ({result['sector_name']} holding up)"; result['sector_color']="#44cc66"
    elif sc==0: result['sector_verdict']=f"NEUTRAL ({result['sector_name']} mixed)"; result['sector_color']="#ffcc00"
    else: result['sector_verdict']=f"SECTOR HEADWIND ({result['sector_name']} weak — caution)"; result['sector_color']="#ff4444"
    return result
 
def _ti_price_action(df: pd.DataFrame) -> dict:
    result={'pattern':'Unknown','pattern_score':0,'trend_structure':'Unknown',
             'entry_type':'No clean entry yet','patterns':[],'details':{}}
    if df is None or len(df)<30: return result
    close=df['Close']; high=df['High']; low=df['Low']; vol=df['Volume']; op=df['Open']
    last=float(close.iloc[-1]); prev=float(close.iloc[-2])
    lh=float(high.iloc[-1]); ll=float(low.iloc[-1]); lo=float(op.iloc[-1])
    sc=0; patterns=[]
    highs=high.iloc[-20:].values; lows=low.iloc[-20:].values
    hh=all(highs[i]>=highs[i-1] for i in range(-5,-1))
    hl=all(lows[i]>=lows[i-1] for i in range(-5,-1))
    lhh=all(highs[i]<=highs[i-1] for i in range(-5,-1))
    ll2=all(lows[i]<=lows[i-1] for i in range(-5,-1))
    if hh and hl: result['trend_structure']="UPTREND (HH+HL)"; sc+=2
    elif lhh and ll2: result['trend_structure']="DOWNTREND (LH+LL)"; sc-=2
    else: result['trend_structure']="CHOPPY/SIDEWAYS"
    day_range=lh-ll
    close_pct=0.5
    if day_range>0:
        close_pct=(last-ll)/day_range
        result['details']['close_pct_of_range']=round(close_pct*100,1)
        if close_pct>0.80: patterns.append("Close near HOD ✅"); sc+=1
        elif close_pct<0.25: patterns.append("Close near LOD ⚠️"); sc-=1
    body=abs(last-lo); upper_wick=lh-max(last,lo); lower_wick=min(last,lo)-ll
    if body>0:
        if lower_wick>body*1.5 and lower_wick>upper_wick: patterns.append("Bullish Hammer/Wick ✅"); sc+=1
        if upper_wick>body*1.5 and upper_wick>lower_wick: patterns.append("Bearish Shooting Star ⚠️"); sc-=1
    if lh<=float(high.iloc[-2]) and ll>=float(low.iloc[-2]): patterns.append("Inside Bar (watch breakout)")
    high_20=float(high.iloc[-21:-1].max()) if len(df)>=21 else float(high.iloc[:-1].max())
    low_20=float(low.iloc[-21:-1].min()) if len(df)>=21 else float(low.iloc[:-1].min())
    if last>high_20: patterns.append("Breakout above 20-day high 🚀"); sc+=2
    if last<low_20: patterns.append("Breakdown below 20-day low ⚠️"); sc-=2
    ema20=float(close.ewm(span=20).mean().iloc[-1])
    ema50=float(close.ewm(span=50).mean().iloc[-1])
    d20=(last-ema20)/ema20*100; d50=(last-ema50)/ema50*100
    if -1.5<d20<1.5 and hh and hl: patterns.append("Pullback to 20 EMA in uptrend ✅"); sc+=2
    if -2.0<d50<2.0 and hh and hl: patterns.append("Pullback to 50 EMA in uptrend ✅"); sc+=1
    if float(high.iloc[-2])>high_20 and last<high_20 and last<float(close.iloc[-2]):
        patterns.append("Failed breakout ⚠️"); sc-=2
    avg_vol=float(vol.rolling(20).mean().iloc[-1]); last_vol=float(vol.iloc[-1])
    rvol=last_vol/avg_vol if avg_vol>0 else 1.0
    result['details']['rvol']=round(rvol,2)
    if rvol>1.5 and last>prev: patterns.append(f"High RVOL ({rvol:.1f}x) on up day ✅"); sc+=1
    elif rvol>1.5 and last<prev: patterns.append(f"High RVOL ({rvol:.1f}x) on down day ⚠️"); sc-=1
    result['pattern_score']=sc; result['patterns']=patterns
    result['details'].update({'last':last,'ema20':round(ema20,2),'ema50':round(ema50,2),
                               'dist_ema20_pct':round(d20,2),'dist_ema50_pct':round(d50,2),
                               'high_20d':round(high_20,2),'low_20d':round(low_20,2),'rvol':round(rvol,2)})
    if last>high_20 and rvol>1.3: result['entry_type']="🚀 Breakout Entry"
    elif -2.0<d20<0 and hh and hl: result['entry_type']="📉 Pullback Entry (Trend Continuation)"
    elif -5.0<d50<0 and close_pct>0.6: result['entry_type']="🔄 Mean Reversion Bounce"
    elif last<low_20: result['entry_type']="🚫 No Entry — Breakdown"
    elif 'Inside Bar' in str(patterns): result['entry_type']="⏳ Wait — Inside Bar Compression"
    elif abs(d20)>8: result['entry_type']="⚠️ Overextended — Wait for Pullback"
    elif lhh or ll2: result['entry_type']="❌ Lower High Rejection — Avoid Longs"
    else: result['entry_type']="👀 Watch — No High-Conviction Setup"
    return result
 
def _ti_support_resistance(df: pd.DataFrame) -> dict:
    result={'nearest_support':None,'nearest_resistance':None,'risk_reward':None,
             'rr_verdict':'Unknown','support_levels':[],'resistance_levels':[],'details':{}}
    if df is None or len(df)<30: return result
    close=df['Close']; high=df['High']; low=df['Low']
    last=float(close.iloc[-1])
    window=10; swing_highs=[]; swing_lows=[]
    for i in range(window, len(df)-window):
        if float(high.iloc[i])==float(high.iloc[i-window:i+window].max()): swing_highs.append(float(high.iloc[i]))
        if float(low.iloc[i])==float(low.iloc[i-window:i+window].min()): swing_lows.append(float(low.iloc[i]))
    ema20=float(close.ewm(span=20).mean().iloc[-1]); ema50=float(close.ewm(span=50).mean().iloc[-1])
    ma200=float(close.rolling(200).mean().iloc[-1]) if len(close)>=200 else None
    sup=[s for s in swing_lows if s<last]; res=[r for r in swing_highs if r>last]
    for ma in [ema20,ema50,ma200]:
        if ma is None: continue
        if ma<last: sup.append(ma)
        else: res.append(ma)
    sup=sorted(sup,reverse=True)[:3]; res=sorted(res)[:3]
    ns=sup[0] if sup else last*0.95; nr=res[0] if res else last*1.05
    ds=(last-ns)/last*100; dr=(nr-last)/last*100
    rr=dr/ds if ds>0 else 0
    result.update({'nearest_support':round(ns,2),'nearest_resistance':round(nr,2),
                   'dist_to_support_pct':round(ds,2),'dist_to_resistance_pct':round(dr,2),
                   'risk_reward':round(rr,2),'support_levels':[round(s,2) for s in sup],
                   'resistance_levels':[round(r,2) for r in res]})
    if rr>=3.0: result['rr_verdict']="EXCELLENT R:R (≥3:1) ✅"; result['rr_color']="#00ff88"
    elif rr>=2.0: result['rr_verdict']="GOOD R:R (≥2:1) ✅"; result['rr_color']="#44cc66"
    elif rr>=1.5: result['rr_verdict']="ACCEPTABLE R:R (≥1.5:1)"; result['rr_color']="#ffcc00"
    elif rr>=1.0: result['rr_verdict']="MARGINAL R:R — Reduce size"; result['rr_color']="#ff8844"
    else: result['rr_verdict']="POOR R:R (<1:1) — Skip ❌"; result['rr_color']="#ff4444"
    return result
 
def _ti_volume_liquidity(df: pd.DataFrame, ticker: str) -> dict:
    result={'avg_dollar_vol':None,'rvol':None,'vol_trend':'Unknown',
             'liquidity_ok':False,'options_liquid':False,'details':{}}
    if df is None or len(df)<20: return result
    close=df['Close']; vol=df['Volume']
    lp=float(close.iloc[-1]); lv=float(vol.iloc[-1])
    avg20=float(vol.rolling(20).mean().iloc[-1]); adv=lp*avg20
    rvol=lv/avg20 if avg20>0 else 1.0
    v5=vol.iloc[-5:].mean(); v10_15=vol.iloc[-15:-5].mean()
    vt="INCREASING" if v5>v10_15*1.1 else "DECREASING" if v5<v10_15*0.9 else "STABLE"
    liq=adv>=20_000_000 and lp>=5.0
    opt_liq=False
    try:
        tk=yf.Ticker(ticker); exps=tk.options
        if exps:
            chain=tk.option_chain(exps[0])
            oi=chain.calls['openInterest'].fillna(0).sum()+chain.puts['openInterest'].fillna(0).sum()
            opt_liq=oi>1000
    except: pass
    adv_fmt=f"${adv/1e6:.1f}M" if adv>=1e6 else f"${adv/1e3:.0f}K"
    result.update({'avg_dollar_vol':adv,'avg_dollar_vol_fmt':adv_fmt,'rvol':round(rvol,2),
                   'vol_trend':vt,'liquidity_ok':liq,'options_liquid':opt_liq,'last_price':round(lp,2)})
    if not liq:
        if lp<5.0: result['liquidity_verdict']="❌ Price below $5 — skip (penny stock)"; result['liquidity_color']="#ff4444"
        else: result['liquidity_verdict']="❌ Avg dollar vol < $20M — too illiquid"; result['liquidity_color']="#ff4444"
    elif rvol>2.0: result['liquidity_verdict']=f"✅ Liquid + High RVOL ({rvol:.1f}x)"; result['liquidity_color']="#00ff88"
    elif rvol>1.2: result['liquidity_verdict']=f"✅ Liquid + Above-avg volume ({rvol:.1f}x)"; result['liquidity_color']="#44cc66"
    else: result['liquidity_verdict']=f"✅ Liquid but low RVOL ({rvol:.1f}x) — watch for confirmation"; result['liquidity_color']="#ffcc00"
    return result
 
def _ti_earnings_risk(ticker: str) -> dict:
    result={'next_earnings':None,'days_to_earnings':None,'earnings_risk':'Unknown',
             'earnings_color':'#aaaaaa','macro_events':[],'risk_score':0,'details':{}}
    ed=_ti_fetch_earnings(ticker); today=datetime.now().date()
    if ed:
        days=(ed-today).days; result['next_earnings']=str(ed); result['days_to_earnings']=days
        if 0<=days<=3: result['earnings_risk']=f"🚨 EARNINGS IN {days} DAYS — Extreme risk, avoid new entries"; result['earnings_color']="#ff4444"; result['risk_score']=-3
        elif days<=7: result['earnings_risk']=f"⚠️ EARNINGS IN {days} DAYS — High IV, binary event risk"; result['earnings_color']="#ff8844"; result['risk_score']=-2
        elif days<=14: result['earnings_risk']=f"⚠️ Earnings in {days} days — Size down"; result['earnings_color']="#ffcc00"; result['risk_score']=-1
        elif days<=30: result['earnings_risk']=f"📅 Earnings in {days} days — On radar"; result['earnings_color']="#88aaff"; result['risk_score']=0
        else: result['earnings_risk']=f"✅ Next earnings {days} days away"; result['earnings_color']="#44cc66"; result['risk_score']=1
    else:
        result['earnings_risk']="📅 Earnings date unavailable — check manually"; result['earnings_color']="#888888"
    try:
        info=yf.Ticker(ticker).info
        sp=info.get('shortPercentOfFloat',0) or 0
        if sp>0.20: result['macro_events'].append(f"⚠️ High short interest ({sp:.1%})"); result['risk_score']-=1
        beta=info.get('beta',1.0) or 1.0
        if beta>2.0: result['macro_events'].append(f"⚡ High beta ({beta:.1f}) — amplified macro sensitivity")
        industry=info.get('industry','').lower()
        if any(k in industry for k in ['biotech','pharmaceutical','drug','clinical']):
            result['macro_events'].append("🧬 Biotech/Pharma — FDA catalyst risk"); result['risk_score']-=1
        result['details']={'short_pct':round(sp*100,1),'beta':round(beta,2),'sector':info.get('sector','N/A')}
    except: pass
    return result
 
def _ti_compute_all(ticker: str) -> dict:
    df = _ti_fetch_history(ticker, "1y")
    with st.spinner("Layer 1: Market condition..."): market = _ti_market_condition()
    with st.spinner("Layer 2: Sector strength..."): sector = _ti_sector_strength(ticker, df if df is not None else pd.DataFrame())
    with st.spinner("Layer 3: Price action..."): pa = _ti_price_action(df)
    with st.spinner("Layer 4: Support & resistance..."): sr = _ti_support_resistance(df)
    with st.spinner("Layer 5: Volume & liquidity..."): volume = _ti_volume_liquidity(df, ticker)
    with st.spinner("Layer 6: Earnings & news risk..."): earnings = _ti_earnings_risk(ticker)
    composite = 0
    composite += min(market['market_score'], 4)
    composite += min(sector['sector_score'], 3)
    composite += min(pa['pattern_score'], 4)
    composite += (1 if (sr.get('risk_reward') or 0) >= 2 else 0 if (sr.get('risk_reward') or 0) >= 1.5 else -1)
    composite += (1 if volume['liquidity_ok'] else -2)
    composite += earnings['risk_score']
    entry_type = pa.get('entry_type', 'No clean entry yet')
    if composite >= 10: final_verdict = "STRONG BUY"; final_color = "#00ff88"
    elif composite >= 6: final_verdict = "BUY"; final_color = "#44cc66"
    elif composite >= 3: final_verdict = "WATCH / WAIT"; final_color = "#ffcc00"
    elif composite >= 0: final_verdict = "NEUTRAL — SKIP"; final_color = "#aaaaaa"
    else: final_verdict = "AVOID"; final_color = "#ff4444"
    if earnings.get('days_to_earnings') is not None and earnings['days_to_earnings'] <= 7:
        if final_verdict in ["STRONG BUY","BUY"]: final_verdict = "WATCH — EARNINGS RISK"; final_color = "#ff8844"
    if market['market_score'] <= -3 and final_verdict in ["STRONG BUY","BUY"]:
        final_verdict = "WATCH — MARKET HEADWIND"; final_color = "#ff8844"
    if not volume['liquidity_ok'] and final_verdict in ["STRONG BUY","BUY"]:
        final_verdict = "AVOID — ILLIQUID"; final_color = "#ff4444"
    return {'ticker':ticker,'df':df,'market':market,'sector':sector,'price_action':pa,
            'sr':sr,'volume':volume,'earnings':earnings,'composite_score':composite,
            'final_verdict':final_verdict,'final_color':final_color,'entry_type':entry_type}
 
def render_trade_intelligence(default_ticker: str = "AAPL"):
    st.write("### 🎯 Trade Intelligence — 7-Layer Institutional Confirmation")
    st.markdown("""
    A signal alone is not enough. This runs **7 independent confirmation layers** before
    declaring a trade valid — the same logic institutional desks use before sizing in.
    """)
    col_in1, col_in2 = st.columns([1,3])
    with col_in1:
        ti_ticker = st.text_input("Ticker to Analyze", default_ticker.upper(), key="ti_ticker_input").upper()
        run_ti = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True, key="run_ti_btn")
    if not run_ti:
        st.info("Enter a ticker and click **Run Full Analysis** to see all 7 layers.")
        return
    result = _ti_compute_all(ti_ticker)
    st.divider()
    # ── MASTER VERDICT BANNER ─────────────────────────────────────────────────
    v_col, s_col, e_col = st.columns(3)
    with v_col:
        st.markdown(f"""<div style="background:{result['final_color']}22;border:2px solid {result['final_color']};
            border-radius:12px;padding:18px;text-align:center;">
            <h2 style="color:{result['final_color']};margin:0;">{result['final_verdict']}</h2>
            <p style="margin:4px 0;color:#ccc;font-size:0.9em;">{ti_ticker}</p></div>""",
            unsafe_allow_html=True)
    with s_col:
        sc = result['composite_score']
        bar_pct = min(100, max(0, (sc+5)/15*100))
        st.markdown(f"""<div style="background:#1a1a2e;border-radius:12px;padding:18px;text-align:center;">
            <p style="color:#aaa;margin:0;font-size:0.85em;">COMPOSITE SCORE</p>
            <h2 style="color:{result['final_color']};margin:4px 0;">{sc} / 15</h2>
            <div style="background:#333;border-radius:4px;height:8px;margin-top:8px;">
            <div style="background:{result['final_color']};width:{bar_pct:.0f}%;height:8px;border-radius:4px;"></div>
            </div></div>""", unsafe_allow_html=True)
    with e_col:
        st.markdown(f"""<div style="background:#1a1a2e;border-radius:12px;padding:18px;text-align:center;">
            <p style="color:#aaa;margin:0;font-size:0.85em;">ENTRY TYPE</p>
            <h3 style="color:#ffffff;margin:4px 0;font-size:1.1em;">{result['entry_type']}</h3>
            </div>""", unsafe_allow_html=True)
    st.divider()
    # ── LAYER 1: MARKET CONDITION ─────────────────────────────────────────────
    with st.expander("📊 Layer 1 — Market Condition", expanded=True):
        mkt = result['market']
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("SPY", mkt['spy_trend']); mc2.metric("QQQ", mkt['qqq_trend'])
        mc3.metric("IWM", mkt['iwm_trend'])
        mc4.metric("VIX", f"{mkt['vix_level']}" if mkt['vix_level'] else "N/A", delta=mkt['vix_signal'])
        bc1,bc2 = st.columns(2)
        bc1.metric("Breadth (% stocks above 50 EMA)", f"{mkt['breadth_score']:.0f}%")
        bc2.metric("Market Score", f"{mkt['market_score']}")
        st.markdown(f"""<div style="background:{mkt.get('market_color','#333')}22;
            border-left:4px solid {mkt.get('market_color','#888')};padding:10px;border-radius:4px;">
            <b>{mkt['market_verdict']}</b></div>""", unsafe_allow_html=True)
        det = mkt.get('details',{})
        rows=[]
        for sym in ['SPY','QQQ','IWM']:
            if sym in det:
                d=det[sym]
                rows.append({'Index':sym,'Last':d['last'],'EMA20':d['ema20'],'EMA50':d['ema50'],
                             'Above 20':'✅' if d['above20'] else '❌',
                             'Above 50':'✅' if d['above50'] else '❌',
                             'Above 200':'✅' if d['above200'] else '❌',
                             '20d Slope':f"{d['slope20_pct']:+.2f}%"})
        if rows: st.dataframe(pd.DataFrame(rows).set_index('Index'), use_container_width=True)
    # ── LAYER 2: SECTOR STRENGTH ──────────────────────────────────────────────
    with st.expander("🏭 Layer 2 — Sector Strength", expanded=True):
        sec = result['sector']
        sc1,sc2,sc3 = st.columns(3)
        sc1.metric("Sector ETF", sec['sector_etf'] or "N/A")
        sc2.metric("Sector", sec['sector_name'])
        sc3.metric("Sector Trend", sec['sector_trend'])
        s1,s2 = st.columns(2)
        s1.metric(f"Stock vs {sec['sector_etf']} (10d RS)",
                  f"{sec['stock_vs_sector']:+.2f}%" if sec['stock_vs_sector'] is not None else "N/A")
        s2.metric("Stock vs SPY (10d RS)",
                  f"{sec['stock_vs_spy']:+.2f}%" if sec['stock_vs_spy'] is not None else "N/A")
        st.markdown(f"""<div style="background:{sec.get('sector_color','#333')}22;
            border-left:4px solid {sec.get('sector_color','#888')};padding:10px;border-radius:4px;">
            <b>{sec['sector_verdict']}</b></div>""", unsafe_allow_html=True)
        if sec['sector_etf'] and result['df'] is not None:
            sec_df_v = _ti_fetch_history(sec['sector_etf'],"6mo")
            if sec_df_v is not None and len(result['df'])>10:
                common=result['df'].index.intersection(sec_df_v.index)
                if len(common)>10:
                    stk_n=(result['df']['Close'].loc[common]/float(result['df']['Close'].loc[common].iloc[0]))*100
                    sec_n=(sec_df_v['Close'].loc[common]/float(sec_df_v['Close'].loc[common].iloc[0]))*100
                    fig_sec=go.Figure()
                    fig_sec.add_trace(go.Scatter(x=common,y=stk_n,name=ti_ticker,line=dict(color='#00f2ff',width=2)))
                    fig_sec.add_trace(go.Scatter(x=common,y=sec_n,name=sec['sector_etf'],line=dict(color='#ff6b35',width=2)))
                    fig_sec.update_layout(title="Stock vs Sector ETF (Indexed to 100)",template="plotly_dark",height=280,hovermode="x unified")
                    st.plotly_chart(fig_sec,use_container_width=True)
    # ── LAYER 3: PRICE ACTION ─────────────────────────────────────────────────
    with st.expander("📈 Layer 3 — Price Action", expanded=True):
        pa = result['price_action']
        pa1,pa2,pa3 = st.columns(3)
        pa1.metric("Trend Structure", pa['trend_structure'])
        pa2.metric("Pattern Score", pa['pattern_score'])
        pa3.metric("Entry Type", pa['entry_type'])
        det=pa.get('details',{})
        if det:
            pa4,pa5,pa6=st.columns(3)
            pa4.metric("Dist from EMA20",f"{det.get('dist_ema20_pct',0):+.2f}%")
            pa5.metric("Dist from EMA50",f"{det.get('dist_ema50_pct',0):+.2f}%")
            pa6.metric("RVOL",f"{det.get('rvol',1):.2f}x")
        if pa.get('patterns'):
            st.write("**Detected patterns:**")
            for p in pa['patterns']: st.markdown(f"- {p}")
        if result['df'] is not None and len(result['df'])>=50:
            df_v=result['df'].tail(120)
            ema20_s=df_v['Close'].ewm(span=20).mean(); ema50_s=df_v['Close'].ewm(span=50).mean()
            fig_pa=go.Figure()
            fig_pa.add_trace(go.Candlestick(x=df_v.index,open=df_v['Open'],high=df_v['High'],
                low=df_v['Low'],close=df_v['Close'],name='Price',
                increasing_line_color='#00ff88',decreasing_line_color='#ff4444'))
            fig_pa.add_trace(go.Scatter(x=df_v.index,y=ema20_s,line=dict(color='orange',width=1.5),name='EMA20'))
            fig_pa.add_trace(go.Scatter(x=df_v.index,y=ema50_s,line=dict(color='#a855f7',width=1.5),name='EMA50'))
            if det.get('high_20d'): fig_pa.add_hline(y=det['high_20d'],line_dash="dash",line_color="#00f2ff",annotation_text="20d High")
            if det.get('low_20d'): fig_pa.add_hline(y=det['low_20d'],line_dash="dash",line_color="#ff8844",annotation_text="20d Low")
            fig_pa.update_layout(title=f"{ti_ticker} Price Action (120d)",template="plotly_dark",height=380,xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_pa,use_container_width=True)
    # ── LAYER 4: SUPPORT & RESISTANCE ─────────────────────────────────────────
    with st.expander("📍 Layer 4 — Support & Resistance", expanded=True):
        sr_r=result['sr']
        sr1,sr2,sr3,sr4=st.columns(4)
        sr1.metric("Nearest Support",f"${sr_r.get('nearest_support','N/A')}")
        sr2.metric("Nearest Resistance",f"${sr_r.get('nearest_resistance','N/A')}")
        sr3.metric("Dist to Support",f"{sr_r.get('dist_to_support_pct',0):.2f}%")
        sr4.metric("Dist to Resistance",f"{sr_r.get('dist_to_resistance_pct',0):.2f}%")
        st.markdown(f"""<div style="background:{sr_r.get('rr_color','#333')}22;
            border-left:4px solid {sr_r.get('rr_color','#888')};padding:12px;border-radius:4px;">
            <b>Risk:Reward = {sr_r.get('risk_reward','N/A')} : 1 — {sr_r.get('rr_verdict','')}</b>
            </div>""", unsafe_allow_html=True)
        if result['df'] is not None:
            df_sr=result['df'].tail(60)
            fig_sr=go.Figure()
            fig_sr.add_trace(go.Candlestick(x=df_sr.index,open=df_sr['Open'],high=df_sr['High'],
                low=df_sr['Low'],close=df_sr['Close'],name='Price',
                increasing_line_color='#00ff88',decreasing_line_color='#ff4444'))
            for lv in sr_r.get('support_levels',[]):
                fig_sr.add_hline(y=lv,line_dash="dot",line_color="#00ff88",opacity=0.6,annotation_text=f"SUP {lv:.2f}")
            for lv in sr_r.get('resistance_levels',[]):
                fig_sr.add_hline(y=lv,line_dash="dot",line_color="#ff4444",opacity=0.6,annotation_text=f"RES {lv:.2f}")
            fig_sr.update_layout(title=f"{ti_ticker} Support & Resistance (60d)",template="plotly_dark",height=360,xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sr,use_container_width=True)
    # ── LAYER 5: VOLUME & LIQUIDITY ───────────────────────────────────────────
    with st.expander("💧 Layer 5 — Volume & Liquidity", expanded=True):
        vol_r=result['volume']
        vc1,vc2,vc3,vc4=st.columns(4)
        vc1.metric("Avg Dollar Vol",vol_r.get('avg_dollar_vol_fmt','N/A'))
        vc2.metric("Relative Volume",f"{vol_r.get('rvol',0):.2f}x")
        vc3.metric("Volume Trend",vol_r.get('vol_trend','N/A'))
        vc4.metric("Options Liquid","✅ Yes" if vol_r.get('options_liquid') else "❌ No")
        st.markdown(f"""<div style="background:{vol_r.get('liquidity_color','#333')}22;
            border-left:4px solid {vol_r.get('liquidity_color','#888')};padding:10px;border-radius:4px;">
            <b>{vol_r.get('liquidity_verdict','')}</b></div>""", unsafe_allow_html=True)
        if result['df'] is not None:
            df_vol=result['df'].tail(40)
            avg20=float(df_vol['Volume'].rolling(20).mean().iloc[-1])
            bar_colors=['#00ff88' if c>=o else '#ff4444' for c,o in zip(df_vol['Close'],df_vol['Open'])]
            fig_vol_r=go.Figure()
            fig_vol_r.add_trace(go.Bar(x=df_vol.index,y=df_vol['Volume'],marker_color=bar_colors,opacity=0.8))
            fig_vol_r.add_hline(y=avg20,line_dash="dash",line_color="white",annotation_text=f"Avg ({avg20/1e6:.1f}M)")
            fig_vol_r.update_layout(title="Volume (40d)",template="plotly_dark",height=220,margin=dict(t=30,b=10))
            st.plotly_chart(fig_vol_r,use_container_width=True)
    # ── LAYER 6: EARNINGS & NEWS RISK ────────────────────────────────────────
    with st.expander("📅 Layer 6 — Earnings & News Risk", expanded=True):
        earn=result['earnings']
        ec1,ec2=st.columns(2)
        ec1.metric("Next Earnings",earn.get('next_earnings','Unknown'))
        ec2.metric("Days Away",earn.get('days_to_earnings','N/A'))
        det_e=earn.get('details',{})
        if det_e:
            ed1,ed2=st.columns(2)
            ed1.metric("Short Interest",f"{det_e.get('short_pct',0):.1f}%")
            ed2.metric("Beta",det_e.get('beta','N/A'))
        st.markdown(f"""<div style="background:{earn.get('earnings_color','#333')}22;
            border-left:4px solid {earn.get('earnings_color','#888')};padding:10px;border-radius:4px;">
            <b>{earn.get('earnings_risk','')}</b></div>""", unsafe_allow_html=True)
        if earn.get('macro_events'):
            st.write("**Additional risk flags:**")
            for ev in earn['macro_events']: st.markdown(f"- {ev}")
    # ── LAYER 7: ENTRY QUALITY CHECKLIST ─────────────────────────────────────
    with st.expander("🏆 Layer 7 — Entry Quality Checklist", expanded=True):
        checks=[
            ("Market environment favorable?", result['market']['market_score']>=2, result['market']['market_verdict']),
            ("Sector tailwind present?", result['sector']['sector_score']>=1, result['sector']['sector_verdict']),
            ("Price action constructive?", result['price_action']['pattern_score']>=1, result['price_action']['trend_structure']),
            ("Clean entry pattern?", result['price_action']['entry_type'] not in
             ["No clean entry yet","⏳ Wait — Inside Bar Compression","⚠️ Overextended — Wait for Pullback",
              "❌ Lower High Rejection — Avoid Longs","🚫 No Entry — Breakdown","👀 Watch — No High-Conviction Setup"],
             result['price_action']['entry_type']),
            ("Risk:Reward ≥ 2:1?", (result['sr'].get('risk_reward') or 0)>=2.0, f"R:R = {result['sr'].get('risk_reward','N/A')}"),
            ("Stock is liquid?", result['volume']['liquidity_ok'], result['volume'].get('liquidity_verdict','')),
            ("No imminent earnings risk?",
             (result['earnings'].get('days_to_earnings') is None or result['earnings'].get('days_to_earnings',999)>14),
             result['earnings'].get('earnings_risk','')),
        ]
        passed=sum(1 for _,ok,_ in checks if ok)
        for label,ok,detail in checks:
            icon="✅" if ok else "❌"; color="#00ff88" if ok else "#ff4444"
            st.markdown(
                f"<div style='display:flex;align-items:center;margin:4px 0;padding:8px;"
                f"background:{'#00ff8811' if ok else '#ff444411'};border-radius:6px;'>"
                f"<span style='font-size:1.2em;margin-right:10px;'>{icon}</span>"
                f"<div><b style='color:{color}'>{label}</b>"
                f"<br><span style='color:#aaa;font-size:0.85em;'>{detail}</span></div></div>",
                unsafe_allow_html=True)
        st.divider()
        pct=passed/len(checks)*100
        st.markdown(f"""<div style='background:{result['final_color']}22;border:2px solid {result['final_color']};
            border-radius:12px;padding:20px;text-align:center;margin-top:12px;'>
            <h3 style='color:{result['final_color']};margin:0;'>{passed}/{len(checks)} checks passed ({pct:.0f}%)</h3>
            <h2 style='color:{result['final_color']};margin:8px 0;'>{result['final_verdict']}</h2>
            <p style='color:#ccc;margin:0;'>{result['entry_type']}</p></div>""", unsafe_allow_html=True)
        layer_scores=[
            ("Market", min(result['market']['market_score'],4)),
            ("Sector", min(result['sector']['sector_score'],3)),
            ("Price Action", min(result['price_action']['pattern_score'],4)),
            ("R:R", 1 if (result['sr'].get('risk_reward') or 0)>=2 else 0 if (result['sr'].get('risk_reward') or 0)>=1.5 else -1),
            ("Liquidity", 1 if result['volume']['liquidity_ok'] else -2),
            ("Earnings", result['earnings']['risk_score']),
        ]
        bar_colors=['#00ff88' if s>0 else '#ff4444' if s<0 else '#aaaaaa' for _,s in layer_scores]
        fig_sc=go.Figure(go.Bar(x=[l for l,_ in layer_scores],y=[s for _,s in layer_scores],
                                 marker_color=bar_colors,text=[f"{s:+d}" for _,s in layer_scores],textposition='outside'))
        fig_sc.add_hline(y=0,line_color="white",opacity=0.5)
        fig_sc.update_layout(title="Layer Score Breakdown",template="plotly_dark",height=260,margin=dict(t=30,b=10))
        st.plotly_chart(fig_sc,use_container_width=True)
 
 
 
# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Thesis Parameters")
    market_region = st.selectbox("Market Region", ["US Market (USD)","Indian Market (INR)","Futures / Commodities (USD)"])
    if market_region == "Indian Market (INR)":
        CURRENCY="₹"; BENCHMARK="^NSEI"; DEFAULT_RF=7.0; SUFFIX=".NS"
    elif market_region == "Futures / Commodities (USD)":
        CURRENCY="$"; BENCHMARK="GC=F"; DEFAULT_RF=4.0; SUFFIX=""
    else:
        CURRENCY="$"; BENCHMARK="SPY"; DEFAULT_RF=4.0; SUFFIX=""
 
    col_t1,col_t2 = st.columns(2)
    with col_t1:
        raw_ticker = st.text_input("Main Ticker","RELIANCE" if market_region=="Indian Market (INR)" else "AAPL").upper()
    with col_t2:
        raw_pair = st.text_input("Pair Ticker","").upper()
 
    TICKER = raw_ticker+SUFFIX if (SUFFIX and not raw_ticker.endswith(SUFFIX)) else raw_ticker
    PAIR_TICKER = raw_pair+SUFFIX if (SUFFIX and raw_pair and not raw_pair.endswith(SUFFIX)) else raw_pair
    st.caption(f"Active Ticker: **{TICKER}**")
 
    start_date = st.date_input("Start Date", datetime.now()-timedelta(days=365))
    end_date = st.date_input("End Date", datetime.now())
    rf_rate = st.number_input("Risk Free Rate (%)",0.0,20.0,DEFAULT_RF)/100
    st.info(f"Benchmark: {BENCHMARK} | Currency: {CURRENCY}")
    st.divider()
 
    st.header("🔬 Model Config")
    regime_mode = st.selectbox("Regime Mode",
        ["Fixed: 4 States (Inst.)","Fixed: 2 States (Bull/Bear)","Fixed: 3 States","Auto: Best Fit (BIC)"])
    regime_val_map = {"Fixed: 4 States (Inst.)":4,"Fixed: 2 States (Bull/Bear)":2,
                      "Fixed: 3 States":3,"Auto: Best Fit (BIC)":"Auto"}
    regime_param = regime_val_map[regime_mode]
 
    with st.expander("Advanced Model Sync"):
        reg_engine = st.selectbox("Engine",["Markov (High Accuracy)","GMM (Fast)"])
        reg_engine_param = "Markov" if "Markov" in reg_engine else "GMM"
        reg_stability = st.slider("Signal Stability",0,10,4)
        reg_opt_goal = st.selectbox("Optimization Goal",["Robustness (BIC)","Performance (PnL)"])
        reg_switch_vol = st.toggle("Switching Volatility",value=True)
        reg_switch_trend = st.toggle("Switching Mean",value=True)
        initial_cap = st.number_input("Initial Capital",1000,1000000,10000)
        use_ts = st.toggle("Trailing Stop Loss",value=False)
        trailing_stop = st.slider("Trailing Stop (%)",0.0,20.0,5.0,step=0.5)/100 if use_ts else 0.0
        use_sl = st.toggle("Hard Stop Loss",value=True)
        stop_loss = st.slider("Hard Stop (%)",0.0,30.0,8.0,step=0.5)/100 if use_sl else 0.0
 
    st.divider()
    st.header("⚡ Live Mode")
    live_mode = st.toggle("Enable Live Data",value=False)
    if live_mode:
        data_interval = st.selectbox("Interval",["1m","5m","15m","60m"],index=1)
        if st.button("🔄 Refresh",use_container_width=True):
            st.cache_data.clear(); st.rerun()
    else:
        data_interval = '1d'
 
    st.divider()
    st.header("📥 Export")
    if not EXPORT_AVAILABLE:
        st.error("Missing: fpdf2, xlsxwriter"); st.info("pip install fpdf2 xlsxwriter")
    else:
        if 'report_gen' not in st.session_state: st.session_state.report_gen = None
        if st.session_state.report_gen:
            c1,c2 = st.columns(2)
            with c1:
                try:
                    raw = st.session_state.report_gen.generate_pdf()
                    st.download_button("📄 PDF",bytes(raw) if isinstance(raw,bytearray) else raw,
                        f"Quant_{TICKER}.pdf","application/pdf")
                except Exception as e: st.error(f"PDF: {e}")
            with c2:
                try:
                    st.download_button("📊 Excel",st.session_state.report_gen.generate_excel(),
                        f"Quant_{TICKER}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as e: st.error(f"Excel: {e}")
        else: st.caption("Run analysis to enable exports.")
 
 
# ==========================================
# DATA LOADING
# ==========================================
now_r = datetime.now().replace(second=0, microsecond=0)
if live_mode:
    lb = 7 if data_interval=='1m' else 30
    df_main = load_data(TICKER, now_r-timedelta(days=lb), now_r, interval=data_interval)
else:
    df_main = load_data(TICKER, start_date, end_date, interval='1d')
 
st.subheader("📈 Asset & Macro Analysis Suite")
 
# ==========================================
# TABS
# ==========================================
tabs = st.tabs([
    "💡 Decision","📉 Volatility","🔀 Regime","🎲 Stochastic",
    "📡 Kalman","🌍 Macro","🏗️ Structural","🛠️ Backtest",
    "🌩️ Vol Clustering","🧠 Adv Regime","📐 SML Alpha",
    "🔍 IV Scanner","📡 Market Scan","🏦 FED Balance",
    "📊 Options IV","〰️ Hurst","🔥 Hot 10","🏆 Trade Intelligence"
])
(tab0,tab1,tab2,tab3,tab4,tab5,tab6,tab7,
 tab8,tab9,tab10,tab11,tab12,tab13,tab14,tab15,tab16,tab17) = tabs
 
# Global signals
if df_main is not None:
    st.session_state.report_gen = ReportGenerator(TICKER, start_date, end_date)
    st.session_state.report_gen.add_data("Historical Data", df_main.tail(100))
    with st.sidebar:
        st.divider(); pb = st.progress(0); st.caption("Computing signals...")
    analysis = get_master_signal(TICKER, df_main, n_regimes=regime_param, freq='Daily',
                                  opt_goal=reg_opt_goal, stability=reg_stability,
                                  switch_vol=reg_switch_vol, switch_trend=reg_switch_trend,
                                  engine=reg_engine_param, initial_cap=initial_cap,
                                  trailing_stop=trailing_stop, stop_loss=stop_loss)
    if analysis:
        regime_sig=analysis['regime_sig']; regime_label=analysis['regime_label']
        regime_data=analysis['regime_data']; regime_prob=analysis['regime_prob']
        pro_detector=analysis['pro_detector']; trend_diff=analysis['trend_diff']
        vol_state=analysis['vol_state']; curr_vol=analysis['curr_vol']
        jump_detected=analysis['jump_detected']; sentiment_score=analysis['sentiment_score']
        res_sum=analysis['garch_res']
        pb.progress(100); pb.empty()
    else:
        st.sidebar.error("Decision Engine failed.")
        regime_sig="N/A"; regime_label="N/A"; regime_data={'label':'Error','confidence':0.0}
        regime_prob=0.0; pro_detector=None; trend_diff=0.0
        vol_state="UNKNOWN"; curr_vol=0.0; jump_detected=False; sentiment_score=0; res_sum=None
        pb.empty()
else:
    regime_sig="N/A"; regime_label="N/A"; regime_data={'label':'N/A','confidence':0.0}
    regime_prob=0.0; pro_detector=None; trend_diff=0.0
    vol_state="UNKNOWN"; curr_vol=0.0; jump_detected=False; sentiment_score=0; res_sum=None
 
 
# Trade Intelligence module embedded inline above
 
# ==========================================
# TAB 0: DECISION SUMMARY
# ==========================================
with tab0:
    if df_main is None:
        st.info("Welcome. Enter a ticker in the sidebar to begin analysis.")
    else:
        st.write("### 🧠 Executive Decision Dashboard")
    st.markdown(f"**Signal for {TICKER}** | `{data_interval}` | {'🔴 Live' if live_mode else '📅 Historical'}")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Regime",regime_label,f"{regime_data['confidence']:.1%} Conf")
        if "BULL" in regime_label: st.success(f"Action: {regime_sig}")
        elif "BEAR" in regime_label: st.error(f"Action: {regime_sig}")
        else: st.info("Transition — wait for confirmation.")
    with c2:
        st.metric("Kalman Trend",f"{trend_diff:+.2%}","vs Trend Line")
        if trend_diff>0.02: st.success("Above support.")
        elif trend_diff<-0.02: st.warning("Trend breakdown.")
        else: st.info("Consolidating.")
    with c3:
        st.metric("Volatility",vol_state,f"{curr_vol:.2f}% daily")
        if vol_state=="HIGH": st.warning("High Vol: Reduce size.")
        elif vol_state=="LOW": st.success("Low Vol: Favorable.")
        else: st.info("Normal risk.")
    st.divider()
    m1,m2=st.columns([1,2])
    with m1:
        st.write("#### Master Quant Score")
        if sentiment_score>=2: st.header(f"🟢 BULLISH ({sentiment_score})")
        elif sentiment_score<=-2: st.header(f"🔴 BEARISH ({sentiment_score})")
        else: st.header(f"🟡 NEUTRAL ({sentiment_score})")
    with m2:
        st.write("#### Risk Alerts")
        if jump_detected: st.error("🚨 FAT TAIL: Jumps detected — use stochastic models.")
        else: st.success("✅ Smooth dynamics — Gaussian models stable.")
        if vol_state=="HIGH": st.warning("⚠️ VOL CLUSTERING: Shocks likely to persist.")
        st.info(f"Recommendation: {regime_sig}. Target exposure: {min(1.0,0.5+0.1*sentiment_score):.0%} risk parity weight.")
    st.caption("Visit respective tabs for detailed model output.")
 
# ==========================================
# TAB 1: VOLATILITY (GARCH)
# ==========================================
with tab1:
    if df_main is None:
        st.warning("Load a ticker to view Volatility models."); st.stop()
    st.write("### 📉 Advanced Volatility Analysis (GARCH/EGARCH)")
    if res_sum is not None:
        lv = res_sum.conditional_volatility.iloc[-1]
        if vol_state=="HIGH": st.error(f"MODEL VERDICT: Volatility HIGH ({lv:.2f}%). Defensive sizing.")
        else: st.success(f"MODEL VERDICT: Volatility {vol_state} ({lv:.2f}%). Stable environment.")
    if not ARCH_AVAILABLE:
        st.error("Install arch: pip install arch"); st.stop()
    rp = df_main['Returns']*100
    with st.expander("⚙️ Config",expanded=True):
        g1,g2,g3=st.columns(3)
        with g1: vmt=st.selectbox("Model",["GARCH","GJR-GARCH","EGARCH"])
        with g2: dt=st.selectbox("Distribution",["Normal","Student's t","Skewed Student's t"])
        with g3: vl=st.slider("Lag (p,q)",1,3,1)
    vm={"GARCH":"Garch","GJR-GARCH":"Garch","EGARCH":"EGarch"}
    dm={"Normal":"Normal","Student's t":"t","Skewed Student's t":"skewt"}
    op=1 if vmt=="GJR-GARCH" else 0
    try:
        am=arch_model(rp,vol=vm[vmt],p=vl,o=op,q=vl,dist=dm[dt]); res=am.fit(disp='off')
        r1,r2=st.columns([2,1])
        with r1:
            fig_v=go.Figure()
            fig_v.add_trace(go.Scatter(x=rp.index,y=res.conditional_volatility,mode='lines',
                                        line=dict(color='#00f2ff',width=1.5),name=f'{vmt} Vol'))
            fig_v.update_layout(title=f"{vmt} Conditional Volatility",template="plotly_dark",
                                 hovermode="x unified",height=380)
            st.plotly_chart(fig_v,use_container_width=True)
            if st.session_state.report_gen: st.session_state.report_gen.add_plot("GARCH Vol",fig_v)
        with r2:
            pdf=pd.DataFrame({"Value":res.params.values,"t-stat":res.tvalues.values},index=res.params.index)
            st.dataframe(pdf.style.format("{:.4f}"))
            if 'alpha[1]' in res.params and 'beta[1]' in res.params:
                pers=res.params['alpha[1]']+res.params['beta[1]']
                st.metric("Persistence",f"{pers:.4f}")
                if pers<1: st.metric("Half-Life",f"{np.log(0.5)/np.log(pers):.1f}d")
            st.metric("AIC",f"{res.aic:.1f}"); st.metric("BIC",f"{res.bic:.1f}")
        td,tc,tr=st.tabs(["Diagnostics","Forecast","Risk"])
        with td:
            sr2=res.std_resid
            d1,d2=st.columns(2)
            with d1:
                fsr=go.Figure(); fsr.add_trace(go.Scatter(x=rp.index,y=sr2,mode='lines',line=dict(color='gray')))
                fsr.add_hline(y=0,line_dash="dash"); fsr.update_layout(title="Std Residuals",template="plotly_dark",height=300)
                st.plotly_chart(fsr,use_container_width=True)
            with d2:
                qq=stats.probplot(sr2,dist="norm"); th,ob=qq[0]; sl,ic,_=qq[1]
                fqq=go.Figure()
                fqq.add_trace(go.Scatter(x=th,y=ob,mode='markers',marker=dict(color='#00f2ff'),name='Data'))
                fqq.add_trace(go.Scatter(x=th,y=sl*th+ic,mode='lines',line=dict(color='red'),name='Fit'))
                fqq.update_layout(title="Q-Q Plot",template="plotly_dark",height=300)
                st.plotly_chart(fqq,use_container_width=True)
            lb=acorr_ljungbox(sr2,lags=[10],return_df=True); arch_t=het_arch(sr2)
            st.table(pd.DataFrame({"p-value":[lb['lb_pvalue'].iloc[0],arch_t[1]],
                                    "Result":["OK" if lb['lb_pvalue'].iloc[0]>0.05 else "Autocorr!",
                                              "OK" if arch_t[1]>0.05 else "ARCH remains!"]},
                                   index=["Ljung-Box","ARCH-LM"]))
        with tc:
            fh=st.slider("Horizon (days)",1,63,21)
            try: fc=res.forecast(horizon=fh,reindex=False)
            except: fc=res.forecast(horizon=fh,method='simulation',simulations=500,reindex=False)
            vf=np.sqrt(fc.variance.iloc[-1])
            fd=[rp.index[-1]+timedelta(days=i) for i in range(1,fh+1)]
            ff=go.Figure()
            ff.add_trace(go.Scatter(x=rp.index[-60:],y=res.conditional_volatility[-60:],
                                     mode='lines',line=dict(color='gray'),name='Historical'))
            ff.add_trace(go.Scatter(x=fd,y=vf,mode='lines+markers',
                                     line=dict(color='red',dash='dash'),name='Forecast'))
            ff.update_layout(title="Volatility Forecast",template="plotly_dark",height=380,hovermode="x unified")
            st.plotly_chart(ff,use_container_width=True)
        with tr:
            rv1,rv2=st.columns(2)
            with rv1:
                pv=st.number_input("Portfolio Value",1000,10000000,100000)
                cl=st.selectbox("Confidence",[0.95,0.99])
                nv=np.sqrt(fc.variance.iloc[-1].iloc[0])/100
                q=stats.norm.ppf(1-cl)
                var_v=-q*nv*pv
                st.metric(f"1-Day VaR ({cl:.0%})",f"{CURRENCY}{var_v:,.0f}")
            with rv2:
                tv=st.slider("Target Annual Vol (%)",5,50,15)/100
                cav=nv*np.sqrt(252)
                lf=tv/cav
                st.metric("Vol-Targeted Exposure",f"{CURRENCY}{pv*lf:,.0f}",f"{lf:.2f}x leverage")
                if lf>1: st.warning("Requires margin.")
                else: st.success("Cash position — no leverage needed.")
    except Exception as e:
        st.error(f"Model failed: {e}")
 
# ==========================================
# TAB 2: REGIME SWITCHING
# ==========================================
with tab2:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🔀 Markov Regime Switching")
    if "LONG" in regime_sig: st.success(f"VERDICT: {regime_sig} — {regime_data['label']}")
    elif "SHORT" in regime_sig: st.error(f"VERDICT: {regime_sig} — Risk elevated.")
    else: st.info(f"VERDICT: {regime_sig} — Await confirmation.")
    rfc1,rfc2,rfc3=st.columns(3)
    with rfc1: rfreq=st.selectbox("Frequency",["Daily","Weekly"],index=1)
    with rfc2: rly=st.slider("Lookback (Years)",1,10,2)
    with rfc3: rnr=st.slider("Regimes",2,4,2)
    rstab=st.slider("Pre-Smoothing",0,10,4)
    rcthr=st.slider("Conviction Threshold",0.5,0.95,0.7,step=0.05)
    rcsw1,rcsw2=st.columns(2)
    with rcsw1: sw_t=st.checkbox("Switching Mean",value=True)
    with rcsw2: sw_v=st.checkbox("Switching Volatility",value=True)
    rstart=datetime.now()-timedelta(days=rly*365)
    dfr=load_data(TICKER,rstart,end_date)
    if dfr is None: st.error("No data."); st.stop()
    ret_r=dfr['Returns'].resample('W').sum() if rfreq=="Weekly" else dfr['Returns']
    if rstab>0: ret_r=ret_r.ewm(span=rstab,adjust=False).mean()
    md=pd.Series(ret_r.dropna().values.flatten().astype(float)*100,index=ret_r.dropna().index)
    if len(md)<10: st.error("Insufficient data."); st.stop()
    with st.spinner("Fitting Markov model..."):
        res_m=fit_regime_model(md,rnr,sw_v,sw_t)
    if res_m is None: st.error("Model failed."); st.stop()
    if not res_m.mle_retvals['converged']: st.error("Did not converge."); st.stop()
    tm=np.atleast_2d(np.squeeze(res_m.regime_transition))
    rs_list=[]
    for i in range(rnr):
        mv=res_m.params[f'const[{i}]'] if f'const[{i}]' in res_m.params else res_m.params.get('const',0.)
        sv=np.sqrt(res_m.params[f'sigma2[{i}]'] if f'sigma2[{i}]' in res_m.params else res_m.params.get('sigma2',1.))
        rs_list.append({'regime':i,'mean':float(mv),'vol':float(sv),'persistence':float(tm[i,i])})
    rs_list=sorted(rs_list,key=lambda x:x['mean'],reverse=True)
    lbls=['🟢 Bull','🟡 Normal','🔴 Bear','⚫ Crisis']
    rcols=st.columns(rnr)
    for idx,(col,rg) in enumerate(zip(rcols,rs_list)):
        with col:
            st.markdown(f"**{lbls[idx]}**")
            st.metric("Mean Ret",f"{rg['mean']:.2f}%")
            st.metric("Volatility",f"{rg['vol']:.2f}%")
            st.metric("Persistence",f"{rg['persistence']:.1%}")
    lp=res_m.filtered_marginal_probabilities.iloc[-1]
    cr=np.argmax(lp); cp=lp.iloc[cr]
    rl2=lbls[[r['regime'] for r in rs_list].index(cr)]
    is_conv=cp>=rcthr
    st.divider()
    dc1,dc2,dc3=st.columns(3)
    with dc1:
        if is_conv: st.subheader(rl2); st.success(f"High Conviction ({cp:.1%})")
        else: st.subheader("⚪ Mixed"); st.warning(f"Low Conviction ({cp:.1%})")
    with dc2:
        sp2=sorted(lp.values,reverse=True)
        sprd=sp2[0]-(sp2[1] if len(sp2)>1 else 0)
        st.metric("Probability Spread",f"{sprd:.1%}"); st.progress(float(min(1,max(0,sprd))))
    with dc3:
        avp=np.mean([r['persistence'] for r in rs_list])
        st.metric("Avg Persistence",f"{avp:.1%}")
    st.divider()
    smth=st.checkbox("Smooth Probabilities",value=True)
    import matplotlib.colors as mcolors
    fig_r=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.05,
                         subplot_titles=("Returns","Probabilities","Expected Return"))
    fig_r.add_trace(go.Scatter(x=md.index,y=md,mode='lines',line=dict(color='gray',width=1)),row=1,col=1)
    for i,rg in enumerate(rs_list):
        ci=1-(i/(rnr-1)) if rnr>1 else 1.
        hc=mcolors.to_hex(plt.cm.RdYlGn(ci))
        prb=res_m.filtered_marginal_probabilities.iloc[:,rg['regime']]
        pp=prb.rolling(4,min_periods=1).mean() if smth else prb
        fig_r.add_trace(go.Scatter(x=md.index,y=pp,mode='lines',line=dict(color=hc,width=1.5),
                                    fill='tozeroy',name=lbls[i]),row=2,col=1)
    er=pd.Series(0.,index=md.index)
    for i in range(rnr):
        m2=res_m.params[f'const[{i}]'] if f'const[{i}]' in res_m.params else res_m.params.get('const',0.)
        er+=res_m.filtered_marginal_probabilities.iloc[:,i]*float(m2)
    fig_r.add_trace(go.Scatter(x=md.index,y=er,mode='lines',line=dict(color='#00f2ff',width=2),name="E[Ret]"),row=3,col=1)
    ep=er.copy(); ep[ep<0]=0
    en=er.copy(); en[en>0]=0
    fig_r.add_trace(go.Scatter(x=md.index,y=ep,line=dict(width=0),fill='tozeroy',fillcolor='green',opacity=0.2,showlegend=False),row=3,col=1)
    fig_r.add_trace(go.Scatter(x=md.index,y=en,line=dict(width=0),fill='tozeroy',fillcolor='red',opacity=0.2,showlegend=False),row=3,col=1)
    fig_r.update_layout(height=750,hovermode="x unified",template="plotly_dark",title="Markov Regime Analysis")
    st.plotly_chart(fig_r,use_container_width=True)
    if st.session_state.report_gen: st.session_state.report_gen.add_plot("Regime",fig_r)
    with st.expander("Technical Parameters"):
        st.dataframe(pd.DataFrame({"Value":res_m.params.values.astype(float),
                                    "SE":res_m.bse.values.astype(float),
                                    "p":res_m.pvalues.values.astype(float)},
                                   index=res_m.params.index).style.format("{:.4f}"))
        st.caption(f"AIC: {res_m.aic:.1f} | BIC: {res_m.bic:.1f}")
 
# ==========================================
# TAB 3: STOCHASTIC MODELS
# ==========================================
with tab3:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🎲 Stochastic Simulations (Heston / Merton)")
    sc1,sc2=st.columns(2)
    with sc1: sim_t=st.radio("Model",["Merton Jump Diffusion","Heston Stochastic Volatility"])
    with sc2:
        dr=st.radio("Drift",["Risk-Neutral (Rf)","Historical Mean","Custom"])
        if dr=="Risk-Neutral (Rf)": mu_d=rf_rate
        elif dr=="Historical Mean": mu_d=df_main['Log_Returns'].mean()*252
        else: mu_d=st.number_input("Annual Return (%)",-50.,100.,10.)/100
    cp=df_main['Close'].iloc[-1]
    fd2=[df_main.index[-1]+timedelta(days=i) for i in range(253)]
    seed=st.number_input("Random Seed",1,9999,42); np.random.seed(int(seed))
    if sim_t=="Merton Jump Diffusion":
        m1,m2=st.columns([1,3])
        with m1:
            lam=st.slider("Jump Intensity",0.1,10.,1.0)
            mj=st.slider("Jump Mean",-0.5,0.5,-0.1)
            sj=st.slider("Jump Std",0.01,0.5,0.1)
            sv=st.slider("Diffusive Vol",0.05,1.,0.2)
        with m2:
            paths=merton_jump_diffusion(cp,1.,mu_d,sv,lam,mj,sj,252,50)
            mp=paths.mean(axis=1); p5=np.percentile(paths,5,axis=1); p95=np.percentile(paths,95,axis=1)
            st.metric("Projected Mean",f"{CURRENCY}{mp[-1]:,.2f}")
            fig_mj=go.Figure()
            fig_mj.add_trace(go.Scatter(x=fd2+fd2[::-1],y=np.concatenate([p95,p5[::-1]]),
                                         fill='toself',fillcolor='rgba(100,100,255,0.15)',
                                         line=dict(color='rgba(0,0,0,0)'),name='90% CI'))
            for i in range(min(15,paths.shape[1])):
                fig_mj.add_trace(go.Scatter(x=fd2,y=paths[:,i],mode='lines',
                                             line=dict(color='rgba(100,100,255,0.05)',width=1),showlegend=False))
            fig_mj.add_trace(go.Scatter(x=fd2,y=mp,mode='lines',line=dict(color='orange',width=3),name='Mean'))
            fig_mj.update_layout(title="Merton Jump Diffusion Paths",template="plotly_dark",
                                  height=400,hovermode="x unified")
            st.plotly_chart(fig_mj,use_container_width=True)
    else:
        if 'h_kappa' not in st.session_state: st.session_state.h_kappa=2.0
        if 'h_theta' not in st.session_state: st.session_state.h_theta=0.04
        if 'h_xi' not in st.session_state: st.session_state.h_xi=0.3
        if 'h_rho' not in st.session_state: st.session_state.h_rho=-0.7
        if 'h_v0' not in st.session_state: st.session_state.h_v0=0.04
        hc1,hc2=st.columns([1,3])
        with hc1:
            if st.button("Calibrate from History"):
                with st.spinner("Calibrating..."):
                    try:
                        cr2=Calibrator.calibrate_heston(df_main['Log_Returns'])
                        for k,v in [('h_kappa',cr2['kappa']),('h_theta',cr2['theta']),
                                    ('h_xi',cr2['xi']),('h_rho',cr2['rho']),('h_v0',cr2['v0'])]:
                            st.session_state[k]=float(v)
                        st.success("Done!")
                    except Exception as e: st.error(str(e))
            hk=st.number_input("Kappa",0.01,1000.,key='h_kappa',format="%.4f")
            ht=st.number_input("Theta",0.,5.,key='h_theta',format="%.6f")
            hx=st.number_input("Xi",0.01,100.,key='h_xi',format="%.4f")
            hr=st.slider("Rho",-0.99,0.99,key='h_rho')
            hv=st.number_input("v0",0.,5.,key='h_v0',format="%.6f")
        with hc2:
            sp,sv2=simulate_heston(cp,1.,mu_d,hk,ht,hx,hr,hv,252,50)
            mp2=sp.mean(axis=1); p52=np.percentile(sp,5,axis=1); p952=np.percentile(sp,95,axis=1)
            st.metric("Projected Mean",f"{CURRENCY}{mp2[-1]:,.2f}")
            fh2=go.Figure()
            fh2.add_trace(go.Scatter(x=fd2+fd2[::-1],y=np.concatenate([p952,p52[::-1]]),
                                      fill='toself',fillcolor='rgba(100,100,255,0.15)',
                                      line=dict(color='rgba(0,0,0,0)'),name='90% CI'))
            for i in range(min(15,sp.shape[1])):
                fh2.add_trace(go.Scatter(x=fd2,y=sp[:,i],mode='lines',
                                          line=dict(color='rgba(100,100,255,0.05)',width=1),showlegend=False))
            fh2.add_trace(go.Scatter(x=fd2,y=mp2,mode='lines',line=dict(color='orange',width=3),name='Mean'))
            fh2.update_layout(title="Heston Price Paths",template="plotly_dark",height=380,hovermode="x unified")
            st.plotly_chart(fh2,use_container_width=True)
 
# ==========================================
# TAB 4: KALMAN FILTER
# ==========================================
with tab4:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 📡 Kalman Filter Analysis")
    if trend_diff>0.03: st.success(f"VERDICT: Price {trend_diff:.1%} ABOVE Kalman trend — uptrend intact.")
    elif trend_diff<-0.03: st.error(f"VERDICT: Price {abs(trend_diff):.1%} BELOW trend — breakdown.")
    else: st.info(f"VERDICT: Within {abs(trend_diff):.1%} of trend — consolidation.")
    km=st.radio("Mode",["Single Asset (Trend)","Pairs Trading"])
    if km=="Single Asset (Trend)":
        kc1,kc2,kc3=st.columns(3)
        with kc1: pn=st.select_slider("Process Noise",[1e-5,1e-4,1e-3,1e-2],value=1e-4)
        with kc2: mn=st.select_slider("Measurement Noise",[1e-3,1e-2,1e-1,1.],value=1e-2)
        with kc3: mm=st.radio("Type",["Smoothed","Standard","Compare"])
        kft=KalmanFilterTrend(pn,mn); pv=df_main['Close'].values
        if mm=="Standard": et,_=kft.filter(pv); lbl_k="Standard"; col_k="blue"
        elif mm=="Smoothed": et,_=kft.smooth(pv); lbl_k="Smoothed"; col_k="purple"
        else:
            ets,_=kft.smooth(pv); etf,_=kft.filter(pv)
        fk=go.Figure()
        fk.add_trace(go.Scatter(x=df_main.index,y=pv,mode='lines',line=dict(color='gray'),opacity=0.5,name='Price'))
        if mm=="Compare":
            fk.add_trace(go.Scatter(x=df_main.index,y=etf,mode='lines',line=dict(color='blue',dash='dash'),name='Standard'))
            fk.add_trace(go.Scatter(x=df_main.index,y=ets,mode='lines',line=dict(color='purple',width=2),name='Smoothed'))
            ct=ets[-1]
        else:
            fk.add_trace(go.Scatter(x=df_main.index,y=et,mode='lines',line=dict(color=col_k,width=2),name=lbl_k))
            ct=et[-1]
        fk.update_layout(title=f"Kalman Trend: {TICKER}",template="plotly_dark",height=450,hovermode="x unified")
        st.plotly_chart(fk,use_container_width=True)
        lp2=float(pv[-1]); dp=(lp2-ct)/ct*100
        kk1,kk2,kk3=st.columns(3)
        kk1.metric("Price",f"{CURRENCY}{lp2:.2f}")
        kk2.metric("Trend",f"{CURRENCY}{ct:.2f}")
        kk3.metric("Deviation",f"{dp:.2f}%",delta_color="inverse")
        if dp>5: st.warning("Overbought vs trend.")
        elif dp<-5: st.success("Oversold vs trend — potential bounce.")
        else: st.info("Trading at trend.")
    else:
        if not PAIR_TICKER:
            st.warning("Enter a Pair Ticker in the sidebar.")
        else:
            dfp=load_data(PAIR_TICKER,start_date,end_date)
            if dfp is not None:
                ci=df_main.index.intersection(dfp.index)
                y=df_main.loc[ci,'Close'].values; x=dfp.loc[ci,'Close'].values
                kfr=KalmanFilterReg(); sm,_=kfr.run_filter(y,x)
                beta_k=sm[:,1]
                spread=y-beta_k*x
                z_k=(spread-spread.mean())/spread.std()
                fkp=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                   subplot_titles=("Dynamic Beta","Spread Z-Score"))
                fkp.add_trace(go.Scatter(x=ci,y=beta_k,mode='lines',line=dict(color='#00f2ff'),name='Beta'),row=1,col=1)
                fkp.add_trace(go.Scatter(x=ci,y=z_k,mode='lines',line=dict(color='purple'),name='Z-Score'),row=2,col=1)
                fkp.add_hline(y=2,line_dash="dash",line_color="red",row=2,col=1)
                fkp.add_hline(y=-2,line_dash="dash",line_color="green",row=2,col=1)
                fkp.update_layout(height=500,template="plotly_dark",hovermode="x unified")
                st.plotly_chart(fkp,use_container_width=True)
                st.write(f"Current Beta: **{beta_k[-1]:.4f}** | Z-Score: **{z_k.iloc[-1]:.2f}**")
            else:
                st.error(f"Could not load {PAIR_TICKER}")
 
# ==========================================
# TAB 5: MACRO
# ==========================================
with tab5:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🌍 Macro Factor Sensitivity")
    macro_map={"Crude Oil":"CL=F","Gold":"GC=F","10Y Yield":"^TNX",
               "USD Index":"DX-Y.NYB","S&P 500":"^GSPC"}
    mdata={n:load_data(s,start_date,end_date) for n,s in macro_map.items()}
    mdata[TICKER]=df_main
    mdf=pd.DataFrame({k:v['Returns'] for k,v in mdata.items() if v is not None}).dropna()
    if not mdf.empty:
        cm=mdf.corr()
        fhm=go.Figure(go.Heatmap(z=cm.values,x=cm.columns,y=cm.columns,
                                   colorscale='RdBu',zmin=-1,zmax=1,
                                   text=np.round(cm.values,2),texttemplate="%{text}"))
        fhm.update_layout(title="Asset Correlations",template="plotly_dark",height=500)
        st.plotly_chart(fhm,use_container_width=True)
        if st.session_state.report_gen: st.session_state.report_gen.add_plot("Macro Corr",fhm)
        if TICKER in cm.columns:
            oc=cm.loc[TICKER,'Crude Oil'] if 'Crude Oil' in cm.columns else 0
            rc=cm.loc[TICKER,'10Y Yield'] if '10Y Yield' in cm.columns else 0
            if abs(oc)>0.3: st.success(f"Energy correlation: {oc:.2f}")
            if abs(rc)>0.3: st.info(f"Rate sensitivity: {rc:.2f}")
 
# ==========================================
# TAB 6: STRUCTURAL
# ==========================================
with tab6:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🏗️ Structural Decomposition")
    per=st.selectbox("Period",[5,21,63,252],index=1)
    if len(df_main)>per*2:
        dc=seasonal_decompose(df_main['Close'],model='multiplicative',period=per)
        fdc=make_subplots(rows=3,cols=1,shared_xaxes=True,subplot_titles=('Trend','Seasonal','Residuals'))
        fdc.add_trace(go.Scatter(x=dc.trend.index,y=dc.trend,name='Trend'),row=1,col=1)
        fdc.add_trace(go.Scatter(x=dc.seasonal.index,y=dc.seasonal,name='Seasonal'),row=2,col=1)
        fdc.add_trace(go.Scatter(x=dc.resid.index,y=dc.resid,name='Residuals'),row=3,col=1)
        fdc.update_layout(height=700,template="plotly_dark",hovermode="x unified")
        st.plotly_chart(fdc,use_container_width=True)
    else:
        st.warning("Insufficient data for selected period.")
 
# ==========================================
# TAB 7: BACKTEST (full with all strategies + fixes)
# ==========================================
with tab7:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🛠️ Strategy Backtest (Vol-Targeted + Fixed Hurst)")
 
    strat=st.radio("Strategy",[
        "Regime Switching","Kalman Trend","EMA/SMA Cross","MAD Trend Modes",
        "Dual MA Cross","Ehlers SuperSmoother","Ehlers Decycler",
        "Mean Reversion Z-Score","Relative Strength","VIX Proxy (GARCH-Fixed)",
        "Hurst Exponent (Fixed)"],horizontal=True)
 
    bts=st.date_input("BT Start",datetime.now()-timedelta(days=365),key="bt_s")
    bte=st.date_input("BT End",datetime.now(),key="bt_e")
    dfbt=load_data(TICKER,bts,bte,'1d') if not live_mode else df_main
    if dfbt is None or dfbt.empty:
        st.error("No backtest data."); st.stop()
    pb=dfbt['Close']; rb=dfbt['Returns']
    sigs_bt=None; strat_prices=pb
 
    if strat=="Regime Switching":
        b1,b2,b3=st.columns(3)
        with b1: bfq=st.selectbox("Freq",["Weekly","Daily"],key="bfq")
        with b2: bnr=st.slider("Regimes",2,4,2,key="bnr")
        with b3: bst=st.slider("Smoothing",0,10,4,key="bst")
        bc1,bc2=st.columns(2)
        with bc1: bsv=st.checkbox("Switching Vol",True,key="bsv")
        with bc2: bst2=st.checkbox("Switching Mean",True,key="bst2")
        if bfq=="Weekly":
            pb_r=pb.resample('W').last().dropna(); rb_r=pb_r.pct_change().dropna()
        else: pb_r=pb; rb_r=rb
        if bst>0: md_bt=rb_r.ewm(span=bst,adjust=False).mean().dropna()*100
        else: md_bt=rb_r.dropna()*100
        md_bt=pd.Series(md_bt.values.flatten().astype(float),index=md_bt.index)
        strat_prices=pb_r
        if len(md_bt)<10: st.error("Insufficient data."); st.stop()
        with st.spinner("Fitting..."):
            res_bt=fit_regime_model(md_bt,bnr,bsv,bst2)
        if res_bt:
            pbt=res_bt.filtered_marginal_probabilities
            rm=[]; 
            for i in range(bnr):
                m=res_bt.params[f'const[{i}]'] if f'const[{i}]' in res_bt.params else res_bt.params.get('const',0.)
                rm.append((i,m))
            bi=sorted(rm,key=lambda x:x[1],reverse=True)[0][0]
            er_bt=pd.Series(0.,index=md_bt.index)
            for i in range(bnr):
                m=res_bt.params[f'const[{i}]'] if f'const[{i}]' in res_bt.params else res_bt.params.get('const',0.)
                er_bt+=pbt.iloc[:,i]*float(m)
            raw_sig=(er_bt>0).astype(int)
            # FIX: vol-targeted sizing
            vt_pct=st.slider("Vol Target (%)",5,30,15,key="vt_reg")/100
            try:
                rs2=(rb_r*100).replace([np.inf,-np.inf],np.nan).dropna()
                am_bt=arch_model(rs2,vol='Garch',p=1,q=1,dist='Normal'); gr=am_bt.fit(disp='off')
                sigs_bt=vol_targeted_signal(raw_sig,gr,vt_pct)
            except: sigs_bt=raw_sig.astype(float)
        else: st.error("Regime model failed.")
 
    elif strat=="Kalman Trend":
        kfn=st.select_slider("Sensitivity",[1e-5,1e-4,1e-3],value=1e-4)
        kcd=st.slider("Confirmation bars",1,5,1)
        kf2=KalmanFilterTrend(kfn,1e-2); te,_=kf2.filter(pb.values)
        ts=pd.Series(te,index=pb.index)
        pos=0; da=0; db=0; sl=[]
        for price,trend in zip(pb,ts):
            if price>trend: da+=1; db=0
            else: db+=1; da=0
            if pos==0 and da>=kcd: pos=1
            elif pos==1 and db>=kcd: pos=0
            sl.append(pos)
        sigs_bt=pd.Series(sl,index=pb.index).astype(float)
 
    elif strat=="EMA/SMA Cross":
        e1,e2=st.columns(2)
        with e1: sh=st.slider("Short EMA",5,50,20)
        with e2: lo=st.slider("Long SMA",20,200,60)
        sigs_bt=(pb.ewm(span=sh,adjust=False).mean()>=pb.rolling(lo).mean()).astype(float)
 
    elif strat=="MAD Trend Modes":
        mp={}
        mp['signal_mode']=st.selectbox("Mode",["Bollinger Bands","For Loop","Combined Signal"])
        mp['bb_ma_type']=st.selectbox("MA Type",["EMA","SMA","WMA","HMA","RMA","ALMA","LSMA"])
        mp['bb_len']=st.number_input("Length",5,100,25)
        sigs_bt=MADTrendModes.get_signals(dfbt,mp).astype(float)
 
    elif strat=="Dual MA Cross":
        ma_opts=["SMA","EMA","WMA","HMA","RMA","ALMA","LSMA"]
        dc1b,dc2b=st.columns(2)
        with dc1b: fmt=st.selectbox("Fast",ma_opts,index=1); fl=st.number_input("Fast Len",1,250,20)
        with dc2b: smt=st.selectbox("Slow",ma_opts); sl2=st.number_input("Slow Len",1,250,50)
        fma=MADTrendModes.ma_switch(pb,fl,fmt); sma=MADTrendModes.ma_switch(pb,sl2,smt)
        def stateful2(lc,sc,idx):
            sig=pd.Series(np.nan,index=idx); sig.loc[lc]=1; sig.loc[sc]=0
            return sig.ffill().fillna(0)
        sigs_bt=stateful2((fma>sma)&(fma.shift(1)<=sma.shift(1)),
                           (fma<sma)&(fma.shift(1)>=sma.shift(1)),pb.index).astype(float)
 
    elif strat=="Ehlers SuperSmoother":
        sp2=st.slider("Period",5,252,15)
        ss=EhlersFilters.super_smoother(pb,sp2)
        sigs_bt=(pb>ss).astype(float)
 
    elif strat=="Ehlers Decycler":
        dp2=st.slider("Period",20,252,60)
        dc_s=EhlersFilters.simple_decycler(pb,dp2)
        sigs_bt=(pb>dc_s).astype(float)
 
    elif strat=="Mean Reversion Z-Score":
        zc1,zc2,zc3=st.columns(3)
        with zc1: zlb=st.slider("Lookback",5,252,20)
        with zc2: zen=st.number_input("Entry Z",0.1,5.,2.,step=0.1)
        with zc3: zex=st.number_input("Exit Z",-2.,2.,0.,step=0.1)
        zma=pb.rolling(zlb).mean(); zstd=pb.rolling(zlb).std()
        zz=(pb-zma)/(zstd+1e-9)
        def stateful3(lc,ec,idx):
            sig=pd.Series(np.nan,index=idx); sig.loc[lc]=1; sig.loc[ec]=0
            return sig.ffill().fillna(0)
        sigs_bt=stateful3(zz<-zen,zz>-zex,pb.index).astype(float)
 
    elif strat=="Relative Strength":
        bt_bench=st.text_input("Benchmark","SPY",key="rs_bench")
        rs_len=st.slider("RS MA",5,200,50)
        bdf2=load_data(bt_bench,bts,bte,'1d')
        if bdf2 is not None:
            ci2=pb.index.intersection(bdf2['Close'].index)
            rs_r=pb.loc[ci2]/bdf2['Close'].loc[ci2]
            rs_ma=rs_r.rolling(rs_len).mean()
            raw_rs=(rs_r>rs_ma).astype(int)
            # FIX: GARCH vol filter instead of VIX
            sigs_bt=garch_vol_filter_signal(pb,rb,raw_rs,15.0)
        else: st.error("Cannot load benchmark.")
 
    elif strat=="VIX Proxy (GARCH-Fixed)":
        st.info("""
        **Fix applied**: The original strategy used ^VIX as a proxy for individual stock risk.
        This version uses the **stock's own GARCH conditional volatility** to size positions.
        When the stock's vol is HIGH → position reduced proportionally (not binary exit).
        When vol is LOW → full position. This eliminates the VIX mismatch problem.
        """)
        gvt=st.slider("Vol Target (%)",5,30,15,key="gvt_fix")/100
        base_sig=pd.Series(1,index=pb.index)  # Always-long base
        sigs_bt=garch_vol_filter_signal(pb,rb,base_sig,gvt*100)
 
    elif strat=="Hurst Exponent (Fixed)":
        st.info("""
        **Fixes applied vs original**:
        1. Dead zone 0.45–0.55 → stay in cash (no whipsaw between strategies)
        2. 5-bar confirmation → only switch regime if H is consistently above/below threshold
        3. H>0.55 confirmed → momentum (EMA cross), H<0.45 confirmed → mean reversion (Bollinger)
        """)
        hc1b,hc2b,hc3b,hc4b=st.columns(4)
        with hc1b: hw=st.number_input("Window",20,500,100,step=10)
        with hc2b: htt=st.number_input("Trend H>",0.4,0.7,0.55,step=0.01)
        with hc3b: hmt=st.number_input("MR H<",0.3,0.6,0.45,step=0.01)
        with hc4b: hcb=st.slider("Confirm bars",1,10,5)
        with st.spinner("Computing Hurst..."):
            try:
                sigs_bt,hs,ist,ismr=hurst_confirmed_signal(pb,int(hw),htt,hmt,hcb)
                sigs_bt=sigs_bt.astype(float)
                fhp=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                   subplot_titles=("Price","Hurst (with dead zone)"))
                fhp.add_trace(go.Scatter(x=pb.index,y=pb,mode='lines',line=dict(color='gray'),name='Price'),row=1,col=1)
                fhp.add_trace(go.Scatter(x=hs.index,y=hs,mode='lines',line=dict(color='cyan'),name='H'),row=2,col=1)
                fhp.add_hline(y=0.5,line_dash="dash",line_color="gray",row=2,col=1)
                fhp.add_hline(y=htt,line_dash="dash",line_color="green",row=2,col=1,annotation_text=f"Trend>{htt}")
                fhp.add_hline(y=hmt,line_dash="dash",line_color="purple",row=2,col=1,annotation_text=f"MR<{hmt}")
                highlight_plotly_zones(fhp,sigs_bt==1,'green',opacity=0.1,row=1,col=1)
                fhp.update_layout(height=500,template="plotly_dark",hovermode="x unified")
                st.plotly_chart(fhp,use_container_width=True)
            except Exception as e:
                st.error(f"Hurst error: {e}"); sigs_bt=None
 
    if sigs_bt is not None:
        btr=BacktestEngine.run_strategy(strat_prices,sigs_bt,initial_cap,trailing_stop,stop_loss)
        last_s=float(sigs_bt.iloc[-1])
        st.divider()
        if last_s>0: st.success(f"SIGNAL: LONG (size={last_s:.0%}) | {sigs_bt.index[-1].date()}")
        else: st.error(f"SIGNAL: CASH | {sigs_bt.index[-1].date()}")
        sm=BacktestEngine.calculate_metrics(btr['returns'],rf_rate)
        m1b,m2b,m3b,m4b=st.columns(4)
        m1b.metric("Total Return",f"{(btr['equity_curve'].iloc[-1]/initial_cap-1)*100:.1f}%")
        m2b.metric("Sharpe",f"{sm.get('Sharpe Ratio',0):.2f}")
        m3b.metric("Max DD",f"{sm.get('Max Drawdown',0)*100:.1f}%")
        m4b.metric("B&H Return",f"{(btr['benchmark_curve'].iloc[-1]/initial_cap-1)*100:.1f}%")
        fbt=go.Figure()
        fbt.add_trace(go.Scatter(x=btr['equity_curve'].index,y=btr['equity_curve'],
                                  mode='lines',line=dict(color='#00f2ff',width=2),name='Strategy'))
        fbt.add_trace(go.Scatter(x=btr['benchmark_curve'].index,y=btr['benchmark_curve'],
                                  mode='lines',line=dict(color='gray',dash='dash'),name='B&H'))
        fbt.update_layout(title="Equity Curve",template="plotly_dark",height=400,hovermode="x unified")
        st.plotly_chart(fbt,use_container_width=True)
        if st.session_state.report_gen: st.session_state.report_gen.add_plot("Backtest",fbt)
        if not btr['trades'].empty:
            td=btr['trades'].copy()
            for dc in ['Entry Date','Exit Date']:
                if dc in td.columns:
                    td[dc]=pd.to_datetime(td[dc]).apply(lambda x:x.date() if pd.notnull(x) else "Open")
            st.dataframe(td.style.format({"Buy Price":"{:.2f}","Sell Price":"{:.2f}","PnL (%)":"{:.2f}%"}),
                         use_container_width=True)
 
# ==========================================
# TAB 8: VOL CLUSTERING
# ==========================================
with tab8:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🌩️ Volatility Clustering & Jump Analysis")
    ra=df_main['Returns'].values
    rv_v=RealizedVolatility.realized_variance(ra); bv_v=RealizedVolatility.bipower_variation(ra)
    jr=RealizedVolatility.jump_component(ra)
    hk=HawkesVolatility().fit(ra); br_v=hk.branching_ratio(); hl_v=hk.half_life()
    vc1,vc2,vc3=st.columns(3)
    vc1.metric("Total RV",f"{np.sqrt(rv_v)*np.sqrt(252):.2%}")
    vc2.metric("Continuous BV",f"{np.sqrt(bv_v)*np.sqrt(252):.2%}")
    vc3.metric("Jump Ratio",f"{jr['jump_ratio']:.1%}")
    if jr['p_value']<0.05: st.error("Significant Jumps Detected")
    else: st.success("No Significant Jumps")
    st.divider()
    hv1,hv2,hv3=st.columns(3)
    hv1.metric("Branching Ratio",f"{br_v:.2f}"); hv2.metric("Half-Life",f"{hl_v:.1f}d")
    hv3.metric("Baseline Intensity",f"{hk.mu:.4f}")
    if br_v>0.9: st.warning("Critical Instability: self-reinforcing vol.")
    elif br_v>0.5: st.info("Moderate clustering.")
    else: st.success("Stable — vol mean-reverts quickly.")
    fvc=make_subplots(rows=2,cols=1,shared_xaxes=True,
                       subplot_titles=("Returns","Squared Returns (Vol Clustering)"))
    fvc.add_trace(go.Scatter(x=df_main.index,y=df_main['Returns'],mode='lines',
                              line=dict(color='gray',width=1)),row=1,col=1)
    sq=df_main['Returns']**2
    fvc.add_trace(go.Scatter(x=df_main.index,y=sq,mode='lines',
                              line=dict(color='orange',width=1)),row=2,col=1)
    fvc.add_hline(y=float(sq.mean()+2*sq.std()),line_dash="dash",line_color="red",row=2,col=1)
    fvc.update_layout(height=500,template="plotly_dark",hovermode="x unified")
    st.plotly_chart(fvc,use_container_width=True)
 
# ==========================================
# TAB 9: ADVANCED REGIME
# ==========================================
with tab9:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 🧠 Pro Regime Detection (Multi-Factor GMM)")
    if pro_detector is None:
        st.info(f"Active Engine: Markov | State: **{regime_label}** ({regime_prob:.1%})")
    else:
        st.metric("Regime",regime_label,f"{regime_prob:.1%}")
        if 'probs' in pro_detector.regimes:
            probs9=pro_detector.regimes['probs']
            lbls9=[pro_detector.state_labels.get(i,f"State {i}") for i in range(probs9.shape[1])]
            fp9=go.Figure()
            for i in range(probs9.shape[1]):
                fp9.add_trace(go.Scatter(x=df_main.index,y=probs9[:,i],mode='lines',
                                          line=dict(width=0),fill='tonexty' if i>0 else 'tozeroy',
                                          stackgroup='one',name=lbls9[i]))
            fp9.update_layout(title="Multi-Factor Regime Probabilities",template="plotly_dark",height=350)
            st.plotly_chart(fp9,use_container_width=True)
 
# ==========================================
# TAB 10: SML & ALPHA
# ==========================================
with tab10:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 📐 SML & Jensen's Alpha (HAC Robust)")
    sb=st.selectbox("Benchmark",["SPY","QQQ","IWM","^NSEI"])
    rw=st.slider("Rolling Window",30,252,90)
    if st.button("Run Alpha Analysis"):
        with st.spinner("Computing..."):
            dbench=load_data(sb,start_date,end_date)
            if dbench is not None:
                sml=SMLAnalyzer(df_main['Returns'],dbench['Returns'],rf_rate)
                rs_sml=sml.calculate_metrics(rw)
                lr=rs_sml.iloc[-1]
                s1,s2,s3,s4=st.columns(4)
                s1.metric("Beta",f"{lr['Beta']:.2f}")
                s2.metric("Alpha",f"{lr['Alpha_Daily']*252:.2%}")
                s3.metric("SML Expected",f"{lr['SML_Exp_Return']:.2%}")
                s4.metric("Mispricing",f"{lr['Mispricing_Spread']*100:.2f}%",
                           delta="Under" if lr['Mispricing_Spread']>0 else "Over")
                fd_sml=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                      subplot_titles=("Beta","Alpha"))
                fd_sml.add_trace(go.Scatter(x=rs_sml.index,y=rs_sml['Beta'],
                                             line=dict(color='purple'),name='Beta'),row=1,col=1)
                fd_sml.add_hline(y=1,line_dash="dash",row=1,col=1)
                aa=rs_sml['Alpha_Daily']*252
                fd_sml.add_trace(go.Scatter(x=rs_sml.index,y=aa,line=dict(color='green'),name='Alpha'),row=2,col=1)
                fd_sml.add_hline(y=0,line_dash="dash",row=2,col=1)
                fd_sml.update_layout(height=500,template="plotly_dark")
                st.plotly_chart(fd_sml,use_container_width=True)
            else: st.error("Cannot load benchmark.")
 
# ==========================================
# TAB 11: IV SCANNER (inline)
# ==========================================
with tab11:
    st.write("### 🔍 Institutional IV Scanner")
    st.markdown("Options-based stock selection: identifies institutional accumulation via IV dynamics.")
 
    @st.cache_data(ttl=300,show_spinner=False)
    def quick_iv(ticker):
        try:
            tk=yf.Ticker(ticker)
            hist=tk.history(period="252d",interval="1d",auto_adjust=True)
            if hist.empty or len(hist)<30: return None
            cp=float(hist['Close'].iloc[-1])
            if cp<=0: return None
            lr=np.log(hist['Close']/hist['Close'].shift(1)).dropna()
            hv30=float(lr.tail(21).std()*np.sqrt(252)*100)
            exps=tk.options
            if not exps: return None
            now=datetime.now()
            atm_ivs=[]; tot_cv=0; tot_pv=0; tot_coi=0; tot_poi=0
            for exp in exps[:3]:
                dte=(datetime.strptime(exp,"%Y-%m-%d")-now).days
                if not 7<=dte<=60: continue
                try:
                    chain=tk.option_chain(exp)
                    calls=chain.calls[(chain.calls['impliedVolatility']>0.01)&(chain.calls['impliedVolatility']<5)].copy()
                    puts=chain.puts[(chain.puts['impliedVolatility']>0.01)&(chain.puts['impliedVolatility']<5)].copy()
                    if calls.empty or puts.empty: continue
                    atm_c=calls[(calls['strike']>=cp*0.97)&(calls['strike']<=cp*1.03)]
                    if not atm_c.empty: atm_ivs.append(float(atm_c['impliedVolatility'].median()))
                    tot_cv+=int(calls['volume'].fillna(0).sum())
                    tot_pv+=int(puts['volume'].fillna(0).sum())
                    tot_coi+=int(calls['openInterest'].fillna(0).sum())
                    tot_poi+=int(puts['openInterest'].fillna(0).sum())
                except: continue
            if not atm_ivs: return None
            atm_iv=float(np.mean(atm_ivs)*100)
            if len(lr)>=252:
                rv=lr.rolling(21).std().dropna()*np.sqrt(252)*100
                ivr=float(np.clip((atm_iv-rv.min())/(rv.max()-rv.min()+1e-6)*100,0,100))
            else: ivr=50.
            pc=tot_pv/(tot_cv+1e-6)
            iv_hv=atm_iv/(hv30+1e-6)
            score=0
            if ivr<30 and iv_hv>1.05: score+=2.5
            if pc<0.6 and atm_iv>hv30: score+=2.0
            if iv_hv>1.3: score+=1.5
            if tot_coi>tot_poi*1.5: score+=1.0
            if iv_hv<0.7: score-=1.5
            if pc>1.5: score-=1.0
            verdict=("STRONG BUY" if score>=4 else "BUY" if score>=2.5 else
                     "WATCH" if score>=1 else "AVOID" if score<=-1 else "NEUTRAL")
            return {'ticker':ticker,'price':round(cp,2),'atm_iv':round(atm_iv,1),
                    'iv_rank':round(ivr,0),'hv_30':round(hv30,1),'iv_hv':round(iv_hv,2),
                    'pc_ratio':round(pc,2),'score':round(score,1),'verdict':verdict}
        except: return None
 
    iv_single=st.text_input("Single Ticker IV Analysis",TICKER,key="iv_single").upper()
    if st.button("Analyze IV",key="iv_btn"):
        with st.spinner("Fetching options..."):
            ir=quick_iv(iv_single)
        if ir:
            ic1,ic2,ic3,ic4,ic5=st.columns(5)
            ic1.metric("ATM IV",f"{ir['atm_iv']:.1f}%")
            ic2.metric("IV Rank",f"{ir['iv_rank']:.0f}/100")
            ic3.metric("HV 30d",f"{ir['hv_30']:.1f}%")
            ic4.metric("IV/HV",f"{ir['iv_hv']:.2f}")
            ic5.metric("P/C Ratio",f"{ir['pc_ratio']:.2f}")
            vc={"STRONG BUY":"#00ff88","BUY":"#44cc66","WATCH":"#ffcc00","NEUTRAL":"#aaa","AVOID":"#ff4444"}
            st.markdown(f"<div style='background:{vc.get(ir['verdict'],'#333')}22;border:2px solid {vc.get(ir['verdict'],'#888')};border-radius:10px;padding:16px;text-align:center;'><h2 style='color:{vc.get(ir['verdict'],'white')};margin:0'>{ir['verdict']}</h2><p>Score: {ir['score']:.1f}/6</p></div>",unsafe_allow_html=True)
        else: st.error("No options data available.")
 
    st.divider()
    st.write("#### Bulk IV Scan")
    iv_uni=st.selectbox("Universe",["S&P 500","NASDAQ 100","Custom"],key="iv_uni")
    iv_dep=st.number_input("Depth",5,200,30,key="iv_dep")
    iv_custom=""
    if iv_uni=="Custom": iv_custom=st.text_area("Tickers","AAPL,TSLA,NVDA,META,AMZN")
    if st.button("Run Bulk IV Scan",key="iv_bulk"):
        if iv_uni=="S&P 500": ul=get_sp500_tickers()[:iv_dep]
        elif iv_uni=="NASDAQ 100": ul=get_nasdaq100_tickers()[:iv_dep]
        else: ul=[t.strip().upper() for t in iv_custom.split(",") if t.strip()][:iv_dep]
        ivres=[]; ivprog=st.progress(0); ivstat=st.empty()
        def iv_worker(t): return quick_iv(t)
        with ThreadPoolExecutor(max_workers=10) as ex:
            fts={ex.submit(iv_worker,t):t for t in ul}
            for i,ft in enumerate(as_completed(fts)):
                r=ft.result()
                if r: ivres.append(r)
                ivprog.progress((i+1)/len(ul))
                ivstat.text(f"{i+1}/{len(ul)} | Found: {len(ivres)}")
        ivprog.empty(); ivstat.empty()
        if ivres:
            ivdf=pd.DataFrame(ivres).sort_values('score',ascending=False)
            buys=ivdf[ivdf['verdict'].isin(['STRONG BUY','BUY'])]
            st.success(f"✅ {len(buys)} BUY signals from {len(ivres)} analyzed")
            vc2={"STRONG BUY":"background-color:#00441b;color:#00ff88",
                 "BUY":"background-color:#1a472a;color:#44cc66",
                 "WATCH":"background-color:#3d3000;color:#ffcc00",
                 "NEUTRAL":"","AVOID":"background-color:#3d0000;color:#ff4444"}
            st.dataframe(ivdf.style.format({"price":"${:.2f}","atm_iv":"{:.1f}%",
                                             "hv_30":"{:.1f}%","iv_hv":"{:.2f}",
                                             "pc_ratio":"{:.2f}","score":"{:.1f}"})
                         .applymap(lambda v:vc2.get(v,""),subset=['verdict']),
                         use_container_width=True)
            st.download_button("Download CSV",ivdf.to_csv(index=False),
                                f"iv_scan_{datetime.now().strftime('%Y%m%d')}.csv","text/csv")
 
# ==========================================
# TAB 12: MARKET SCAN
# ==========================================
with tab12:
    st.write("### 📡 Institutional Total Market Scanner")
    sc_uni=st.selectbox("Universe",["S&P 500","NASDAQ 100","Custom"],key="sc_uni")
    sc_dep=st.number_input("Depth",5,500,30,key="sc_dep")
    sc_custom=""
    if sc_uni=="Custom": sc_custom=st.text_area("Tickers","AAPL,TSLA,NVDA",key="sc_custom")
    sc_eng=st.selectbox("Engine",["GMM (Fast)","Markov (Accurate)"],key="sc_eng")
    sc_ep=sc_eng.split(" ")[0]
    if st.button("Execute Scan",key="sc_run",type="primary"):
        if sc_uni=="S&P 500": sc_ul=get_sp500_tickers()[:sc_dep]
        elif sc_uni=="NASDAQ 100": sc_ul=get_nasdaq100_tickers()[:sc_dep]
        else: sc_ul=[t.strip().upper() for t in sc_custom.split(",") if t.strip()]
        sc_long=[]; sc_cash=[]; sc_prog=st.progress(0); sc_stat=st.empty()
        def sc_worker(t):
            sdf=load_data(t,start_date,end_date)
            if sdf is None: return None
            a=get_master_signal(t,sdf,engine=sc_ep)
            if not a: return None
            return {'Ticker':t,'Price':round(float(sdf['Close'].iloc[-1]),2),
                    'Score':a['sentiment_score'],'Regime':a['regime_label'],
                    'Action':a['regime_sig'],'Vol':a['vol_state']}
        with ThreadPoolExecutor(max_workers=10) as ex:
            fts={ex.submit(sc_worker,t):t for t in sc_ul}
            for i,ft in enumerate(as_completed(fts)):
                r=ft.result()
                if r:
                    if r['Score']>=1: sc_long.append(r)
                    else: sc_cash.append(r)
                sc_prog.progress((i+1)/len(sc_ul)); sc_stat.text(f"{i+1}/{len(sc_ul)}")
        sc_prog.empty(); sc_stat.empty()
        c1s,c2s=st.columns(2)
        with c1s:
            st.subheader(f"🚀 LONG ({len(sc_long)})")
            if sc_long: st.dataframe(pd.DataFrame(sc_long).sort_values('Score',ascending=False),use_container_width=True)
        with c2s:
            st.subheader(f"🛑 CASH ({len(sc_cash)})")
            if sc_cash: st.dataframe(pd.DataFrame(sc_cash).sort_values('Score'),use_container_width=True)
 
# ==========================================
# TAB 13: FED BALANCE SHEET
# ==========================================
with tab13:
    st.write("### 🏦 Federal Reserve Balance Sheet")
    fed_start=st.date_input("From",datetime(2015,1,1))
    with st.spinner("Loading FRED data..."):
        adfs={n:load_fred_data(s) for s,n in FED_ASSETS.items()}
        ldfs={n:load_fred_data(s) for s,n in FED_LIABILITIES.items()}
    asm={n:d.iloc[:,0] for n,d in adfs.items() if d is not None}
    lsm={n:d.iloc[:,0] for n,d in ldfs.items() if d is not None}
    if asm:
        adf2=pd.DataFrame(asm).fillna(0)
        adf2=adf2[adf2.index>=pd.Timestamp(fed_start)]
        fa=go.Figure()
        for col in adf2.columns:
            fa.add_trace(go.Scatter(x=adf2.index,y=adf2[col]/1e3,mode='lines',stackgroup='one',name=col))
        fa.update_layout(title="FED Assets",yaxis_title="Billions $",template="plotly_dark",height=450)
        st.plotly_chart(fa,use_container_width=True)
    walcl=load_fred_data("WALCL")
    if walcl is not None:
        wdf=walcl[walcl.index>=pd.Timestamp(fed_start)].diff().dropna()/1e3
        fc_w=go.Figure()
        wvals=wdf.iloc[:,0]
        fc_w.add_trace(go.Bar(x=wdf.index,y=wvals,
                               marker_color=['green' if v>=0 else 'red' for v in wvals]))
        fc_w.update_layout(title="Weekly Balance Sheet Change",yaxis_title="Billions $",template="plotly_dark",height=350)
        st.plotly_chart(fc_w,use_container_width=True)
    if lsm:
        ldf2=pd.DataFrame(lsm).fillna(0)
        ldf2=ldf2[ldf2.index>=pd.Timestamp(fed_start)]
        fl=go.Figure()
        for col in ldf2.columns:
            fl.add_trace(go.Scatter(x=ldf2.index,y=ldf2[col]/1e3,mode='lines',stackgroup='one',name=col))
        fl.update_layout(title="FED Liabilities",yaxis_title="Billions $",template="plotly_dark",height=400)
        st.plotly_chart(fl,use_container_width=True)
 
# ==========================================
# TAB 14: OPTIONS IV SURFACE
# ==========================================
with tab14:
    st.write("### 📊 3D Implied Volatility Surface")
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    with st.spinner("Fetching options..."):
        try:
            tk14=yf.Ticker(TICKER); exps14=tk14.options
            if not exps14: st.error("No options data."); st.stop()
            mx=st.slider("Max Expirations",1,min(15,len(exps14)),min(6,len(exps14)))
            sd14=[]; cp14=float(df_main['Close'].iloc[-1])
            for exp in exps14[:mx]:
                dte14=(datetime.strptime(exp,"%Y-%m-%d")-datetime.now()).days
                if dte14<1: continue
                try:
                    c14=tk14.option_chain(exp).calls
                    for _,row in c14.iterrows():
                        if row['impliedVolatility']>0 and (row.get('volume') or 0)>0:
                            mn=row['strike']/cp14
                            if 0.7<=mn<=1.3:
                                sd14.append({'DTE':dte14,'Moneyness':mn,'IV':row['impliedVolatility']})
                except: continue
            if sd14:
                sf14=pd.DataFrame(sd14)
                sf14['Mb']=sf14['Moneyness'].round(2)
                sp14=sf14.groupby(['DTE','Mb'])['IV'].mean().unstack()
                sp14=sp14.interpolate(method='linear',axis=1).bfill(axis=1).ffill(axis=1)
                sp14=sp14.interpolate(method='linear',axis=0).bfill(axis=0).ffill(axis=0)
                f3d=go.Figure(go.Surface(z=sp14.values,x=sp14.columns,y=sp14.index,colorscale='Viridis'))
                f3d.update_layout(title=f"{TICKER} IV Surface",height=650,template="plotly_dark",
                                   scene=dict(xaxis_title='Moneyness',yaxis_title='DTE',zaxis_title='IV'))
                st.plotly_chart(f3d,use_container_width=True)
            else: st.warning("Insufficient liquid options data.")
        except Exception as e: st.error(f"Options error: {e}")
 
# ==========================================
# TAB 15: HURST EXPONENT
# ==========================================
with tab15:
    if df_main is None:
        st.warning("Load a ticker first."); st.stop()
    st.write("### 〰️ Hurst Exponent (Trend vs Mean-Reversion)")
    hc1h,hc2h=st.columns(2)
    with hc1h: hw_h=st.slider("Window",20,252,100,key="hh_win")
    with hc2h: hth=st.slider("Trend threshold H>",0.4,0.7,0.55,step=0.01,key="hh_thr")
    with st.spinner("Computing..."):
        try:
            hs_h=rolling_hurst(df_main['Close'],window=hw_h)
            fh_h=make_subplots(rows=2,cols=1,shared_xaxes=True,
                                subplot_titles=("Price","Hurst Exponent"))
            fh_h.add_trace(go.Scatter(x=df_main.index,y=df_main['Close'],
                                       mode='lines',line=dict(color='gray'),name='Price'),row=1,col=1)
            fh_h.add_trace(go.Scatter(x=hs_h.index,y=hs_h,
                                       mode='lines',line=dict(color='cyan'),name='H'),row=2,col=1)
            fh_h.add_hline(y=0.5,line_dash="dash",line_color="gray",row=2,col=1,annotation_text="Random Walk")
            fh_h.add_hline(y=hth,line_dash="dash",line_color="green",row=2,col=1,annotation_text=f"Trend>{hth}")
            fh_h.add_hline(y=1-hth,line_dash="dash",line_color="purple",row=2,col=1,annotation_text=f"MR<{1-hth:.2f}")
            is_tr=hs_h>hth; is_mr=hs_h<(1-hth)
            highlight_plotly_zones(fh_h,is_tr,'green',opacity=0.1,row=1,col=1)
            highlight_plotly_zones(fh_h,is_mr,'purple',opacity=0.1,row=1,col=1)
            fh_h.update_layout(height=550,template="plotly_dark",hovermode="x unified")
            st.plotly_chart(fh_h,use_container_width=True)
            last_h=float(hs_h.dropna().iloc[-1])
            st.metric("Current H",f"{last_h:.3f}",
                       delta="TRENDING" if last_h>hth else "MEAN REVERTING" if last_h<1-hth else "RANDOM WALK")
        except Exception as e: st.error(f"Hurst error: {e}")
 
# ==========================================
# TAB 16: HOT 10 + TRADE INTELLIGENCE
# ==========================================
with tab16:
    hot_tabs=st.tabs(["🔥 Hot 10 (Daily Momentum)","🎯 Trade Intelligence"])
 
    with hot_tabs[0]:
        st.write("### 🔥 Daily Top 10 (Institutional Hot List)")
        hc1,hc2,hc3=st.columns(3)
        with hc1: h_uni=st.selectbox("Universe",["S&P 500","NASDAQ 100"],key="h_uni")
        with hc2: h_mp=st.number_input("Min Price ($)",5.,50.,5.,key="h_mp")
        with hc3: h_top=st.number_input("Target Buys",1,50,10,key="h_top")
        h_vxm=st.number_input("VIX Multiplier",0.5,3.,1.5,step=0.1,key="h_vxm")
        if st.button("🚀 Scan Hot List",key="h_run",type="primary"):
            ul_h=get_sp500_tickers() if h_uni=="S&P 500" else get_nasdaq100_tickers()
            import gc
            with st.spinner("Bulk download..."):
                dl_h=ul_h+["^VIX"]
                chunk_size=200; dfs_h=[]
                hprog=st.progress(0)
                for i in range(0,len(dl_h),chunk_size):
                    chunk=dl_h[i:i+chunk_size]
                    df_c=yf.download(chunk,period="20d",threads=True,progress=False)
                    dfs_h.append(df_c); gc.collect()
                    hprog.progress(min(1,(i+chunk_size)/len(dl_h)))
                hprog.empty()
            if dfs_h:
                df_bulk=pd.concat(dfs_h,axis=1) if len(dfs_h)>1 else dfs_h[0]
                closes=df_bulk['Close'].ffill()
                opens=df_bulk.get('Open',closes).ffill()
                vols=df_bulk.get('Volume',pd.DataFrame()).ffill()
                if len(closes)>=2:
                    vix_lvl=float(closes["^VIX"].dropna().iloc[-1]) if "^VIX" in closes.columns else 15.
                    thresh=(vix_lvl/np.sqrt(252))/100*h_vxm
                    st.info(f"VIX: {vix_lvl:.1f} | Adaptive threshold: {thresh*100:.2f}%")
                    lc=closes.iloc[-1]; pc=closes.iloc[-2]
                    dr=(lc-pc)/pc
                    lo_=opens.iloc[-1] if not opens.empty else lc
                    gap=(lo_-pc)/pc; intra=(lc-lo_)/(lo_+1e-6)
                    if not vols.empty:
                        lv2=vols.iloc[-1]
                        av2=vols.rolling(20).mean().iloc[-1] if len(vols)>=20 else vols.mean()
                        rvol2=lv2/(av2+1)
                        dv2=lc*lv2
                    else:
                        rvol2=pd.Series(0,index=lc.index); dv2=pd.Series(0,index=lc.index)
                    valid=(lc>=h_mp)&(dr>thresh)
                    valid=valid.fillna(False)
                    vtks=[t for t in valid[valid].index if t!="^VIX"]
                    hot=[]
                    for t in vtks:
                        try:
                            sc_h=float(dr[t])/thresh
                            hot.append({'Ticker':str(t),'Price':round(float(lc[t]),2),
                                        'Daily%':round(float(dr[t])*100,2),'Gap%':round(float(gap.get(t,0))*100,2),
                                        'Intraday%':round(float(intra.get(t,0))*100,2),
                                        'VIX Multiple':round(sc_h,2),
                                        'RVOL':round(float(rvol2.get(t,0)),2) if hasattr(rvol2,'get') else 0,
                                        'Dollar Vol':f"${float(dv2.get(t,0))/1e6:.1f}M" if hasattr(dv2,'get') else "N/A"})
                        except: pass
                    if hot:
                        hot_df=pd.DataFrame(hot).sort_values('VIX Multiple',ascending=False)
                        st.write(f"**{len(hot_df)} momentum candidates** → running IV + Regime verification...")
                        final=[]; fp=st.progress(0)
                        for i,t in enumerate(hot_df['Ticker'].tolist()):
                            if len(final)>=int(h_top): break
                            tdf=load_data(t,datetime.now()-timedelta(days=730),datetime.now())
                            if tdf is not None:
                                a=get_master_signal(t,tdf,engine="GMM")
                                if a and ("LONG" in a.get('regime_sig','') or "BUY" in a.get('regime_sig','')):
                                    row=hot_df[hot_df['Ticker']==t].iloc[0].to_dict()
                                    row['Regime']=a['regime_label']; row['Vol State']=a['vol_state']
                                    final.append(row)
                            fp.progress((i+1)/len(hot_df))
                        fp.empty()
                        if final:
                            st.success(f"🔥 {len(final)} Institutional BUY setups confirmed!")
                            st.dataframe(pd.DataFrame(final).style.background_gradient(
                                subset=['Daily%','VIX Multiple'],cmap='YlOrRd'),use_container_width=True)
                        else: st.error("No setups passed institutional verification today.")
                    else: st.warning("No stocks cleared the momentum threshold today.")
 
    with hot_tabs[1]:
        st.info("👆 Use the dedicated **🏆 Trade Intelligence** tab for full 7-layer analysis.")
 
# ==========================================
# TAB 17: TRADE INTELLIGENCE (7-Layer System)
# ==========================================
with tab17:
    st.markdown("""
    ## 🏆 Trade Intelligence — 7-Layer Institutional Confirmation
    
    This module runs **7 independent confirmation layers** on any stock before declaring 
    it a valid trade — exactly how institutional desks confirm signals before sizing in.
    
    **The 7 Layers:**  
    `1` 📊 Market Condition (SPY/QQQ/IWM trend + VIX level + Market Breadth)  
    `2` 🏭 Sector Strength (Sector ETF trend + Stock vs Sector RS + Stock vs SPY)  
    `3` 📈 Price Action (HH/HL structure + Entry patterns + Candle analysis)  
    `4` 📍 Support & Resistance (Key S/R levels + Risk:Reward ratio)  
    `5` 💧 Volume & Liquidity (Dollar volume filter + RVOL + Options liquidity)  
    `6` 📅 Earnings & News Risk (Earnings date + Short interest + Sector risk flags)  
    `7` 🏆 Entry Quality Checklist (Pass/fail scorecard + Composite verdict)  
    """)
    default_ti_ticker = TICKER if df_main is not None else "AAPL"
    render_trade_intelligence(default_ti_ticker)
 
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Unified Quant Suite | GARCH · Markov · Kalman · Heston · IV Scanner · 🏆 Trade Intelligence")
