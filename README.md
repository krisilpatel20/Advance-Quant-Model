# Advanced Quantitative Financial Thesis Dashboard

A comprehensive Streamlit-based financial modeling dashboard featuring advanced quantitative techniques including GARCH/EGARCH volatility modeling, Markov regime switching, stochastic simulations (Heston/Jump Diffusion), Kalman filtering, and macro factor analysis.

## Features

- **GARCH/EGARCH Volatility Modeling**: Advanced volatility dynamics with leverage effect detection
- **Markov Regime Switching**: Identify hidden market states (Bull/Bear/Crisis)
- **Stochastic Simulations**: Heston and Merton Jump Diffusion models
- **Kalman Filter Analysis**: Pairs trading and trend extraction
- **Macro Factor Sensitivity**: Correlation analysis with commodities, bonds, and currencies
- **Structural Decomposition**: Time series decomposition (Trend/Seasonal/Residual)

## Installation & Local Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Clone/Download the Files
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Locally
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Deployment Options

### Option 1: Deploy on Streamlit Cloud (Recommended - Free & Public URL)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Financial Thesis Dashboard"
   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select:
     - Repository: `YOUR_USERNAME/REPO_NAME`
     - Branch: `main`
     - Main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Your public URL** will be: `https://YOUR_USERNAME-REPO_NAME-RANDOM.streamlit.app`

### Option 2: Deploy on AWS/GCP/Heroku

For cloud deployment (AWS, GCP, Heroku), create an additional `Procfile`:

```bash
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### Option 3: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY streamlit_app.py .
COPY .streamlit/ .streamlit/
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t financial-thesis .
docker run -p 8501:8501 financial-thesis
```

## Usage Guide

### Sidebar Parameters
- **Market Region**: Choose between US (USD) or Indian (INR) markets
- **Main Ticker**: Enter any valid stock ticker (e.g., BTC-USD, RELIANCE.NS)
- **Pair Ticker**: For pairs trading analysis
- **Date Range**: Select historical period for analysis
- **Risk-Free Rate**: Set the risk-free rate for calculations

### Tab Features

**Tab 1 - Volatility (GARCH)**
- GARCH(1,1) and GJR-GARCH models
- Conditional volatility estimation
- Leverage effect analysis

**Tab 2 - Regime Switching**
- Markov regime identification
- Bull/Bear/Crisis state detection
- Regime persistence and duration analysis

**Tab 3 - Stochastic Simulations**
- Heston stochastic volatility model
- Merton Jump Diffusion paths
- Monte Carlo price projections
- Confidence intervals

**Tab 4 - Kalman Filter**
- Pairs trading hedge ratio estimation
- Trend extraction and smoothing
- Mean reversion signals

**Tab 5 - Macro Factors**
- Cross-asset correlations
- Energy, commodity, and FX sensitivity
- Structural thesis validation

**Tab 6 - Structural**
- Time series decomposition
- Trend/Seasonal/Residual analysis
- Periodicity detection

## Performance Notes

- For large datasets (5+ years daily data), some models may take 10-30 seconds to fit
- Regime switching models require 3+ years of data for stable results
- Use weekly data for faster computations if daily is too slow

## Troubleshooting

**"arch library is not installed"**
- Run: `pip install arch`

**Model convergence issues**
- Reduce lookback period
- Disable "Switching Trend" in regime switching
- Use weekly instead of daily data

**Slow performance**
- Reduce date range
- Use lower data frequency (weekly vs daily)
- Reduce number of Monte Carlo paths

## Requirements

See `requirements.txt` for full dependencies. Main packages:
- `streamlit` - Web framework
- `yfinance` - Financial data
- `statsmodels` - Statistical modeling
- `arch` - GARCH models
- `plotly` - Interactive charts

## License

MIT License - Feel free to modify and distribute

## Support & Contributions

For issues or feature requests, please open an issue on GitHub.

---

**Created with ❤️ for quantitative finance enthusiasts**
