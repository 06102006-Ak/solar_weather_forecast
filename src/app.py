from pathlib import Path
import sys

# Make `src.*` imports work whether launched as a module or with Streamlit.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.utils import load_config, project_path
from src.signals.fourier import fft_spectrum
from src.signals.filters import causal_moving_average
from src.forecasting.baselines import persistence, seasonal_naive, fourier
from src.evaluation.metrics import regression_metrics

st.set_page_config(page_title="SolarCast | Explainable Forecast Lab", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#eaf1ff; --muted:#9eacc4; }
.block-container {padding-top:1.5rem; padding-bottom:3rem; max-width:1450px;}
[data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#091322 0%,#101c2d 52%,#0b1523 100%);}
[data-testid="stSidebar"] {background:#0b1728; border-right:1px solid #24334a;}
.hero {padding:1.5rem 1.7rem; border:1px solid #2b405a; border-radius:20px; background:linear-gradient(120deg,rgba(28,61,91,.88),rgba(24,42,68,.65)); margin-bottom:1.1rem;}
.eyebrow {text-transform:uppercase; letter-spacing:.16em; color:#8ed9ff; font-size:.75rem; font-weight:700;}
.hero h1 {font-size:2.45rem; line-height:1.1; margin:.45rem 0; color:#f5f8ff;}
.hero p {color:#c4d2e7; max-width:900px; font-size:1.05rem; margin-bottom:.3rem;}
.pill {display:inline-block;border:1px solid #39536f;border-radius:30px;padding:.28rem .65rem;color:#c8eaff;font-size:.8rem;margin:.25rem .3rem .1rem 0;}
.section {padding:1rem 1.1rem; border:1px solid #293a52; border-radius:16px; background:rgba(17,31,49,.72); margin-bottom:.8rem;}
.small-muted {color:#9eacc4;font-size:.9rem;}
</style>
<div class="hero">
 <div class="eyebrow">Signal processing · Time-series forecasting · Model explainability</div>
 <h1>☀️ SolarCast</h1>
 <p>Explore historical weather, uncover periodic patterns, generate baseline forecasts, and test whether signal-processing methods improve prediction.</p>
 <span class="pill">Fourier analysis</span><span class="pill">Digital filtering</span><span class="pill">Forecast comparison</span><span class="pill">Transparent evaluation</span>
</div>
""", unsafe_allow_html=True)

try:
    cfg = load_config()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("From the project root, copy `config/settings.example.json` to `config/settings.json`, then fetch and preprocess the data. See the Setup page in this app's README.")
    st.stop()

with st.sidebar:
    st.markdown("## ☀️ SolarCast Lab")
    page = st.radio("Navigate", ["Project overview", "Explore the data", "Forecast lab", "Signal-processing lab", "Model evaluation", "How to explain this project"], index=0)
    st.divider()
    resolution = st.selectbox("Data resolution", ["hourly", "daily"], index=1, help="Hourly data is useful for within-day patterns; daily data is easier for long-term trends.")
    st.caption("Research prototype · NASA POWER historical data")

path = project_path(cfg["paths"]["processed_dir"]) / f"weather_{resolution}.csv"
if not path.exists():
    st.warning(f"No processed {resolution} dataset found yet.")
    st.markdown("1. Run `python -m src.data.fetch_data --resolution " + resolution + "`\n2. Run `python -m src.data.preprocess --resolution " + resolution + "`\n3. Refresh this page.")
    st.stop()

df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp").set_index("timestamp")
labels = {"ALLSKY_SFC_SW_DWN": "Solar radiation", "T2M": "Air temperature"}
units = {"ALLSKY_SFC_SW_DWN": "NASA POWER solar-radiation units", "T2M": "°C"}
colors = {"ALLSKY_SFC_SW_DWN": "#f7b844", "T2M": "#55c7ff"}
targets = [c for c in labels if c in df.columns]
if not targets:
    st.error("Expected target columns were not found in the processed dataset. Check the NASA POWER response and preprocessing output.")
    st.stop()

def nice_name(c): return labels.get(c, c)
def clean_series(c):
    s = pd.to_numeric(df[c], errors="coerce").dropna()
    return s[np.isfinite(s.to_numpy())]
def chart_layout(fig, height=370):
    fig.update_layout(height=height, margin=dict(l=12,r=12,t=50,b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,17,29,.3)", font=dict(color="#dce7f7"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), hovermode="x unified")
    fig.update_xaxes(gridcolor="rgba(130,155,185,.14)")
    fig.update_yaxes(gridcolor="rgba(130,155,185,.14)")
    return fig

def make_forecast(s, model, horizon, resolution, harmonics):
    x = s.to_numpy(dtype=float)
    if len(x) < 30: raise ValueError("At least 30 valid observations are needed for a meaningful demonstration.")
    if model == "Persistence (last value)": pred = persistence(x, horizon)
    elif model == "Seasonal naive":
        season = 24 if resolution == "hourly" else 7
        if len(x) < season: raise ValueError(f"Need at least {season} observations for seasonal naive.")
        pred = seasonal_naive(x, horizon, season)
    else:
        period = 24 if resolution == "hourly" else 7
        pred = fourier(x, horizon, period, harmonics)
    freq = "h" if resolution == "hourly" else "D"
    future_idx = pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)
    return pd.Series(pred, index=future_idx, name="Forecast")

if page == "Project overview":
    st.subheader("Your weather data, made explorable")
    st.write("SolarCast is a small forecasting research lab. It does not claim to be an official live weather service: it uses historical observations to study patterns and compare simple forecasting methods.")
    all_n = len(df); date_min = df.index.min(); date_max = df.index.max()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Observations", f"{all_n:,}")
    c2.metric("Signals available", str(len(targets)))
    c3.metric("Start date", date_min.strftime("%d %b %Y"))
    c4.metric("End date", date_max.strftime("%d %b %Y"))
    st.markdown("### What can you do here?")
    a,b,c = st.columns(3)
    with a:
        st.markdown("<div class='section'><h3>📈 Explore</h3><p>Choose solar radiation or temperature, inspect historical changes, and see gaps in the dataset.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='section'><h3>🔮 Forecast</h3><p>Choose a prediction horizon and compare persistence, seasonal-naive, and Fourier-based forecasts.</p></div>", unsafe_allow_html=True)
    with c:
        st.markdown("<div class='section'><h3>〰️ Understand</h3><p>Apply a causal moving-average filter and inspect the FFT spectrum to discuss periodic structure.</p></div>", unsafe_allow_html=True)
    st.markdown("### At a glance")
    cols = st.columns(len(targets))
    for col, target in zip(cols, targets):
        s=clean_series(target)
        col.metric(f"Latest {nice_name(target).lower()}", f"{s.iloc[-1]:.2f}", help=f"{units[target]} · latest available historical observation")
    st.info("Start with **Explore the data**, then go to **Forecast lab** to generate a forecast interactively.")

elif page == "Explore the data":
    st.subheader("Explore historical observations")
    target = st.selectbox("Signal to inspect", targets, format_func=nice_name)
    s = clean_series(target)
    max_points = min(len(s), 5000)
    default_points = min(max_points, 500 if resolution == "hourly" else 365)
    n_points = st.slider("Observations to display", min_value=min(30,max_points), max_value=max_points, value=max(min(30,max_points),default_points), step=max(1,max_points//1000))
    shown = s.iloc[-n_points:]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=shown.index,y=shown.values,mode="lines",name=nice_name(target),line=dict(color=colors[target],width=1.8)))
    fig.update_layout(title=f"{nice_name(target)} · most recent {len(shown):,} observations", yaxis_title=units[target])
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Valid values", f"{s.size:,}")
    c2.metric("Missing values", f"{int(df[target].isna().sum()):,}")
    c3.metric("Mean", f"{s.mean():.2f}")
    c4.metric("Standard deviation", f"{s.std():.2f}")
    with st.expander("View data sample"):
        st.dataframe(df[[target]].tail(100), use_container_width=True)
    st.download_button("Download displayed signal as CSV", shown.rename(nice_name(target)).to_frame().to_csv().encode(), file_name=f"{target}_{resolution}.csv", mime="text/csv")

elif page == "Forecast lab":
    st.subheader("Forecast lab")
    st.write("Select a target and forecasting method. The model uses only observations available up to the final historical timestamp; future values are predictions, not downloaded future weather.")
    c1,c2,c3 = st.columns([1.2,1,1])
    with c1: target=st.selectbox("What should we forecast?",targets,format_func=nice_name)
    with c2:
        default_h=24 if resolution=="hourly" else 7
        horizon=st.slider("Forecast horizon", min_value=1, max_value=168 if resolution=="hourly" else 90, value=default_h, help="Hours for hourly data; days for daily data.")
    with c3: model=st.selectbox("Forecasting method",["Fourier harmonic regression","Seasonal naive","Persistence (last value)"], help="Compare a periodic model against simple baselines.")
    harmonics=st.slider("Fourier harmonics",1,8,3,disabled=model!="Fourier harmonic regression",help="More harmonics fit more periodic detail but may overfit.")
    s=clean_series(target)
    if len(s)<30:
        st.warning("Not enough observations for a forecast. Fetch a longer date range first.")
    else:
        try:
            pred=make_forecast(s,model,horizon,resolution,harmonics)
            history=s.iloc[-min(len(s), max(60,horizon*3)):]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=history.index,y=history.values,name="Historical",mode="lines",line=dict(color=colors[target],width=2)))
            fig.add_trace(go.Scatter(x=pred.index,y=pred.values,name="Forecast",mode="lines+markers",line=dict(color="#c79cff",width=3,dash="dash"),marker=dict(size=5)))
            fig.add_vline(x=s.index[-1],line_dash="dot",line_color="#aab8cc",annotation_text="Forecast starts",annotation_position="top")
            fig.update_layout(title=f"{nice_name(target)} forecast · {horizon} {'hours' if resolution=='hourly' else 'days'}",yaxis_title=units[target])
            st.plotly_chart(chart_layout(fig,430),use_container_width=True)
            a,b,c=st.columns(3)
            a.metric("Last observation",f"{s.iloc[-1]:.2f}")
            b.metric("Forecast mean",f"{pred.mean():.2f}")
            c.metric("Forecast range",f"{pred.min():.2f} to {pred.max():.2f}")
            st.caption("Interpretation note: these are baseline forecasts. Fourier extrapolation assumes repeating patterns; real weather can change abruptly. Solar radiation forecasts may not enforce physically realistic night-time zeros.")
            st.download_button("Download forecast CSV",pred.rename(f"{nice_name(target)} forecast").to_frame().to_csv().encode(),file_name=f"forecast_{target}_{resolution}.csv",mime="text/csv")
        except Exception as e: st.error(f"Could not generate this forecast: {e}")
    st.markdown("### How to interpret the methods")
    st.markdown("- **Persistence:** assumes the next values remain like the latest observation.\n- **Seasonal naive:** repeats the most recent daily cycle (hourly data) or weekly cycle (daily data).\n- **Fourier regression:** fits sine/cosine components to represent repeating patterns and extrapolates them forward.")

elif page == "Signal-processing lab":
    st.subheader("Signal-processing lab")
    target=st.selectbox("Signal",targets,format_func=nice_name)
    s=clean_series(target)
    if len(s)<4: st.warning("Need at least four valid observations to inspect a spectrum.")
    else:
        maxwin=24 if resolution=="hourly" else 14
        window=st.slider("Moving-average window",1,min(maxwin,max(1,len(s)//3)),min(3,max(1,len(s)//3)),help="This is a causal trailing window; it does not use future observations.")
        n_points=st.slider("Recent observations",min_value=min(60,len(s)),max_value=min(len(s),5000),value=min(len(s),500 if resolution=="hourly" else 365),step=max(1,min(len(s),5000)//100))
        shown=s.iloc[-n_points:]
        filtered=causal_moving_average(shown.to_numpy(),window)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=shown.index,y=shown.values,name="Raw signal",line=dict(color=colors[target],width=1.2))); fig.add_trace(go.Scatter(x=shown.index,y=filtered,name=f"Causal moving average ({window})",line=dict(color="#c79cff",width=2.5)))
        fig.update_layout(title="Raw signal vs. smoothed signal",yaxis_title=units[target]); st.plotly_chart(chart_layout(fig),use_container_width=True)
        st.markdown("#### FFT magnitude spectrum")
        spec=fft_spectrum(shown.to_numpy()); spec=spec[spec.frequency>0].copy()
        if not spec.empty:
            top=spec.nlargest(min(20,len(spec)),"magnitude").sort_values("frequency")
            f=px.bar(top,x="frequency",y="magnitude",hover_data=["frequency","magnitude"],title="Strongest non-zero frequency components")
            st.plotly_chart(chart_layout(f,330),use_container_width=True)
        with st.expander("What does this mean?"):
            st.markdown("**Filtering:** a trailing moving average smooths short-term variation. A larger window smooths more, but can hide fast changes and introduces lag.\n\n**FFT:** the Fast Fourier Transform decomposes a sampled signal into frequency components. Strong peaks can indicate repeating structure. A peak alone does not prove a real physical cycle; sampling interval, missing data, trends, and window length matter.\n\n**Connection to forecasting:** Fourier harmonic regression uses sine and cosine terms to model repeating patterns. The spectrum helps you investigate whether that assumption is plausible.")

elif page == "Model evaluation":
    st.subheader("Model evaluation: evidence, not just a pretty graph")
    result=project_path(cfg["paths"]["results_dir"])/f"baseline_metrics_{resolution}.csv"
    if result.exists():
        metrics=pd.read_csv(result)
        st.caption("These metrics come from the project's saved chronological holdout experiment. The current experiment forecasts the entire test block at once, so treat it as an initial benchmark rather than a final validation.")
        st.dataframe(metrics,use_container_width=True,hide_index=True)
        if {"model","rmse"}.issubset(metrics.columns):
            target_filter=st.selectbox("Target to compare",metrics["target"].dropna().unique().tolist()) if "target" in metrics else None
            subset=metrics[metrics["target"]==target_filter] if target_filter else metrics
            fig=px.bar(subset,x="model",y="rmse",color="model",title="RMSE by baseline model (lower is better)")
            st.plotly_chart(chart_layout(fig,350),use_container_width=True)
        st.download_button("Download evaluation metrics",metrics.to_csv(index=False).encode(),file_name=f"baseline_metrics_{resolution}.csv",mime="text/csv")
    else:
        st.info("No saved metric file yet. Run the experiment command below from the project root.")
        st.code(f"python -m src.experiments.run_baselines --resolution {resolution}",language="powershell")
    st.markdown("### What should a fair evaluation include?")
    st.markdown("1. Split by time: train on the past, evaluate on later observations.\n2. Compare against simple baselines.\n3. Report MAE and RMSE in understandable units.\n4. Use rolling-origin evaluation and several forecast horizons before making strong claims.\n5. Never use future observations as model inputs.")

elif page == "How to explain this project":
    st.subheader("Your project explanation")
    st.markdown("### 30-second explanation")
    st.info("SolarCast is an explainable time-series forecasting system for historical solar radiation and air temperature. It combines signal-processing tools such as moving-average filtering and Fourier analysis with baseline forecasting methods. Users can explore the data, generate forecasts for a chosen horizon, and compare model errors to understand both the predictions and their limitations.")
    st.markdown("### The problem")
    st.write("Weather measurements contain long-term trends, repeating patterns, and irregular changes. A model should learn useful temporal structure without accidentally using information from the future.")
    st.markdown("### The workflow")
    st.markdown("**Historical data → preprocessing → signal analysis → forecast generation → chronological evaluation → interpretation.**")
    st.markdown("### Where is the signal processing?")
    st.markdown("- **Discrete-time signal:** measurements indexed by hour or day.\n- **Digital filter:** a causal moving average smooths short-term fluctuations.\n- **Fourier analysis:** FFT reveals the relative strength of frequency components.\n- **Harmonic regression:** sine/cosine terms model periodic patterns and extend them into the future.\n- **Evaluation:** MAE/RMSE quantify prediction errors on held-out time periods.")
    st.markdown("### What can the user do?")
    st.markdown("- Choose hourly or daily historical data.\n- Select solar radiation or temperature.\n- Adjust the time window and filter strength.\n- Inspect frequency components.\n- Select a forecasting algorithm and prediction horizon.\n- Download the forecast and metrics as CSV.\n- Compare baseline methods and discuss limitations.")
    st.warning("Be precise in your presentation: NASA POWER data used here are historical data, not a live forecast feed. Fourier and naive methods are baselines, not guaranteed high-accuracy weather predictions. Demonstrate accuracy with chronological/rolling-origin tests before claiming the model is superior.")

st.divider()
st.markdown("<p class='small-muted'>SolarCast is an educational research prototype. Forecasts are experimental and should not be used for operational decisions.</p>",unsafe_allow_html=True)
