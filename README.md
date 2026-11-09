# Solar Radiation & Weather Temperature Forecast Engine

Research prototype focused on Unit 4: signals, Fourier analysis, convolution, and filters.

## Quick start (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config\settings.example.json config\settings.json
python -m src.data.fetch_data --resolution hourly
python -m src.data.fetch_data --resolution daily
python -m src.data.preprocess --resolution hourly
python -m src.data.preprocess --resolution daily
python -m src.experiments.run_baselines --resolution hourly
python -m src.experiments.run_baselines --resolution daily
python -m streamlit run src/app.py
```
If activation is blocked, run ` .\.venv\Scripts\python.exe -m pip install -r requirements.txt` and replace `python` with ` .\.venv\Scripts\python.exe`.

## Data
NASA POWER docs:
- Hourly: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- Daily: https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- Parameters: https://power.larc.nasa.gov/parameters/

Initial parameters: `ALLSKY_SFC_SW_DWN` and `T2M`. Verify endpoint support, units, missing values, and temporal meaning before interpreting results. NASA POWER historical data are not a live weather forecast.

## Method
FFT spectrum, causal moving-average filter, persistence, seasonal-naive, and Fourier harmonic regression. The initial evaluator uses a fixed chronological holdout as a smoke test; extend to rolling-origin evaluation before making final claims. Do not shuffle time-series data or use future weather variables unavailable at prediction time.


## Interactive dashboard
The redesigned SolarCast dashboard includes Project overview, Explore the data, Forecast lab, Signal-processing lab, Model evaluation, and How to explain this project pages. Start it from the project root with:

```powershell
python -m streamlit run src/app.py
```

If PowerShell activation is blocked, use ` .\.venv\Scripts\python.exe -m streamlit run src/app.py`. The Forecast lab offers baseline forecasts only; they are for learning and comparison, not official weather guidance.
