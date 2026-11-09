# Experiment plan

1. Validate timestamps, units, coverage and missingness.
2. Inspect time series and FFT spectra.
3. Compare persistence, seasonal-naive and Fourier baselines.
4. Compare raw signals with causally filtered variants.
5. Add rolling-origin backtesting before final reporting.
6. Do not shuffle rows or use future weather covariates unavailable at forecast time.
7. Report MAE/RMSE in original units and document data provenance.
