import argparse
import numpy as np
import pandas as pd
from src.utils import load_config,project_path
from src.forecasting.baselines import persistence,seasonal_naive,fourier
from src.evaluation.metrics import regression_metrics

TARGETS=["ALLSKY_SFC_SW_DWN","T2M"]
def evaluate(series,resolution,test_fraction,harmonics):
    x=series.astype(float).to_numpy(); x=x[np.isfinite(x)]
    if len(x)<100: raise ValueError(f"Only {len(x)} finite values")
    cut=int(len(x)*(1-test_fraction)); train,test=x[:cut],x[cut:]; h=len(test)
    preds={"persistence":persistence(train,h)}
    if resolution=="hourly":
        preds["seasonal_naive_24h"]=seasonal_naive(train,h,24)
        preds["fourier_daily"]=fourier(train,h,24,harmonics)
    else:
        preds["seasonal_naive_7d"]=seasonal_naive(train,h,7)
        preds["fourier_weekly"]=fourier(train,h,7,min(harmonics,3))
        if len(train)>=730: preds["fourier_annual"]=fourier(train,h,365.2425,3)
    return [{"target":series.name,"model":k,**regression_metrics(test,v)} for k,v in preds.items()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--resolution",choices=["hourly","daily"],required=True); ap.add_argument("--config")
    a=ap.parse_args(); cfg=load_config(a.config); proc=project_path(cfg["paths"]["processed_dir"]); path=proc/f"weather_{a.resolution}.csv"
    if not path.exists(): raise FileNotFoundError(f"{path} missing; fetch and preprocess first")
    df=pd.read_csv(path,parse_dates=["timestamp"]).set_index("timestamp"); rows=[]
    for target in TARGETS:
        if target in df and df[target].notna().sum()>=100:
            rows += evaluate(df[target].dropna(),a.resolution,cfg["forecast"]["test_fraction"],cfg["forecast"].get(f"fourier_harmonics_{a.resolution}",3))
    result=pd.DataFrame(rows); out=project_path(cfg["paths"]["results_dir"]); out.mkdir(parents=True,exist_ok=True)
    dest=out/f"baseline_metrics_{a.resolution}.csv"; result.to_csv(dest,index=False)
    print(result.to_string(index=False)); print(f"Saved {dest}")
    print("Note: this is a fixed-holdout stress test forecasting the entire test block. Implement rolling-origin evaluation before final conclusions.")
if __name__=="__main__": main()
