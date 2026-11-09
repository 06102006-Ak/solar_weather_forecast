import argparse
import pandas as pd
from src.utils import load_config, project_path

def clean(df,resolution):
    if "timestamp_key" not in df: raise ValueError("timestamp_key missing")
    raw=df["timestamp_key"].astype(str)
    fmt="%Y%m%d%H" if resolution=="hourly" else "%Y%m%d"
    df=df.copy(); df["timestamp"]=pd.to_datetime(raw,format=fmt,errors="coerce")
    df=df.dropna(subset=["timestamp"]).drop(columns=["timestamp_key"]).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
    df=df.replace([-999,-999.0,-9999,-9999.0],float("nan"))
    if not df.empty:
        freq="h" if resolution=="hourly" else "D"
        df=df.reindex(pd.date_range(df.index.min(),df.index.max(),freq=freq,name="timestamp"))
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--resolution",choices=["hourly","daily"],required=True); ap.add_argument("--config")
    a=ap.parse_args(); cfg=load_config(a.config)
    raw=project_path(cfg["paths"]["raw_dir"]); proc=project_path(cfg["paths"]["processed_dir"]); proc.mkdir(parents=True,exist_ok=True)
    matches=sorted(raw.glob(f"nasa_power_{a.resolution}_*.csv"))
    if not matches: raise FileNotFoundError(f"No raw {a.resolution} data. Run fetch_data first.")
    df=clean(pd.read_csv(matches[-1],dtype={"timestamp_key":str}),a.resolution)
    out=proc/f"weather_{a.resolution}.csv"; df.to_csv(out,index_label="timestamp")
    print(f"Saved {out}; rows={len(df)}"); print("Missing values:\n",df.isna().sum())
if __name__=="__main__": main()
