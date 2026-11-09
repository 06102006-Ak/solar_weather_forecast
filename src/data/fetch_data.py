import argparse, json
import pandas as pd
import requests
from src.utils import load_config, project_path

BASE = "https://power.larc.nasa.gov/api/temporal"
def fetch(resolution, cfg):
    if resolution not in ("hourly","daily"): raise ValueError("resolution must be hourly or daily")
    loc, d = cfg["location"], cfg["data"]
    params = {"parameters": ",".join(d[f"{resolution}_parameters"]), "community": d.get("community","RE"),
              "longitude": loc["longitude"], "latitude": loc["latitude"],
              "start": d["start_date"].replace("-",""), "end": d["end_date"].replace("-",""),
              "format": d.get("format","JSON")}
    r = requests.get(f"{BASE}/{resolution}/point", params=params, timeout=90)
    r.raise_for_status()
    payload = r.json()
    try: values = payload["properties"]["parameter"]
    except KeyError as e: raise ValueError(f"Unexpected API response: {str(payload)[:500]}") from e
    frame = pd.DataFrame({k: pd.Series(v, dtype="float64") for k,v in values.items()})
    frame.index = pd.Index(list(next(iter(values.values())).keys()), name="timestamp_key")
    return payload, frame.reset_index()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--resolution", choices=["hourly","daily"], required=True); ap.add_argument("--config")
    a=ap.parse_args(); cfg=load_config(a.config); payload, df=fetch(a.resolution,cfg)
    out=project_path(cfg["paths"]["raw_dir"]); out.mkdir(parents=True,exist_ok=True)
    stamp=f'{cfg["data"]["start_date"]}_{cfg["data"]["end_date"]}'
    df.to_csv(out/f"nasa_power_{a.resolution}_{stamp}.csv",index=False)
    (out/f"nasa_power_{a.resolution}_{stamp}.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"Saved {len(df)} rows to {out}. Inspect units and missing values before modeling.")
if __name__=="__main__": main()
