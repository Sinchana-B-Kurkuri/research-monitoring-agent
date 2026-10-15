import glob
import pandas as pd
from pathlib import Path

files = glob.glob("data/labels/*.csv")
print(f"Total CSV files: {len(files)}")

sample_df = pd.read_parquet("data/evaluation_sample.parquet")
sample_pids = set(sample_df["paper_id"].astype(str))
print(f"Total sample pids: {len(sample_pids)}")

latest_df = pd.read_csv("data/labels/latest_labels.csv", dtype=str)
print(f"Latest labels rows: {len(latest_df)}")
print(f"Latest labels unique pids: {latest_df['paper_id'].nunique()}")

for idx, pid in enumerate(latest_df["paper_id"]):
    if pid not in sample_pids:
        print(f"Row {idx}: {pid} not in sample!")

all_records = []
for f in files:
    try:
        df = pd.read_csv(f, dtype=str)
        all_records.append((f, len(df)))
    except Exception as e:
        print(f"Error reading {f}: {e}")

print("Checked all files successfully.")
