
"""
analyze_inr_classifier.py
-------------------------
Summarize INR classifier logs and produce LaTeX table rows.

Usage:
python analyze_inr_classifier.py --log ./inr_mnist/clf_log.csv --threshold 0.9

Outputs (in same folder as log):
  - inr_summary.json
  - inr_table_row.tex
"""

import argparse, json
from pathlib import Path
import pandas as pd

def epoch_meeting_threshold(csv_path: Path, threshold: float):
    df = pd.read_csv(csv_path)
    df = df.sort_values("epoch")
    hit = df[df["val_acc"] >= threshold]
    if hit.empty:
        return None
    return int(hit.iloc[0]["epoch"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True, help="Path to clf_log.csv")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--label", type=str, default="INR-Param-MLP", help="Label for table (e.g., classifier type)")
    args = ap.parse_args()

    log = Path(args.log)
    if not log.exists():
        raise FileNotFoundError(f"Missing {log}")
    df = pd.read_csv(log)
    best_val = float(df["val_acc"].max())
    last_val = float(df["val_acc"].iloc[-1])
    ep = epoch_meeting_threshold(log, args.threshold)

    summary = {
        "log": str(log),
        "threshold": args.threshold,
        "epoch_to_threshold": ep,
        "best_val_acc": best_val,
        "last_val_acc": last_val
    }
    out_json = log.parent / "inr_summary.json"
    out_tex = log.parent / "inr_table_row.tex"
    out_json.write_text(json.dumps(summary, indent=2))

    if ep is None:
        row = f"{args.label} & {best_val:.3f} & -- & No late generalization \\\\"
    else:
        row = f"{args.label} & {best_val:.3f} & {ep} & Late generalization observed \\\\"
    out_tex.write_text(row)

    print("Wrote:", out_json)
    print("Wrote:", out_tex)
    print(f"Best val acc: {best_val:.3f}")
    if ep is None:
        print("Did not meet threshold.")
    else:
        print(f"Epoch to reach threshold: {ep}")

if __name__ == "__main__":
    main()
