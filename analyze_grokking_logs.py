
"""
analyze_grokking_logs.py
------------------------
Compute "time-to-grok" from modular_grokking.py outputs.

Usage:
python analyze_grokking_logs.py --run_dir ./outputs/auto_mod97_transformer_wd0.001 --threshold 0.9

Inputs expected in --run_dir:
  - val_log.csv : columns [step, acc, loss]
Optional:
  - train_log.csv

Outputs written into --run_dir:
  - grokking_summary.json
  - grokking_table_row.tex  (LaTeX row snippet)
"""

import argparse, json
from pathlib import Path
import pandas as pd

def first_step_meeting_threshold(val_csv: Path, threshold: float):
    df = pd.read_csv(val_csv)
    df = df.sort_values("step")
    hit = df[df["acc"] >= threshold]
    if hit.empty:
        return None
    return int(hit.iloc[0]["step"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default=None, help="Optional: model name to include in table")
    ap.add_argument("--weight_decay", type=str, default=None, help="Optional: wd to include in table")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    val_csv = run_dir / "val_log.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing {val_csv}")
    step = first_step_meeting_threshold(val_csv, args.threshold)

    summary = {
        "run_dir": str(run_dir),
        "threshold": args.threshold,
        "time_to_grok_steps": step,
        "met_threshold": step is not None
    }

    # Try to infer model/wd from folder name if not provided
    name = run_dir.name
    model = args.model or ("transformer" if "transformer" in name else ("mlp" if "mlp" in name else "unknown"))
    wd = args.weight_decay or (name.split("_wd")[-1] if "_wd" in name else "NA")

    (run_dir / "grokking_summary.json").write_text(json.dumps(summary, indent=2))

    # LaTeX row
    if step is None:
        row = f"{model} & {wd} & -- & No grokking \\\\"
    else:
        row = f"{model} & {wd} & {step} & Clear late generalization \\\\"

    (run_dir / "grokking_table_row.tex").write_text(row)
    print("Wrote:", run_dir / "grokking_summary.json")
    print("Wrote:", run_dir / "grokking_table_row.tex")
    if step is None:
        print("No grokking (threshold not met).")
    else:
        print(f"Time-to-grok: {step} steps")

if __name__ == "__main__":
    main()
