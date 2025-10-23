
"""
main.py — Standalone main program to call functions from tps_core.py

Usage examples:
1) Single file:
   python tps_main.py --data ./data/sample1.xlsx --rho 1200 --Cp 1500 --h 200e-6 --r 2e-3 --P0 0.1 --t_end 5

2) Batch folder (process all csv/xlsx in ./data):
   python tps_main.py --data ./data --rho 1200 --Cp 1500 --h 200e-6 --r 2e-3 --P0 0.1 --t_end 5 --save-curves

Notes:
- Data format: either two columns [time, temperature], or two rows (first row=time, second row=temperature).
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from tps_core import Options, load_time_temperature, predict_lambda_grid

def is_data_file(p: Path) -> bool:
    return p.suffix.lower() in {".csv", ".xls", ".xlsx"}

def collect_files(data_path: Path):
    if data_path.is_file() and is_data_file(data_path):
        return [data_path]
    elif data_path.is_dir():
        return sorted([p for p in data_path.iterdir() if is_data_file(p)])
    else:
        raise FileNotFoundError(f"Invalid --data path: {data_path}")

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="TPS thermal conductivity fitting (main)")
    ap.add_argument("--data", type=str, required=True,
                    help="Path to a data file (.csv/.xlsx) or a folder containing data files.")
    # Physical params
    ap.add_argument("--rho", type=float, default=None, help="Density [kg/m^3]")
    ap.add_argument("--Cp", type=float, default=None, help="Specific heat [J/(kg·K)]")
    ap.add_argument("--Cv", type=float, default=None, help="Volumetric heat [J/(m^3·K)] (overrides rho*Cp)")
    ap.add_argument("--h", type=float, default=200e-6, help="Sample thickness [m]")
    ap.add_argument("--r", type=float, default=2e-3, help="Sensor radius [m]")
    ap.add_argument("--P0", type=float, default=0.10, help="Heating power [W]")
    ap.add_argument("--t_end", type=float, default=5.0, help="Heating time if time vector missing [s]")
    # Numerics
    ap.add_argument("--m", type=int, default=8, help="SLAB series level m")
    ap.add_argument("--plot", action="store_true", help="Plot is handled at package-level (not enabled here).")
    ap.add_argument("--lambda-min", type=float, default=0.1, help="Lower bound for lambda search [W/m·K]")
    # I/O
    ap.add_argument("--save-dir", type=str, default="./outputs", help="Directory to save summary and curves.")
    ap.add_argument("--save-curves", action="store_true", help="Save fitted curves for each file.")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    data_path = Path(args.data).resolve()
    files = collect_files(data_path)
    out_dir = Path(args.save_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build options template
    opts_template = Options(
        rho=args.rho, Cp=args.Cp, Cv=args.Cv,
        r=args.r, h=args.h, P0=args.P0, t_end=args.t_end,
        m=args.m, LAMBDA_MIN=args.lambda_min,
        plot=False, fit_save=args.save_curves
    )

    results = []
    for f in files:
        try:
            arr = load_time_temperature(f)
            # Always save per-file fitted curves in outputs folder
            try:
                t_last = float(arr[-1, 0])  # 第一列最后一行
                dT_last = float(arr[-1, 1])  # 第二列最后一行
            except Exception:
                t_last, dT_last = None, None

            # Always save per-file fitted curves in outputs folder
            save_csv = (out_dir / f"{f.stem}_fit.csv").as_posix()
            opts = opts_template
            opts.save_curve_csv = save_csv

            lam, report = predict_lambda_grid(arr, opts)
            results.append({
                "file": f.name,
                "lambda_raw_W_mK": report.get("lambda_raw", None),
                "lambda_hat_corrected_W_mK": lam,
                "err_rate_pred": report.get("err_rate_pred", None),
                "s3_min": report.get("s3_min", None),
                "r_m": opts.r,
                "h_m": opts.h,
                "P0_W": opts.P0,
                "t_end_s": opts.t_end,
                "Cv_J_m3K": opts.Cv_effective(),
                "t_last_s": t_last,
                "dT_last_K": dT_last
            })
            print(
                f"[OK] {f.name}: λ_raw={report.get('lambda_raw', float('nan')):.4f} → λ_hat={lam:.4f} W/(m·K), s3_min={report.get('s3_min', float('nan')):.6g}")

        except Exception as e:
            # 如果本文件拟合出错，不影响后续文件
            print(f"[FAIL] {f.name}: {e}")

    # Save summary
    if results:
        df = pd.DataFrame(results)
        summary_path = out_dir / "lambda_fit_summary.csv"
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"✅ Summary saved to: {summary_path}")
    else:
        print("No successful results to summarize.")

if __name__ == "__main__":
    main(sys.argv[1:])
