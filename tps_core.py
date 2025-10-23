
"""
tps_core.py — TPS thermal conductivity fitting core (functions package)

Features:
- Options dataclass with physical parameters (supports rho*Cp or Cv directly)
- Robust loader for time-temperature data from CSV/XLSX
- Three-stage grid search to estimate lambda (thermal conductivity)
- D_of_tau kernel identical in spirit to user's PredictFunction_V_1.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import iv
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from joblib import load
import json

# ----------------------------
# Dataclass for options
# ----------------------------

@dataclass
class Options:
    # Physical parameters
    rho: Optional[float] = None              # density [kg/m^3]
    Cp: Optional[float] = None               # specific heat [J/(kg·K)]
    Cv: Optional[float] = None               # volumetric heat [J/(m^3·K)]  (overrides rho*Cp if given)
    r: float = 2e-3                          # sensor radius [m]
    h: float = 200e-6                        # sample thickness [m]
    P0: float = 0.10                         # heating power [W]
    t_end: float = 5.0                       # heating time (if time vector missing) [s]

    # Numerical parameters
    m: int = 20
    SIGMA_MIN: float = 1e-3
    I_MAX: int = 150
    t0_idx: int = 20                         # baseline align index (1-based)
    LAMBDA_MIN: float = 0.1                  # lower bound for lambda grid

    # I/O
    plot: bool = False
    fit_save: bool = False
    save_curve_csv: Optional[str] = None
    save_curve_excel: Optional[str] = None
    save_encoding: str = "utf-8-sig"
    include_experiment: bool = True

    def Cv_effective(self) -> float:
        """Return effective volumetric heat capacity (Cv)."""
        if self.Cv is not None:
            return float(self.Cv)
        if self.rho is not None and self.Cp is not None:
            return float(self.rho) * float(self.Cp)
        # default fallback
        return 4.0e6


# ----------------------------
# Data loader
# ----------------------------

def load_time_temperature(file: Union[str, Path]) -> np.ndarray:
    """
    Load [t, T] data from .csv or .xlsx. Accepts either:
    - 2 columns (time, temperature), any number of rows
    - 2 rows (first row time, second row temperature), any number of columns
    Returns array shape (N, 2).
    """
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path, header=None)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path, header=None)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Convert to numeric and drop all-NaN rows/cols
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)

    # Case A: 2 columns
    if df.shape[1] >= 2 and df.shape[0] >= 2:
        # Prefer first two columns if they look valid
        arr = df.iloc[:, :2].to_numpy(dtype=float)
        # If more than 2 rows but exactly 2 rows is also allowed
        # No further change needed
    # Case B: 2 rows (first row time, second row temperature)
    elif df.shape[0] == 2 and df.shape[1] >= 2:
        arr = df.T.iloc[:, :2].to_numpy(dtype=float)  # transpose to N×2
    else:
        raise ValueError("Expect either 2 columns or 2 rows (time, temperature).")

    # Drop rows with NaNs and ensure order by time if time monotonic
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.size == 0:
        raise ValueError("No valid numeric data found in the file.")
    return arr


# ----------------------------
# Core kernel D(τ)
# ----------------------------

def D_of_tau(tau: np.ndarray, r: float, h: float, m: int,
             SIGMA_MIN: float, I_MAX: int) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    tau_finite = tau[np.isfinite(tau)]
    tau_max = np.max(tau_finite) if tau_finite.size else np.nan
    if not np.isfinite(tau_max) or tau_max <= 0:
        tau_max = max(1e-2, SIGMA_MIN * 200)

    S_lo = max(SIGMA_MIN, 1e-6)
    S_hi = max(tau_max, S_lo * 50.0)
    Ns = 1200

    sigma = np.logspace(np.log10(S_lo), np.log10(S_hi), Ns)
    sigma2 = sigma * sigma

    L = np.arange(1, m + 1, dtype=float)[:, None]
    K = np.arange(1, m + 1, dtype=float)[None, :]
    LK = L * K
    LmK2 = (L - K) ** 2
    IK = np.arange(1, I_MAX + 1, dtype=float)
    hr2 = (h / r) ** 2

    def safe_exp(x):
        return np.exp(np.clip(x, -745.0, 709.0))

    def I0_scaled(x):
        x = np.maximum(x, np.finfo(float).tiny)
        out = np.empty_like(x, dtype=float)
        mask = x <= 50.0
        out[mask] = np.exp(-x[mask]) * iv(0, x[mask])
        out[~mask] = 1.0 / np.sqrt(2.0 * np.pi * x[~mask])
        return out

    inv_4m2 = 1.0 / (4.0 * (m ** 2))
    inv_2m2 = 1.0 / (2.0 * (m ** 2))

    S1 = np.zeros(Ns)
    S2 = np.zeros(Ns)
    for i, s2 in enumerate(sigma2):
        exp_core = safe_exp(-LmK2 * (inv_4m2 / s2))
        barg = LK * (inv_2m2 / s2)
        I0s = I0_scaled(barg)
        S1[i] = np.sum(LK * exp_core * I0s)
        S2[i] = np.sum(safe_exp(-(IK ** 2) * (hr2 / s2)))

    integrand = (S1 * (1.0 + 2.0 * S2)) / np.maximum(sigma2, np.finfo(float).tiny)
    integrand[~np.isfinite(integrand)] = 0.0
    F = cumulative_trapezoid(integrand, sigma, initial=0.0)

    D = np.zeros_like(tau)
    mask = np.isfinite(tau) & (tau >= S_lo)
    if np.any(mask):
        interp = PchipInterpolator(sigma, F, extrapolate=True)
        D_val = interp(tau[mask])
        D[mask] = (1.0 / (m * (m + 1))) ** 2 * D_val
    return D


# ----------------------------
# Fitting (three-stage grid search)
# ----------------------------

def predict_lambda_grid(experiment_data, opts: Optional[Options] = None) -> Tuple[float, Dict]:
    """
    Three-stage grid search to estimate lambda:
      1) coarse:  step 1.0
      2) medium:  ±1.0 around best1, step 0.1
      3) fine:    ±0.1 around best2, step 0.01

    Input:
      experiment_data: (N,2) array-like with columns [t, T]

    Returns:
      lambda_hat, report dict (grids, s arrays, best curves, s3_min, etc.)
    """
    if opts is None:
        opts = Options()

    # Accept DataFrame/Series/array
    if isinstance(experiment_data, (pd.DataFrame, pd.Series)):
        X = experiment_data.to_numpy()
    else:
        X = np.asarray(experiment_data, dtype=float)

    assert X.ndim == 2 and X.shape[1] >= 2, "experiment_data must be N×2 [t, T]."

    t_exp = X[:, 0].astype(float)
    T_exp = X[:, 1].astype(float)

    # If time invalid, synthesize
    if not np.isfinite(t_exp).any() or np.allclose(t_exp, t_exp[0]):
        t_exp = np.linspace(opts.t_end/200, opts.t_end, 200)

    # Clip to at most 200 points
    N = min(200, t_exp.size)
    t = t_exp[:N].copy()
    Te = T_exp[:N].copy()

    # Baseline align
    t0 = int(np.clip(opts.t0_idx, 1, N)) - 1
    Te = Te - Te[t0]

    Cv_eff = float(opts.Cv_effective())

    def arange_clamped(center, halfspan, step, lo_min, ndigits=10):
        lo = max(center - halfspan, lo_min)
        hi = center + halfspan
        n = int(np.floor((hi - lo) / step)) + 1
        grid = lo + np.arange(n, dtype=float) * step
        return np.round(grid, ndigits)

    def s_of_lambda(lmbd: float) -> Tuple[float, np.ndarray]:
        alpha = lmbd / Cv_eff
        tau = np.sqrt(np.maximum(alpha * t, 0.0) / (opts.r ** 2))
        D = D_of_tau(tau, opts.r, opts.h, opts.m, opts.SIGMA_MIN, opts.I_MAX)
        coeff = opts.P0 / (np.pi ** 1.5 * opts.r * lmbd)
        Tc_full = coeff * D
        Tc = Tc_full - Tc_full[t0]

        # Normalized SSE from index 20 onwards (as in user's script)
        Te_sub = Te[20:] if Te.size > 20 else Te
        Tc_sub = Tc[20:] if Tc.size > 20 else Tc
        diff = Te_sub - Tc_sub
        # s = sqrt( sum(diff^2) / sum(Te_sub^2) )
        den = np.sum(Te_sub * Te_sub)
        s = np.sqrt(np.sum(diff * diff) / den) if den > 0 else np.inf
        return s, Tc

    # Stage 1
    grid1 = np.arange(opts.LAMBDA_MIN, opts.LAMBDA_MIN + 20.0, 1.0, dtype=float)
    s1 = np.empty_like(grid1)
    for i, lmbd in enumerate(grid1):
        s1[i], _ = s_of_lambda(lmbd)
    best1 = float(grid1[np.argmin(s1)])

    # Stage 2
    grid2 = arange_clamped(best1, halfspan=1.0, step=0.1, lo_min=opts.LAMBDA_MIN)
    s2 = np.empty_like(grid2)
    for i, lmbd in enumerate(grid2):
        s2[i], _ = s_of_lambda(lmbd)
    best2 = float(grid2[np.argmin(s2)])

    # Stage 3
    grid3 = arange_clamped(best2, halfspan=0.1, step=0.01, lo_min=opts.LAMBDA_MIN)
    s3 = np.empty_like(grid3)
    Tc_best = None
    for i, lmbd in enumerate(grid3):
        s3[i], Tc_tmp = s_of_lambda(lmbd)
        if i == 0 or s3[i] < np.min(s3[:i]):
            Tc_best = Tc_tmp

    lambda_hat = float(grid3[np.argmin(s3)])
    s3_min = float(np.min(s3))

    report = {
        "grid1": grid1, "s1": s1, "best1": best1,
        "grid2": grid2, "s2": s2, "best2": best2,
        "grid3": grid3, "s3": s3, "s3_min": s3_min, "best3": lambda_hat,
        "t": t, "Te": Te, "Tc": Tc_best
    }

    # Optional save
    need_save = bool(opts.save_curve_csv or opts.save_curve_excel or opts.fit_save)
    if need_save:
        data = {"t": t, "Tc": Tc_best}
        if opts.include_experiment:
            data["Te"] = Te
        df_save = pd.DataFrame(data)
        if opts.save_curve_csv:
            df_save.to_csv(opts.save_curve_csv, index=False, encoding=opts.save_encoding)
            report["saved_curve_csv"] = opts.save_curve_csv
        if opts.save_curve_excel:
            df_save.to_excel(opts.save_curve_excel, index=False)
            report["saved_curve_excel"] = opts.save_curve_excel
        if opts.fit_save and not (opts.save_curve_csv or opts.save_curve_excel):
            fname = f"fit_curve_lambda_{lambda_hat:.4f}.csv"
            df_save.to_csv(fname, index=False, encoding=opts.save_encoding)
            report["saved_curve_csv"] = fname


    # 加载模型（若缺失则仅返回原始 lambda）
    err_rate_pred = None
    lambda_hat_corrected = float(lambda_hat)
    try:
        with open("model/lambda_calibration_artifacts.json", "r") as f:
            artifacts = json.load(f)
        error_model = load("model/lambda_err_model.joblib")

        # 提取当前拟合的输入参数（CV, h, r, P0, t_end等）
        features = pd.DataFrame([{
            "Cv": opts.Cv_effective(),
            "h": opts.h,
            "r": opts.r,
            "P0": opts.P0,
            "t_end": opts.t_end,
            "lambda_cal": lambda_hat
        }])

        # 模型预测误差率（Error_rate = (lambda_true - lambda_cal)/lambda_true）
        err_rate_pred = float(error_model.predict(features)[0])
        # 修正后的 λ_hat
        lambda_hat_corrected = float(lambda_hat / (1.0 + err_rate_pred))
    except Exception:
        # 若模型或文件不可用，保持原值并让 err_rate_pred 为空
        err_rate_pred = None
        lambda_hat_corrected = float(lambda_hat)

    # 在报告中追加关键结果
    report.update({
        "lambda_raw": float(lambda_hat),
        "lambda_corrected": float(lambda_hat_corrected),
        "err_rate_pred": err_rate_pred,
        "s3_min": s3_min
    })

    return lambda_hat_corrected, report




