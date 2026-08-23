from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OLSResult:
    coefficients: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    r_squared: float
    n_obs: int
    residuals: np.ndarray
    fitted: np.ndarray
    covariance_type: str
    hac_lags: int | None = None


@dataclass
class ADFResult:
    test_name: str
    regression: str
    used_lag: int
    n_obs: int
    test_stat: float
    critical_values: dict[str, float]
    note: str


def newey_west_covariance(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    xtx_inv = np.linalg.inv(X.T @ X)
    n_obs, n_params = X.shape
    S = np.zeros((n_params, n_params))

    for t in range(n_obs):
        xt = X[t : t + 1].T
        S += resid[t] ** 2 * (xt @ xt.T)

    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        Gamma = np.zeros((n_params, n_params))
        for t in range(lag, n_obs):
            xt = X[t : t + 1].T
            xl = X[t - lag : t - lag + 1].T
            Gamma += resid[t] * resid[t - lag] * (xt @ xl.T)
        S += weight * (Gamma + Gamma.T)

    return xtx_inv @ S @ xtx_inv


def fit_ols(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None) -> OLSResult:
    y_vec = y.to_numpy(dtype=float)
    X_mat = X.to_numpy(dtype=float)

    beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    fitted = X_mat @ beta
    resid = y_vec - fitted

    n_obs, n_params = X_mat.shape
    sse = float(resid.T @ resid)
    y_centered = y_vec - y_vec.mean()
    tss = float(y_centered.T @ y_centered)
    r_squared = 1 - sse / tss if tss > 0 else np.nan

    if hac_lags is None or hac_lags <= 0:
        sigma2 = sse / (n_obs - n_params)
        xtx_inv = np.linalg.inv(X_mat.T @ X_mat)
        cov = sigma2 * xtx_inv
        cov_type = "classic"
    else:
        cov = newey_west_covariance(X_mat, resid, hac_lags)
        cov_type = "HAC"

    std_err = np.sqrt(np.diag(cov))
    t_stats = beta / std_err

    return OLSResult(
        coefficients=pd.Series(beta, index=X.columns),
        std_errors=pd.Series(std_err, index=X.columns),
        t_stats=pd.Series(t_stats, index=X.columns),
        r_squared=r_squared,
        n_obs=n_obs,
        residuals=resid,
        fitted=fitted,
        covariance_type=cov_type,
        hac_lags=hac_lags,
    )


def default_hac_lags(n_obs: int) -> int:
    return max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))


def _build_adf_design(series: pd.Series, lags: int, regression: str) -> tuple[pd.Series, pd.DataFrame]:
    s = pd.Series(series).dropna().astype(float).reset_index(drop=True)
    ds = s.diff()

    y = ds.iloc[lags + 1 :].reset_index(drop=True)
    data: dict[str, np.ndarray] = {}
    if regression == "c":
        data["const"] = np.ones(len(y))

    data["lagged_level"] = s.shift(1).iloc[lags + 1 :].to_numpy()

    for i in range(1, lags + 1):
        data[f"diff_lag_{i}"] = ds.shift(i).iloc[lags + 1 :].to_numpy()

    X = pd.DataFrame(data).reset_index(drop=True)
    return y, X


def adf_test(
    series: pd.Series,
    regression: str = "c",
    max_lags: int | None = None,
    autolag: str = "bic",
    test_name: str = "ADF",
    critical_values: dict[str, float] | None = None,
    note: str = "",
) -> ADFResult:
    s = pd.Series(series).dropna().astype(float)
    n_obs = len(s)
    if max_lags is None:
        max_lags = int(np.floor(12 * (n_obs / 100) ** 0.25))

    best_lag = None
    best_score = np.inf
    best_result: OLSResult | None = None

    for lag in range(max_lags + 1):
        try:
            y, X = _build_adf_design(s, lag, regression)
            res = fit_ols(y, X)
            k = X.shape[1]
            n = res.n_obs
            sse = float(residual_sum_of_squares(res.residuals))
            if autolag.lower() == "aic":
                score = n * np.log(sse / n) + 2 * k
            else:
                score = n * np.log(sse / n) + k * np.log(n)
            if score < best_score:
                best_score = score
                best_lag = lag
                best_result = res
        except Exception:
            continue

    if best_result is None or best_lag is None:
        raise RuntimeError("ADF lag selection failed.")

    if critical_values is None:
        if regression == "c":
            critical_values = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
        else:
            critical_values = {"1%": -2.58, "5%": -1.95, "10%": -1.62}

    test_stat = float(best_result.t_stats["lagged_level"])
    return ADFResult(
        test_name=test_name,
        regression=regression,
        used_lag=best_lag,
        n_obs=best_result.n_obs,
        test_stat=test_stat,
        critical_values=critical_values,
        note=note,
    )


def residual_sum_of_squares(resid: np.ndarray) -> float:
    return float(resid.T @ resid)
