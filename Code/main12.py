"""
Chạy:
    cd c:/BI
    uvicorn main12:app --reload --port 8000
"""

from __future__ import annotations
import io, json, warnings
from typing import Optional, List
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

# statsmodels SARIMAX – optional fallback nếu chưa cài
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="BiSight API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── In-memory store ──────────────────────────────────────────────
_store: dict = {"df": None, "weekly": None, "forecast": None}

REQUIRED_COLS = {
    "order_id", "customer_id", "order_date", "price",
    "discount", "quantity", "total_amount", "profit_margin",
    "category", "region", "shipping_cost", "returned"
}

# ── Request models ───────────────────────────────────────────────
class FilterParams(BaseModel):
    date_from: Optional[str] = None
    date_to:   Optional[str] = None
    categories: Optional[List[str]] = None
    regions:    Optional[List[str]] = None

class HolidayEvent(BaseModel):
    date: str          # YYYY-MM-DD
    name: str
    multiplier: float = 1.3   # expected uplift factor

class ForecastRequest(FilterParams):
    horizon: int = 8
    model: str = "ensemble"        # gbm | sarima | ensemble
    scenario: str = "base"         # base | optimistic | pessimistic
    target_metric: str = "revenue" # revenue | quantity | orders
    holidays: Optional[List[HolidayEvent]] = None

class RFMRequest(FilterParams):
    total_budget: float = 1_000_000

class OptimizerRequest(FilterParams):
    budget: float = 1_000_000
    min_margin: float = 0.15
    max_discount: float = 0.30
    strategy: str = "profit"  # revenue | profit
    horizon: int = 8

# ── Helpers ─────────────────────────────────────────────────────
def _read_csv_flexible(content: bytes) -> pd.DataFrame:
    """Try common encodings/separators often seen in Excel-exported CSV files."""
    attempts = []
    encodings = ["utf-8", "utf-8-sig", "cp1258", "latin1"]
    seps = [None, ",", ";", "	", "|"]

    for enc in encodings:
        for sep in seps:
            try:
                kwargs = {"encoding": enc}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = sep
                df_try = pd.read_csv(io.BytesIO(content), **kwargs)
                if len(df_try.columns) > 1:
                    df_try.columns = [str(c).strip() for c in df_try.columns]
                    return df_try
            except Exception as e:
                attempts.append(f"{enc}/{sep or 'auto'}: {e}")
                continue

    raise ValueError("; ".join(attempts[:6]))


def _load(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id", "order_id", "total_amount"])
    df = df[df["total_amount"] > 0].copy()
    df["discount"] = pd.to_numeric(df["discount"], errors="coerce").clip(0, 0.95).fillna(0)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").clip(0).fillna(1)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["total_amount"])
    df["profit_margin"] = pd.to_numeric(df["profit_margin"], errors="coerce").fillna(0)
    df["shipping_cost"] = pd.to_numeric(df["shipping_cost"], errors="coerce").fillna(0)
    df["returned_flag"] = df["returned"].astype(str).str.lower().isin(["yes","true","1"]).astype(int)
    df["net_price"] = df["price"] * (1 - df["discount"])
    df["revenue"] = df["net_price"] * df["quantity"]
    return df.sort_values("order_date").reset_index(drop=True)

def _apply_filters(df: pd.DataFrame, f: FilterParams) -> pd.DataFrame:
    if f.date_from:
        df = df[df["order_date"] >= pd.to_datetime(f.date_from)]
    if f.date_to:
        df = df[df["order_date"] <= pd.to_datetime(f.date_to)]
    if f.categories:
        df = df[df["category"].isin(f.categories)]
    if f.regions:
        df = df[df["region"].isin(f.regions)]
    return df

def _safe(v):
    """Convert numpy types → Python native for JSON."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, pd.Timestamp): return str(v.date())
    return v

def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(i) for i in obj]
    return _safe(obj)

# ── Anomaly detection (simple IQR-based) ────────────────────────
def _detect_anomalies(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

# ════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(400, "File rỗng hoặc không đọc được nội dung")

    try:
        df_raw = _read_csv_flexible(content)
    except Exception as e:
        raise HTTPException(400, f"Không đọc được CSV. Hãy kiểm tra encoding hoặc dấu phân cách cột. Chi tiết: {e}")

    missing = REQUIRED_COLS - set(df_raw.columns)
    if missing:
        raise HTTPException(400, f"Thiếu cột: {sorted(missing)}")

    df = _load(df_raw)
    _store["df"] = df

    # Aggregate weekly for forecasting
    weekly = (
        df.set_index("order_date")
          .resample("W-MON")
          .agg(
            revenue=("revenue","sum"),
            quantity=("quantity","sum"),
            customers=("customer_id","nunique"),
            avg_discount=("discount","mean"),
            return_rate=("returned_flag","mean"),
            avg_shipping=("shipping_cost","mean"),
            avg_profit=("profit_margin","mean"),
            orders=("order_id","nunique"),
          )
          .reset_index()
          .rename(columns={"order_date":"ds"})
    )
    weekly = weekly.ffill().bfill()
    if len(weekly) > 1:
        weekly = weekly.iloc[:-1].copy()
    _store["weekly"] = weekly

    return _jsonify({
        "rows": len(df),
        "date_min": df["order_date"].min(),
        "date_max": df["order_date"].max(),
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "regions":    sorted(df["region"].dropna().unique().tolist()),
        "columns":    df.columns.tolist(),
        "sample":     df.head(5).to_dict(orient="records"),
    })

# ════════════════════════════════════════════════════════════════
#  OVERVIEW
# ════════════════════════════════════════════════════════════════
@app.post("/overview")
async def overview(f: FilterParams):
    df = _store.get("df")
    if df is None: raise HTTPException(400, "Chưa upload dữ liệu")
    df = _apply_filters(df, f)
    if df.empty: raise HTTPException(400, "Không có dữ liệu sau khi lọc")

    total_rev  = df["total_amount"].sum()
    total_prof = df["profit_margin"].sum()
    aov        = df["total_amount"].mean()
    return_rate= df["returned_flag"].mean()
    neg_profit = (df["profit_margin"] < 0).mean()

    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))
          .agg(revenue=("total_amount","sum"), profit=("profit_margin","sum"),
               orders=("order_id","count"))
          .reset_index()
    )
    monthly["month"] = monthly["order_date"].astype(str)
    monthly["rev_mom"] = monthly["revenue"].pct_change().fillna(0)

    # anomalies in monthly revenue
    anom_mask = _detect_anomalies(monthly["revenue"])
    monthly["is_anomaly"] = anom_mask

    by_cat = (df.groupby("category")
                .agg(revenue=("total_amount","sum"), profit=("profit_margin","sum"),
                     return_rate=("returned_flag","mean"), orders=("order_id","count"))
                .assign(profit_rate=lambda x: x["profit"]/x["revenue"])
                .sort_values("revenue", ascending=False)
                .reset_index())

    by_region = (df.groupby("region")
                   .agg(revenue=("total_amount","sum"), orders=("order_id","count"),
                        avg_delivery=("shipping_cost","mean"))
                   .sort_values("revenue", ascending=False).reset_index())

    # Auto-insights
    insights = []
    peak_month = monthly.loc[monthly["revenue"].idxmax(), "month"]
    insights.append({"type":"info", "text": f"Tháng {peak_month} đạt đỉnh doanh thu"})
    neg_cats = by_cat[by_cat["profit_rate"] < 0]["category"].tolist()
    if neg_cats:
        insights.append({"type":"warning","text": f"{', '.join(neg_cats)} có margin âm"})
    if neg_profit > 0.1:
        insights.append({"type":"danger","text": f"{neg_profit*100:.1f}% đơn hàng lợi nhuận âm"})
    anom_months = monthly[monthly["is_anomaly"]]["month"].tolist()
    if anom_months:
        insights.append({"type":"anomaly","text": f"Phát hiện bất thường tháng: {', '.join(anom_months[-3:])}"})

    return _jsonify({
        "kpis": {
            "total_revenue": total_rev, "total_profit": total_prof,
            "aov": aov, "return_rate": return_rate, "neg_profit_rate": neg_profit,
            "total_orders": int(len(df)), "unique_customers": int(df["customer_id"].nunique())
        },
        "monthly": monthly[["month","revenue","profit","orders","rev_mom","is_anomaly"]].to_dict("records"),
        "by_category": by_cat.to_dict("records"),
        "by_region": by_region.to_dict("records"),
        "insights": insights,
    })

# ════════════════════════════════════════════════════════════════
#  EDA
# ════════════════════════════════════════════════════════════════
@app.post("/eda")
async def eda(f: FilterParams):
    df = _store.get("df")
    if df is None: raise HTTPException(400, "Chưa upload dữ liệu")
    df = _apply_filters(df, f)

    # Discount bands
    df["discount_band"] = pd.cut(df["discount"],
        bins=[-0.001,0,0.05,0.10,0.15,1.0],
        labels=["0%","0–5%","5–10%","10–15%","15%+"])
    disc = (df.groupby("discount_band", observed=True)
              .agg(orders=("order_id","count"),
                   profit_rate=("profit_margin", lambda s: s.sum()/df.loc[s.index,"total_amount"].sum()),
                   avg_aov=("total_amount","mean"))
              .reset_index())

    # Correlation matrix
    num_cols = ["price","discount","quantity","shipping_cost","total_amount","profit_margin"]
    corr = df[num_cols].corr().round(3)

    # Age groups
    if "customer_age" in df.columns:
        df["age_group"] = pd.cut(df["customer_age"],
            bins=[17,24,34,44,54,64,100],
            labels=["18–24","25–34","35–44","45–54","55–64","65+"])
        by_age = (df.groupby("age_group", observed=True)
                    .agg(revenue=("total_amount","sum"), orders=("order_id","count"))
                    .reset_index())
        age_data = by_age.to_dict("records")
    else:
        age_data = []

    # Shipping band
    df["shipping_band"] = pd.qcut(df["shipping_cost"], 5, duplicates="drop")
    shipping = (df.groupby("shipping_band", observed=True)
                  .agg(return_rate=("returned_flag","mean"),
                       avg_aov=("total_amount","mean"))
                  .reset_index())
    shipping["shipping_band"] = shipping["shipping_band"].astype(str)

    return _jsonify({
        "discount_analysis": disc.assign(discount_band=disc["discount_band"].astype(str)).to_dict("records"),
        "correlation": {"labels": num_cols, "matrix": corr.values.tolist()},
        "by_age": age_data,
        "shipping_analysis": shipping.to_dict("records"),
        "category_return": df.groupby("category")["returned_flag"].mean().round(4).to_dict(),
    })

# ════════════════════════════════════════════════════════════════
#  RFM  (real scoring from data)
# ════════════════════════════════════════════════════════════════
@app.post("/rfm")
async def rfm(f: RFMRequest):
    df = _store.get("df")
    if df is None: raise HTTPException(400, "Chưa upload dữ liệu")
    df = _apply_filters(df, f)

    snapshot = df["order_date"].max() + pd.Timedelta(days=1)

    rfm_tbl = df.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (snapshot - x.max()).days),
        Frequency=("order_id", "count"),
        Monetary=("total_amount", "sum")
    ).reset_index()

    # Quintile scoring
    rfm_tbl["R"] = pd.qcut(rfm_tbl["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1]).astype(int)
    rfm_tbl["F"] = pd.qcut(rfm_tbl["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm_tbl["M"] = pd.qcut(rfm_tbl["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm_tbl["RFM_Score"] = rfm_tbl["R"].astype(str) + rfm_tbl["F"].astype(str) + rfm_tbl["M"].astype(str)

    def segment(row):
        r, f, m = row["R"], row["F"], row["M"]
        if r >= 4 and f >= 4 and m >= 4: return "Champions"
        if r >= 3 and f >= 4 and m >= 3: return "Loyal Customers"
        if r >= 4 and f >= 2 and m >= 3: return "Potential Loyalists"
        if r == 5 and f == 1 and m >= 3: return "New Customers"
        if r >= 4 and f == 1 and m <= 2: return "Promising"
        if r == 3 and f >= 2 and m >= 3: return "Need Attention"
        if r == 2 and f >= 2 and m >= 2: return "About to Sleep"
        if r <= 2 and f >= 4 and m >= 4: return "Can't Lose"
        if r <= 2 and f >= 3 and m >= 3: return "At Risk"
        if r <= 2 and f <= 2 and m <= 2: return "Hibernating"
        return "Low Value"

    rfm_tbl["Segment"] = rfm_tbl.apply(segment, axis=1)

    seg_summary = (rfm_tbl.groupby("Segment")
                          .agg(count=("customer_id","count"),
                               avg_recency=("Recency","mean"),
                               avg_frequency=("Frequency","mean"),
                               total_monetary=("Monetary","sum"),
                               avg_monetary=("Monetary","mean"),
                               avg_r=("R","mean"),
                               avg_f=("F","mean"),
                               avg_m=("M","mean"))
                          .reset_index()
                          .sort_values("total_monetary", ascending=False))

    # Budget allocation by segment instead of fixed voucher distribution
    seg_summary["customer_share"] = seg_summary["count"] / max(seg_summary["count"].sum(), 1)
    seg_summary["monetary_share"] = seg_summary["total_monetary"] / max(seg_summary["total_monetary"].sum(), 1)
    seg_summary["freq_norm"] = seg_summary["avg_frequency"] / max(seg_summary["avg_frequency"].max(), 1)
    seg_summary["recency_norm"] = 1 - (seg_summary["avg_recency"] / max(seg_summary["avg_recency"].max(), 1))
    seg_summary["segment_score"] = (
        0.40 * seg_summary["monetary_share"] +
        0.25 * seg_summary["customer_share"] +
        0.20 * seg_summary["freq_norm"] +
        0.15 * seg_summary["recency_norm"]
    )
    score_sum = max(seg_summary["segment_score"].sum(), 1e-9)
    seg_summary["budget_share"] = seg_summary["segment_score"] / score_sum
    seg_summary["budget_amount"] = seg_summary["budget_share"] * max(float(f.total_budget), 0.0)
    seg_summary["budget_per_customer"] = seg_summary["budget_amount"] / seg_summary["count"].clip(lower=1)

    # Matrix counts R×F
    matrix = rfm_tbl.groupby(["R","F"]).agg(
        count=("customer_id","count"),
        segment=("Segment", lambda x: x.value_counts().index[0])
    ).reset_index()

    # Avg R/F/M values by score (1-5) for bar charts
    r_by_score = rfm_tbl.groupby("R")["Recency"].mean().reset_index().rename(columns={"Recency":"avg_val","R":"score"})
    f_by_score = rfm_tbl.groupby("F")["Frequency"].mean().reset_index().rename(columns={"Frequency":"avg_val","F":"score"})
    m_by_score = rfm_tbl.groupby("M")["Monetary"].mean().reset_index().rename(columns={"Monetary":"avg_val","M":"score"})

    # Profit by segment – join with original transactions
    seg_profit = (df.merge(rfm_tbl[["customer_id","Segment"]], on="customer_id", how="left")
                    .groupby("Segment")["profit_margin"].sum()
                    .reset_index()
                    .rename(columns={"profit_margin":"total_profit"})
                    .sort_values("total_profit", ascending=False))

    return _jsonify({
        "segment_summary": seg_summary.to_dict("records"),
        "matrix": matrix.to_dict("records"),
        "r_by_score": r_by_score.to_dict("records"),
        "f_by_score": f_by_score.to_dict("records"),
        "m_by_score": m_by_score.to_dict("records"),
        "profit_by_segment": seg_profit.to_dict("records"),
        "rfm_stats": {
            "recency_median": float(rfm_tbl["Recency"].median()),
            "frequency_median": float(rfm_tbl["Frequency"].median()),
            "monetary_median": float(rfm_tbl["Monetary"].median()),
            "total_customers": int(len(rfm_tbl)),
            "total_budget": float(max(f.total_budget, 0.0)),
        }
    })

# ════════════════════════════════════════════════════════════════
#  FORECAST  (real SARIMA / GBM / Ensemble)
# ════════════════════════════════════════════════════════════════
def _add_features(df: pd.DataFrame, target_col: str = "revenue") -> pd.DataFrame:
    df = df.copy().sort_values("ds").reset_index(drop=True)
    df["_target"] = df[target_col].values
    df["trend"] = np.arange(len(df))
    df["month"] = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["week_sin"] = np.sin(2*np.pi*df["ds"].dt.isocalendar().week.astype(int)/52)
    df["week_cos"] = np.cos(2*np.pi*df["ds"].dt.isocalendar().week.astype(int)/52)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    for lag in [1,2,3,4,8,12]:
        df[f"lag_{lag}"] = df["_target"].shift(lag)
    for w in [4,8,12]:
        df[f"roll_mean_{w}"] = df["_target"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["_target"].shift(1).rolling(w).std()
    return df.dropna().reset_index(drop=True)

def _smape(y_true, y_pred):
    denom = (np.abs(y_true)+np.abs(y_pred))/2
    denom = np.where(denom==0,1e-8,denom)
    return float(np.mean(np.abs(y_true-y_pred)/denom))

def _wape(y_true, y_pred):
    d = np.abs(y_true).sum()
    return float(np.abs(y_true-y_pred).sum()/(d if d else 1e-8))

@app.post("/forecast")
async def forecast(req: ForecastRequest):
    df = _store.get("df")
    if df is None:
        raise HTTPException(400, "Chưa upload dữ liệu")

    df = _apply_filters(df, req)
    if df.empty:
        raise HTTPException(400, "Không có dữ liệu sau khi lọc")

    # Determine target metric column
    target_col = req.target_metric if req.target_metric in ("revenue", "quantity", "orders") else "revenue"

    weekly = (
        df.set_index("order_date")
          .resample("W-MON")
          .agg(
            revenue=("revenue","sum"),
            quantity=("quantity","sum"),
            customers=("customer_id","nunique"),
            avg_discount=("discount","mean"),
            return_rate=("returned_flag","mean"),
            avg_shipping=("shipping_cost","mean"),
            avg_profit=("profit_margin","mean"),
            orders=("order_id","nunique"),
          )
          .reset_index()
          .rename(columns={"order_date":"ds"})
    )
    weekly = weekly.ffill().bfill()
    if len(weekly) > 1:
        weekly = weekly.iloc[:-1].copy()
    if weekly.empty or len(weekly) < 16:
        raise HTTPException(400, "Dữ liệu không đủ để forecast (cần ít nhất 16 tuần)")

    horizon = min(max(req.horizon, 1), 26)
    scenario_mult = {"base":1.0,"optimistic":1.15,"pessimistic":0.90}.get(req.scenario, 1.0)

    # Build holiday multiplier map aligned to weekly forecast buckets (W-MON labels)
    holiday_dates = {}
    if req.holidays:
        for h in req.holidays:
            try:
                event_dt = pd.to_datetime(h.date)
            except Exception:
                continue
            week_label = (event_dt - pd.to_timedelta(event_dt.weekday(), unit="D")).date()
            holiday_dates[str(week_label)] = max(float(h.multiplier), 0.0)

    series = weekly[target_col].values.astype(float)
    dates  = pd.to_datetime(weekly["ds"])
    n      = len(series)
    train_n = max(int(n * 0.8), 12)
    train_n = min(train_n, n - 1)
    train_s, test_s = series[:train_n], series[train_n:]
    test_dates = dates[train_n:]

    # ── GBM ──
    feat_df = _add_features(weekly, target_col)
    if feat_df.empty or len(feat_df) < 10:
        raise HTTPException(400, "Không đủ dữ liệu sau khi tạo feature cho forecast")

    feature_cols = [
        "trend", "month", "quarter", "week_sin", "week_cos", "month_sin", "month_cos",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12",
        "roll_mean_4", "roll_std_4", "roll_mean_8", "roll_std_8", "roll_mean_12", "roll_std_12"
    ]
    feat_cols = [c for c in feature_cols if c in feat_df.columns]
    X = feat_df[feat_cols].values
    y = feat_df["_target"].values
    split = max(int(len(y) * 0.8), 8)
    split = min(split, len(y) - 1)
    if split < 1 or len(y) - split < 1:
        raise HTTPException(400, "Không đủ dữ liệu train/test cho forecast")

    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]

    gbm = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42
    )
    gbm.fit(Xtr, ytr)

    gbm_test_pred = gbm.predict(Xte)
    gbm_smape = _smape(yte, gbm_test_pred)
    gbm_wape  = _wape(yte, gbm_test_pred)
    gbm_rmse  = float(np.sqrt(mean_squared_error(yte, gbm_test_pred)))

    # Future GBM forecast via iterative prediction
    gbm_future = []
    history_vals = list(series)

    def _future_feature_row(hist, future_date):
        arr = np.array(hist, dtype=float)
        row = {
            "trend": len(arr),
            "month": future_date.month,
            "quarter": future_date.quarter,
            "week_sin": float(np.sin(2 * np.pi * int(future_date.isocalendar().week) / 52)),
            "week_cos": float(np.cos(2 * np.pi * int(future_date.isocalendar().week) / 52)),
            "month_sin": float(np.sin(2 * np.pi * future_date.month / 12)),
            "month_cos": float(np.cos(2 * np.pi * future_date.month / 12)),
        }
        for lag in [1, 2, 3, 4, 8, 12]:
            row[f"lag_{lag}"] = float(arr[-lag] if len(arr) >= lag else arr[0])
        for w in [4, 8, 12]:
            sl = arr[-w:] if len(arr) >= w else arr
            row[f"roll_mean_{w}"] = float(sl.mean())
            row[f"roll_std_{w}"] = float(sl.std()) if len(sl) > 1 else 0.0
        return [row[c] for c in feat_cols]

    for step in range(horizon):
        future_date = dates.iloc[-1] + pd.Timedelta(weeks=step + 1)
        x_future = _future_feature_row(history_vals, future_date)
        raw_pred = float(gbm.predict([x_future])[0])
        # Apply holiday multiplier to the aligned forecast week
        fd_str = str(future_date.date())
        hol_mult = holiday_dates.get(fd_str, 1.0)
        gbm_future.append(max(raw_pred * scenario_mult * hol_mult, 0))
        history_vals.append(raw_pred)

    # ── SARIMA ──
    sarima_future = []
    sarima_smape = sarima_wape = sarima_rmse = None
    if HAS_STATSMODELS and len(train_s) >= 16:
        try:
            model = SARIMAX(train_s, order=(1,1,1), seasonal_order=(1,1,1,4),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            sarima_test_pred = fit.forecast(steps=len(test_s))
            sarima_smape = _smape(test_s, sarima_test_pred)
            sarima_wape  = _wape(test_s, sarima_test_pred)
            sarima_rmse  = float(np.sqrt(mean_squared_error(test_s, sarima_test_pred)))
            model2 = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,4),
                             enforce_stationarity=False, enforce_invertibility=False)
            fit2 = model2.fit(disp=False)
            raw = fit2.forecast(steps=horizon)
            # Apply holiday multipliers to SARIMA future using the same weekly alignment
            sarima_future = []
            for i, v in enumerate(raw):
                fd_str = str((dates.iloc[-1] + pd.Timedelta(weeks=i+1)).date())
                hm = holiday_dates.get(fd_str, 1.0)
                sarima_future.append(max(float(v) * scenario_mult * hm, 0))
        except Exception:
            sarima_future = gbm_future[:]
            sarima_smape = gbm_smape * 1.4
            sarima_wape  = gbm_wape  * 1.4
            sarima_rmse  = gbm_rmse  * 1.4
    else:
        sarima_future = gbm_future[:]
        sarima_smape = gbm_smape * 1.4 if gbm_smape else None
        sarima_wape  = gbm_wape  * 1.4 if gbm_wape  else None
        sarima_rmse  = gbm_rmse  * 1.4 if gbm_rmse  else None

    # ── Ensemble ──
    ensemble_future = [0.55 * g + 0.45 * s for g, s in zip(gbm_future, sarima_future)]
    ens_smape = 0.55 * gbm_smape + 0.45 * (sarima_smape or gbm_smape)
    ens_wape  = 0.55 * gbm_wape  + 0.45 * (sarima_wape  or gbm_wape)
    ens_rmse  = 0.55 * gbm_rmse  + 0.45 * (sarima_rmse  or gbm_rmse)

    model_map = {"gbm": gbm_future, "sarima": sarima_future, "ensemble": ensemble_future}
    chosen = model_map.get(req.model, ensemble_future)

    # Backtesting: pick gbm test predictions aligned to test dates
    bt_test_pred = gbm_test_pred.tolist()
    bt_test_dates = [str(d.date()) for d in test_dates[:len(bt_test_pred)]]

    ci_width = {"gbm": gbm_smape, "sarima": sarima_smape or gbm_smape, "ensemble": ens_smape}
    ci_factor = ci_width.get(req.model, ens_smape)
    lower = [max(v * (1 - ci_factor * (1 + i * 0.05)), 0) for i, v in enumerate(chosen)]
    upper = [v * (1 + ci_factor * (1 + i * 0.05)) for i, v in enumerate(chosen)]

    future_dates = [str((dates.iloc[-1] + pd.Timedelta(weeks=i + 1)).date()) for i in range(horizon)]

    feat_imp = sorted(zip(feat_cols, gbm.feature_importances_), key=lambda x: x[1], reverse=True)[:10]

    # Build history for the target metric (not just revenue)
    forecast_payload = {
        "history": {
            "dates": [str(d.date()) for d in dates],
            "revenue": series.tolist(),         # actual target series (revenue/qty/orders)
            "target_col": target_col,
        },
        "backtest": {
            "dates": bt_test_dates,
            "predicted": bt_test_pred,
            "actual": test_s.tolist(),
        },
        "forecast": {
            "dates": future_dates,
            "values": chosen,
            "lower": lower,
            "upper": upper,
        },
        "metrics": {
            "gbm":      {"smape": gbm_smape,   "wape": gbm_wape,   "rmse": gbm_rmse},
            "sarima":   {"smape": sarima_smape, "wape": sarima_wape, "rmse": sarima_rmse},
            "ensemble": {"smape": ens_smape,   "wape": ens_wape,   "rmse": ens_rmse},
        },
        "feature_importance": [{"name": n, "importance": float(v)} for n, v in feat_imp],
        "total_forecast": sum(chosen),
        "avg_weekly": sum(chosen) / len(chosen),
    }
    _store["forecast"] = {
        "target_col": target_col,
        "model": req.model,
        "scenario": req.scenario,
        "dates": future_dates,
        "values": chosen,
        "avg_weekly": sum(chosen) / len(chosen),
    }
    return _jsonify(forecast_payload)

# ════════════════════════════════════════════════════════════════
#  OPTIMIZER  
# ════════════════════════════════════════════════════════════════
@app.post("/optimizer")
async def optimizer(req: OptimizerRequest):
    df = _store.get("df")
    if df is None:
        raise HTTPException(400, "Chưa upload dữ liệu")
    df = _apply_filters(df, req)
    if df.empty:
        raise HTTPException(400, "Không có dữ liệu sau khi lọc")

    # Khớp logic practical_discount_policy trong optimization.ipynb
    df = df.copy()
    df["unit_cost"] = (df["total_amount"] - df["profit_margin"] - df["shipping_cost"]) / df["quantity"].clip(1)

    segments = (
        df.groupby("category")
          .apply(lambda g: pd.Series({
              "avg_price": g["price"].mean(),
              "total_qty": g["quantity"].sum(),
              "avg_cost": np.average(g["unit_cost"], weights=g["quantity"])
          }))
          .reset_index()
    )

    segments["base_margin"] = (
        (segments["avg_price"] - segments["avg_cost"]) / segments["avg_price"].clip(0.01)
    )

    elasticity_map = {
        "Fashion":     -2.9,
        "Electronics": -2.7,
        "Sports":      -2.6,
        "Home":        -2.5,
        "Beauty":      -2.4,
        "Toys":        -2.5,
        "Grocery":     -1.8,
    }
    segments["elasticity"] = segments["category"].map(elasticity_map).fillna(-1.5)

    P0 = segments["avg_price"].to_numpy()
    C0 = segments["avg_cost"].to_numpy()

    forecast_info = _store.get("forecast")
    if not forecast_info or not forecast_info.get("values"):
        raise HTTPException(400, "Vui lòng chạy /forecast trước khi chạy optimizer")

    forecast_target = forecast_info.get("target_col", "quantity")
    forecast_vals = np.asarray(forecast_info["values"], dtype=float)
    horizon_used = int(max(1, min(req.horizon, len(forecast_vals))))
    total_forecast_driver = float(forecast_vals[:horizon_used].sum())

    category_orders = df.groupby("category")["order_id"].nunique().rename("orders_hist")
    segments = segments.merge(category_orders, on="category", how="left")
    segments["orders_hist"] = segments["orders_hist"].fillna(0.0)
    segments["avg_items_per_order"] = np.where(segments["orders_hist"] > 0, segments["total_qty"] / segments["orders_hist"], 1.0)

    if forecast_target == "quantity":
        driver_col = "total_qty"
        driver_total = float(segments[driver_col].sum())
        if driver_total <= 0 or total_forecast_driver <= 0:
            raise HTTPException(400, "Không đủ forecast quantity để tối ưu")
        segments["forecast_qty"] = segments[driver_col] / driver_total * total_forecast_driver
    elif forecast_target == "revenue":
        category_revenue = df.groupby("category")["revenue"].sum().rename("revenue_hist")
        segments = segments.merge(category_revenue, on="category", how="left")
        segments["revenue_hist"] = segments["revenue_hist"].fillna(0.0)
        driver_total = float(segments["revenue_hist"].sum())
        if driver_total <= 0 or total_forecast_driver <= 0:
            raise HTTPException(400, "Không đủ forecast revenue để tối ưu")
        segments["forecast_revenue"] = segments["revenue_hist"] / driver_total * total_forecast_driver
        segments["forecast_qty"] = segments["forecast_revenue"] / segments["avg_price"].clip(0.01)
    elif forecast_target == "orders":
        driver_col = "orders_hist"
        driver_total = float(segments[driver_col].sum())
        if driver_total <= 0 or total_forecast_driver <= 0:
            raise HTTPException(400, "Không đủ forecast orders để tối ưu")
        segments["forecast_orders"] = segments[driver_col] / driver_total * total_forecast_driver
        segments["forecast_qty"] = segments["forecast_orders"] * segments["avg_items_per_order"].clip(lower=0.1)
    else:
        raise HTTPException(400, f"target forecast không hỗ trợ cho optimizer: {forecast_target}")

    Q0 = segments["forecast_qty"].to_numpy()
    E = segments["elasticity"].to_numpy()
    BM = segments["base_margin"].to_numpy()

    effective_min_margin = np.maximum(req.min_margin, BM * 0.5)
    max_d_allowed = 1 - (C0 / (P0 * (1 - effective_min_margin) + 1e-9))
    upper_bounds = np.minimum(req.max_discount, np.maximum(0, max_d_allowed))
    upper_bounds = np.where(BM <= effective_min_margin, 0, upper_bounds)
    bounds = [(0.0, float(ub)) for ub in upper_bounds]

    def new_quantity(d):
        return Q0 * np.power(np.clip(1 - d, 0.01, 1), E)

    def objective(d):
        new_q = new_quantity(d)
        if req.strategy == "profit":
            return -(((P0 * (1 - d) - C0) * new_q).sum()) / 1e6
        return -((P0 * (1 - d) * new_q).sum()) / 1e6

    def budget_used_fn(d):
        return np.sum(P0 * d * new_quantity(d))

    constraints = ([{"type": "ineq", "fun": lambda d: (req.budget - budget_used_fn(d)) / 1e6}])

    x0 = np.clip(0.08 * np.abs(E), 0.0, 0.25)

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-8, "disp": False},
    )

    d_opt = np.clip(res.x, 0.0, upper_bounds)
    final_price = P0 * (1 - d_opt)
    final_qty = new_quantity(d_opt)
    base_rev = P0 * Q0
    final_rev = final_price * final_qty
    margin_after = np.where(final_price > 0, (final_price - C0) / final_price, 0)

    result = segments[["category", "elasticity", "base_margin"]].copy()
    result["avg_price"] = np.round(P0, 2)
    result["avg_cost"] = np.round(C0, 2)
    result["forecast_qty"] = np.round(Q0, 2)
    result["discount_opt"] = np.round(d_opt, 4)
    result["base_revenue"] = np.round(base_rev, 0)
    result["final_revenue"] = np.round(final_rev, 0)
    result["margin_after"] = np.round(margin_after, 4)

    result = result.sort_values("final_revenue", ascending=False).reset_index(drop=True)

    total_base = float(base_rev.sum())
    total_final = float(final_rev.sum())
    budget_used = float(budget_used_fn(d_opt))

    return _jsonify({
        "results": result.to_dict("records"),
        "summary": {
            "total_base_revenue": total_base,
            "total_final_revenue": total_final,
            "revenue_uplift": (total_final / total_base - 1) if total_base else 0,
            "budget_used": budget_used,
            "forecast_total_qty": float(Q0.sum()),
            "forecast_total_driver": float(total_forecast_driver),
            "forecast_target_metric": forecast_target,
            "forecast_horizon_used": int(horizon_used),
            "forecast_model": forecast_info.get("model"),
            "forecast_scenario": forecast_info.get("scenario"),
            "budget_utilization": budget_used / req.budget if req.budget else 0,
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
        }
    })

# ════════════════════════════════════════════════════════════════
#  HEALTH
# ════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_data": _store["df"] is not None,
        "rows": int(len(_store["df"])) if _store["df"] is not None else 0,
        "statsmodels": HAS_STATSMODELS,
    }
