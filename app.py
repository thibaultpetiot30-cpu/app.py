import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mini Barra Risk Model", layout="wide")
st.title("Mini Barra Risk Model – Full Demo 📊")

st.markdown("""
**Input format (long form)**  
Columns required:
- `date` (YYYY-MM-DD)
- `asset` (asset name)
- one or more factor columns named like `factor_Mkt`, `factor_SMB`, `factor_HML`, ...
- `return` (asset return, same periodicity as factors, e.g. monthly)

Tip: factors are **auto-detected** by the prefix `factor_`.
""")

# ---------- Sidebar
st.sidebar.header("Load data")
demo_btn = st.sidebar.button("Use built-in demo dataset (15 assets × 36 months × 3 factors)")
uploaded = st.sidebar.file_uploader("...or upload your CSV", type="csv")
st.sidebar.markdown("Need a sample? Download one below and re-upload it.")

# ---------- Load data
if demo_btn:
    url = None
    st.session_state["df"] = None  # reset
    st.sidebar.success("Scroll down and click the demo download link in the main page, then re-upload it here if you want.")
elif uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
else:
    st.session_state.setdefault("df", None)

st.markdown("#### Download a big sample dataset")
st.write("Use this file for a convincing live demo (15 assets × 36 months × 3 factors):")
st.link_button("Download mini_barra_big_sample.csv", "sandbox:/mnt/data/mini_barra_big_sample.csv")

df = st.session_state["df"]
if df is None:
    st.info("👉 Upload your CSV in the sidebar (or download the sample first).")
    st.stop()

# ---------- Basic validation
required_cols = {"date","asset","return"}
factor_cols = [c for c in df.columns if c.startswith("factor_")]
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns: {missing}. Also include at least one factor column starting with `factor_`.")
    st.stop()
if not factor_cols:
    st.error("No factor columns detected. Please include columns like `factor_Mkt`, `factor_SMB`, ...")
    st.stop()

# Keep only useful cols
df = df[["date","asset","return"] + factor_cols].copy()
df["date"] = pd.to_datetime(df["date"])

# ---------- Align returns (assets) and factor series
pivot_ret = df.pivot(index="date", columns="asset", values="return").sort_index()
factors = df.groupby("date")[factor_cols].mean().sort_index()

# Common dates only
idx = pivot_ret.index.intersection(factors.index)
pivot_ret = pivot_ret.loc[idx].dropna(axis=0, how="any")  # drop dates with missing returns
factors = factors.loc[pivot_ret.index]

if pivot_ret.shape[0] < len(factor_cols) + 2:
    st.error("Not enough time points after alignment to run regressions. Check your data.")
    st.stop()

st.success(f"Data aligned: {pivot_ret.shape[0]} periods, {pivot_ret.shape[1]} assets, {len(factor_cols)} factors.")

st.subheader("Preview")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Asset returns (wide)**")
    st.dataframe(pivot_ret.head())
with col2:
    st.markdown("**Factors**")
    st.dataframe(factors.head())

# ---------- Estimate betas (OLS, no intercept on purpose – returns explained by pure factors)
X = factors.values  # T x K
betas = {}
spec_vars = {}
for asset in pivot_ret.columns:
    y = pivot_ret[asset].values  # T
    # OLS: b = (X'X)^(-1) X'y
    b = np.linalg.lstsq(X, y, rcond=None)[0]  # K
    betas[asset] = b
    resid = y - X @ b
    spec_vars[asset] = resid.var(ddof=len(factor_cols))  # unbiased w.r.t K

betas_df = pd.DataFrame(betas, index=factor_cols).T  # assets x factors
spec_var_ser = pd.Series(spec_vars, name="spec_var")

# --- Factor covariance: Sample vs EWMA
st.sidebar.subheader("Factor covariance")
cov_method = st.sidebar.selectbox("Method", ["Sample (unweighted)", "EWMA (half-life)"])

X = factors.values  # T × K
K = X.shape[1]

if cov_method == "EWMA (half-life)":
    hl = st.sidebar.slider("Half-life (periods)", 3, 60, 12)
    lam = 0.5 ** (1.0 / hl)
    Xc = X - X.mean(axis=0, keepdims=True)
    Tn = Xc.shape[0]
    wts = np.array([lam ** (Tn-1-t) for t in range(Tn)], dtype=float)
    wts /= wts.sum()
    F_cov = np.zeros((K, K))
    for t in range(Tn):
        xt = Xc[t:t+1].T
        F_cov += wts[t] * (xt @ xt.T)
else:
    F_cov = np.cov(X.T, ddof=1)

F_cov_df = pd.DataFrame(F_cov, index=factor_cols, columns=factor_cols)

st.subheader("Estimated exposures (betas) & risks")
c1, c2 = st.columns([2,1])
with c1:
    st.markdown("**Exposures (betas)**")
    st.dataframe(betas_df.round(3))
with c2:
    st.markdown("**Factor covariance (F)**")
    st.dataframe(F_cov_df.round(4))
    st.markdown("**Specific variances**")
    st.dataframe(spec_var_ser.round(6))

# ---------- Portfolio weights (editable)
st.subheader("Portfolio weights")
weights_df = pd.DataFrame({"asset": pivot_ret.columns, "weight": 1.0/len(pivot_ret.columns)})
weights_df = st.data_editor(weights_df, num_rows="dynamic", use_container_width=True)
# sanitize & normalize
weights_df = weights_df.dropna()
weights_df["weight"] = pd.to_numeric(weights_df["weight"], errors="coerce").fillna(0.0)
# Keep only assets present
weights_df = weights_df[weights_df["asset"].isin(pivot_ret.columns)]
# Reindex to asset order
weights_df = weights_df.set_index("asset").reindex(pivot_ret.columns).fillna(0.0)
w = weights_df["weight"].values
if w.sum() == 0:
    st.error("All weights are zero. Please set some positive weights.")
    st.stop()
w = w / w.sum()
# --- Factor shock scenario (Δf in standard deviations)
st.sidebar.subheader("Factor shock scenario (Δf)")
std_f = np.sqrt(np.diag(F_cov))
shock = []
for i, f in enumerate(factor_cols):
    s = st.sidebar.slider(f"{f} shock (in σ)", -3.0, 3.0, 0.0, 0.1)
    shock.append(s * std_f[i])
delta_f = np.array(shock)

# portfolio factor exposure
B = betas_df.values          # N × K
b_p = (B.T @ w)              # K
exp_dP = float(b_p @ delta_f)
st.sidebar.metric("Scenario P&L (one period)", f"{exp_dP:.2%}")

# --- Portfolio risk calculations ---
B = betas_df.values  # N x K
Delta = np.diag(spec_var_ser.reindex(pivot_ret.columns).values)  # N x N
cov_assets = B @ F_cov @ B.T + Delta
var_p = float(w.T @ cov_assets @ w)
vol_p = np.sqrt(var_p)
var_p_factor = float(w.T @ (B @ F_cov @ B.T) @ w)
var_p_specific = float(w.T @ Delta @ w)

# --- Asset-level variance contributions (sum = total portfolio variance) ---
Sigma = cov_assets  # N x N
asset_contrib = w * (Sigma @ w)

asset_contrib_df = pd.DataFrame({
    "asset": pivot_ret.columns,
    "var_contrib": asset_contrib,
})
asset_contrib_df["pct_total_var"] = (
    asset_contrib_df["var_contrib"] / var_p * 100.0 if var_p > 0 else 0.0
)
asset_contrib_df = asset_contrib_df.sort_values("var_contrib", ascending=False)

st.subheader("Asset-level risk contributions")
st.dataframe(asset_contrib_df.head(10).round(6))

figA, axA = plt.subplots()
axA.bar(asset_contrib_df["asset"].head(10), asset_contrib_df["var_contrib"].head(10))
axA.set_title("Top 10 asset variance contributions")
axA.set_ylabel("Variance")
axA.set_xlabel("Asset")
st.pyplot(figA, clear_figure=True)

# Factor exposure of the **portfolio**
b_p = (B.T @ w)                  # K x 1
# Factor variance breakdown using risk contributions RC_k = b_p[k] * (F b_p)[k]
Fb = F_cov @ b_p
rc_factors = b_p * Fb            # K vector, sums to b_p' F b_p (= var_p_factor)
rc_df = pd.DataFrame({"factor": factor_cols, "variance_contrib": rc_factors})
rc_df["pct_of_factor_var"] = rc_df["variance_contrib"] / var_p_factor * 100.0
rc_df = rc_df.sort_values("variance_contrib", ascending=False)

# ---------- Display metrics
st.subheader("Portfolio risk")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Variance (monthly)", f"{var_p:.6f}")
with c2:
    st.metric("Volatility (monthly, σ)", f"{vol_p:.2%}")
with c3:
    ann_vol = vol_p * np.sqrt(12.0)  # if monthly data
    st.metric("Volatility (annualized, σ)", f"{ann_vol:.2%}")

st.markdown("**Decomposition**")
st.write(f"- Factor variance: `{var_p_factor:.6f}`  |  Specific variance: `{var_p_specific:.6f}`")
st.dataframe(rc_df.reset_index(drop=True).round(6))

# --- Backtest: predicted vs realized volatility ---
st.subheader("Backtest: predicted vs realized volatility")
do_bt = st.checkbox("Run rolling backtest (one-step-ahead forecast)")
win = st.slider("Estimation window (periods)", 12, max(24, len(pivot_ret)-2), 24)
if do_bt and len(pivot_ret) > win + 2:
    dates_bt = pivot_ret.index
    pred_vol, real_vol, when = [], [], []
    for t in range(win, len(dates_bt)-1):
        # window t-win..t-1 to estimate betas & factor covariance
        Rw = pivot_ret.iloc[t-win:t, :]
        Fw = factors.iloc[t-win:t, :]
        Xw = Fw.values

        # betas (OLS, no intercept)
        Bw = []
        for asset in Rw.columns:
            yw = Rw[asset].values
            bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            Bw.append(bw)
        Bw = np.array(Bw)  # N × K

        # factor covariance on the window
        Fw_cov = np.cov(Xw.T, ddof=1)

        # one-step-ahead risk forecast using current weights w
        cov_assets_w = Bw @ Fw_cov @ Bw.T
        var_pred = float(w.T @ cov_assets_w @ w)
        pred_vol.append(np.sqrt(var_pred))

        # realized next-period absolute portfolio return as a proxy
        rp_next = float((ret_wide.iloc[t+1, :].values @ w))
        real_vol.append(abs(rp_next))

        when.append(dates_bt[t+1])

    bt = pd.DataFrame({
        "date": when,
        "pred_vol": pred_vol,
        "realized_abs_ret": real_vol
    }).set_index("date")

    # plot
    figbt, axbt = plt.subplots()
    axbt.plot(bt.index, bt["pred_vol"], label="Predicted σ (t→t+1)")
    axbt.plot(bt.index, bt["realized_abs_ret"], label="|Realized return| (t+1)")
    axbt.set_title("Backtest: risk forecast vs realized")
    axbt.legend()
    st.pyplot(figbt, clear_figure=True)

    st.dataframe(bt.tail().round(6))

# ---------- Downloads
st.subheader("Download results")
# exposures
csv_betas = betas_df.round(6).to_csv().encode("utf-8")
st.download_button("Download exposures (betas).csv", data=csv_betas, file_name="exposures_betas.csv", mime="text/csv")

# factor cov
csv_F = F_cov_df.round(6).to_csv().encode("utf-8")
st.download_button("Download factor_covariance.csv", data=csv_F, file_name="factor_covariance.csv", mime="text/csv")

# specific
csv_spec = spec_var_ser.round(8).to_csv().encode("utf-8")
st.download_button("Download specific_variances.csv", data=csv_spec, file_name="specific_variances.csv", mime="text/csv")

# portfolio report
report = pd.DataFrame({
    "metric": ["var_total_monthly","vol_monthly","vol_annualized","var_factor","var_specific"],
    "value":  [var_p, vol_p, ann_vol, var_p_factor, var_p_specific]
})
csv_report = report.round(8).to_csv(index=False).encode("utf-8")
st.download_button("Download portfolio_report.csv", data=csv_report, file_name="portfolio_report.csv", mime="text/csv")

st.caption("Method: OLS betas on factors; factor covariance from sample; variance contributions RC_k = b_p[k] * (F b_p)[k].")
