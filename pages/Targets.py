# pages/05_Targets.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="🎯 Targets", layout="wide")
st.title("🎯 Targets — Who to focus on this month")

# ---- Pull processed data from session_state ----
if "sales_data" not in st.session_state:
    st.info("Please go to the main page, upload your Excel, then return here.")
    st.stop()

data        = st.session_state["sales_data"]
long_df     = data["long"]        # CustomerName, Salesperson, Month (datetime), Year, Sales
monthly_sc  = data["monthly_sc"]  # CustomerName, Salesperson, Month (datetime), Sales

if long_df.empty or monthly_sc.empty:
    st.info("No processed rows found.")
    st.stop()

# Normalize Month to first-of-month timestamps (safety)
long_df["Month"] = pd.to_datetime(long_df["Month"]).dt.to_period("M").dt.to_timestamp()
monthly_sc["Month"] = pd.to_datetime(monthly_sc["Month"]).dt.to_period("M").dt.to_timestamp()

anchor = pd.to_datetime(long_df["Month"].max())  # latest month in dataset
if pd.isna(anchor):
    st.info("No monthly data available.")
    st.stop()

prev_m    = anchor - pd.DateOffset(months=1)
same_m_ly = anchor - pd.DateOffset(years=1)
ty, ly    = anchor.year, anchor.year - 1

# ---------------- Helper to compute targets per salesperson ----------------
def compute_targets_for_salesperson(df_sp: pd.DataFrame, anchor_ts: pd.Timestamp) -> pd.DataFrame:
    """Return a targets dataframe for ONE salesperson (df_sp already filtered to that Salesperson)."""

    def month_sum(ts: pd.Timestamp) -> pd.Series:
        # Sum by CustomerName for a specific month
        month_key = pd.Timestamp(ts.year, ts.month, 1)
        return (
            df_sp[df_sp["Month"] == month_key]
            .groupby("CustomerName")["Sales"].sum()
        )

    # Latest, previous, and same month last year
    M0   = month_sum(anchor_ts).rename("M0")                      # this month
    M_1  = month_sum(anchor_ts - pd.DateOffset(months=1)).rename("M-1")
    M_LY = month_sum(anchor_ts - pd.DateOffset(years=1)).rename("LY_same")

    # Averages
    LY_avg = (
        df_sp[df_sp["Month"].dt.year == (anchor_ts.year - 1)]
        .groupby("CustomerName")["Sales"].mean()
        .rename("LY_avg")
    )
    TY_avg = (
        df_sp[(df_sp["Month"].dt.year == anchor_ts.year) &
              (df_sp["Month"] <= pd.Timestamp(anchor_ts.year, anchor_ts.month, 1))]
        .groupby("CustomerName")["Sales"].mean()
        .rename("TY_avg")
    )

    # Recency (months since last >0 sale)
    last_sale = (
        df_sp[df_sp["Sales"] > 0]
        .groupby("CustomerName")["Month"].max()
        .rename("last_month")
    )
    recency = ((anchor_ts.to_period("M") - last_sale.dt.to_period("M"))
               .apply(lambda x: x.n if pd.notna(x) else np.inf)
               .rename("Recency_m"))

    # --- build the base table (guarantee a 'CustomerName' column) ---
    base = pd.concat([M0, M_1, M_LY, LY_avg, TY_avg, recency], axis=1).fillna(0)
    base.index.name = "CustomerName"
    base = base.reset_index()

    # Replace inf recency with a large sentinel (e.g., 999)
    if "Recency_m" in base.columns:
        base["Recency_m"] = (
            pd.to_numeric(base["Recency_m"], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(999)
        )

    # Metrics
    base["MoM_Δ"]     = base["M0"] - base["M-1"]
    base["MoM_%"]     = np.where(base["M-1"] != 0, base["MoM_Δ"] / base["M-1"] * 100, np.nan)
    base["YoY_Δ"]     = base["M0"] - base["LY_same"]
    base["YoY_%"]     = np.where(base["LY_same"] != 0, base["YoY_Δ"] / base["LY_same"] * 100, np.nan)
    base["Gap_vs_LY"] = (base["LY_avg"] - base["M0"]).clip(lower=0)
    base["Gap_vs_TY"] = (base["TY_avg"] - base["M0"]).clip(lower=0)
    base["Potential"] = base["LY_avg"]  # proxy for wallet size

    # Normalized scoring (tweak weights if you like)
    def _norm(s: pd.Series) -> pd.Series:
        s = s.copy()
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng != 0 else pd.Series(0.0, index=s.index)

    score = (
        0.40 * _norm(base["Gap_vs_LY"]) +
        0.20 * _norm(base["Gap_vs_TY"]) +
        0.20 * _norm(base["Potential"]) +
        0.20 * _norm((-base["MoM_Δ"]).clip(lower=0))
        # + 0.15 * _norm(base["Recency_m"])  # enable if you want to push stale clients up
    )

    base["TargetScore"] = score.round(4)
    return base

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
salespeople_all = sorted(monthly_sc["Salesperson"].dropna().astype(str).unique())
sp_pick = st.sidebar.selectbox("Salesperson", ["All"] + salespeople_all, index=0)

min_score      = st.sidebar.number_input("Min TargetScore", value=0.20, step=0.05, format="%.2f")
min_gap_ly     = st.sidebar.number_input("Min Gap vs LY (absolute)", value=0.0, step=1000.0, format="%.2f")
min_potential  = st.sidebar.number_input("Min Potential (LY avg)", value=0.0, step=1000.0, format="%.2f")
only_losses    = st.sidebar.checkbox("Only show MoM losses (M0 < M-1)", value=True)
min_recency    = st.sidebar.number_input("Min recency (months since last sale)", value=0.0, step=1.0, format="%.0f")
search_customer = st.sidebar.text_input("Search customer (contains)", value="")
top_n          = st.sidebar.number_input("Top N", min_value=5, max_value=200, value=50, step=5)

# Optional margin filter if present
has_margin = any(c.lower().replace(" ", "") in ("gp", "grossprofit", "gross_profit", "margin")
                 for c in monthly_sc.columns)
if has_margin:
    min_margin = st.sidebar.number_input("Min margin (if available)", value=0.0, step=1000.0, format="%.2f")

# ---------------- Build Targets table ----------------
targets_list = []

if sp_pick == "All":
    for sp in salespeople_all:
        df_sp = monthly_sc[monthly_sc["Salesperson"] == sp].copy()
        t = compute_targets_for_salesperson(df_sp, anchor)
        t.insert(0, "Salesperson", sp)
        targets_list.append(t)
else:
    df_sp = monthly_sc[monthly_sc["Salesperson"] == sp_pick].copy()
    t = compute_targets_for_salesperson(df_sp, anchor)
    t.insert(0, "Salesperson", sp_pick)
    targets_list.append(t)

targets = pd.concat(targets_list, ignore_index=True) if targets_list else pd.DataFrame()

# Apply filters
if not targets.empty:
    if only_losses:
        targets = targets[targets["MoM_Δ"] < 0]
    if min_gap_ly > 0:
        targets = targets[targets["Gap_vs_LY"] >= min_gap_ly]
    if min_potential > 0:
        targets = targets[targets["Potential"] >= min_potential]
    if min_recency > 0:
        targets = targets[targets["Recency_m"] >= min_recency]
    if search_customer:
        mask = targets["CustomerName"].str.contains(search_customer, case=False, na=False)
        targets = targets[mask]
    if has_margin and "margin" in targets.columns:
        targets = targets[targets["margin"] >= min_margin]

# Sort & cap and min score
if not targets.empty:
    targets = targets[targets["TargetScore"] >= float(min_score)]
    targets = targets.sort_values(
        ["TargetScore", "Gap_vs_LY", "Potential"],
        ascending=[False, False, False]
    ).head(int(top_n))

# Display
lm_label  = anchor.strftime("%b %Y")
pm_label  = prev_m.strftime("%b %Y")
lym_label = same_m_ly.strftime("%b %Y")

st.subheader(f"📋 Ranked Targets — {lm_label}")
if targets.empty:
    st.info("No rows match the current filters.")
else:
    show_cols = [
        "Salesperson", "CustomerName", "M0", "M-1", "MoM_Δ", "MoM_%", "LY_same", "YoY_Δ", "YoY_%",
        "LY_avg", "TY_avg", "Gap_vs_LY", "Gap_vs_TY", "Recency_m", "Potential", "TargetScore"
    ]
    have = [c for c in show_cols if c in targets.columns]
    disp = targets[have].rename(columns={
        "M0": f"Sales {lm_label}",
        "M-1": f"Sales {pm_label}",
        "LY_same": f"Sales {lym_label}",
        "LY_avg": f"Avg {ly}",
        "TY_avg": f"Avg {ty} (to-date)"
    })
    st.dataframe(disp, use_container_width=True)

    st.download_button(
        "⬇️ Download Targets (CSV)",
        disp.to_csv(index=False).encode("utf-8"),
        file_name=f"targets_{'all' if sp_pick=='All' else sp_pick}_{anchor.strftime('%Y_%m')}.csv",
        mime="text/csv",
    )

    # ===================== AI Summaries (per filtered targets) =====================
    st.markdown("### 🤖 AI Targeting Summaries")

    # Controls
    default_model = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
    model_name = st.text_input("Model", value=default_model, help="Deployed text model, e.g., gpt-4.1-mini")
    n_rows = len(disp)
    topN_for_ai = st.slider("Summarize top N rows", min_value=1, max_value=max(1, n_rows), value=min(10, n_rows))
    tone = st.selectbox("Tone", ["Crisp & direct", "Supportive", "Data-first"], index=0)
    add_risk_flag = st.checkbox("Include churn risk flag", value=True)

    # Provider (OpenAI by default)
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = None
        st.warning("`openai` package not found. Install with: pip install openai")

    @st.cache_resource(show_spinner=False)
    def _get_openai_client():
        if OpenAI is None:
            return None
        try:
            return OpenAI()  # uses OPENAI_API_KEY from env or st.secrets
        except Exception as e:
            st.error(f"OpenAI client init failed: {e}")
            return None

    client = _get_openai_client()

    SYSTEM_PROMPT = """You are a sales coach. Write a brief, actionable, non-fluffy summary for a single customer using these metrics:
- MoM Δ/%, YoY Δ/%, gap vs last-year average, gap vs this-year average (to date), recency (months since last >0 sale), last-year average as wallet size (Potential).
Guidelines:
- 2–3 bullet points (max ~60 words total).
- Start with a decision: “Target” or “Monitor”, then 1–2 concise reasons from the data, then 1 action.
- If risk is high (sharp MoM & YoY drop and large gap vs LY), call it out.
- No greetings. No markdown headings. Keep numeric signals where helpful.
"""

    def _row_to_prompt(r: pd.Series, month_label: str, last_year: int, this_year: int, tone_str: str, include_risk: bool) -> str:
        # Safe numeric getter
        def _get(col, default=np.nan):
            v = r.get(col, default)
            try:
                return float(v) if pd.notna(v) else np.nan
            except Exception:
                return np.nan

        base = {
            "CustomerName": r["CustomerName"],
            "Salesperson": r["Salesperson"],
            "M0": _get("M0", 0.0),
            "M-1": _get("M-1", 0.0),
            "MoM_Δ": _get("MoM_Δ", 0.0),
            "MoM_%": _get("MoM_%", np.nan),
            "LY_same": _get("LY_same", 0.0),
            "YoY_Δ": _get("YoY_Δ", 0.0),
            "YoY_%": _get("YoY_%", np.nan),
            "LY_avg": _get("LY_avg", 0.0),
            "TY_avg": _get("TY_avg", 0.0),
            "Gap_vs_LY": _get("Gap_vs_LY", 0.0),
            "Gap_vs_TY": _get("Gap_vs_TY", 0.0),
            "Recency_m": _get("Recency_m", np.nan),
            "Potential": _get("Potential", 0.0),
            "TargetScore": _get("TargetScore", 0.0),
        }

        # Format helpers (avoid nested f-strings)
        def fmt_pct(x):
            return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.1f}%"

        mom_pct_str = fmt_pct(base["MoM_%"])
        yoy_pct_str = fmt_pct(base["YoY_%"])
        recency_str = "NA" if (np.isnan(base["Recency_m"])) else str(int(base["Recency_m"]))

        risk_hint = (
            "Include a churn-risk note if both MoM and YoY are negative and Gap_vs_LY is large."
            if include_risk else
            "No churn-risk flags."
        )

        return (
            f"Month: {month_label}\n"
            f"Salesperson: {base['Salesperson']}\n"
            f"Customer: {base['CustomerName']}\n\n"
            f"Data:\n"
            f"- This month (M0): {base['M0']:,.0f}\n"
            f"- Prev month (M-1): {base['M-1']:,.0f} | MoM Δ: {base['MoM_Δ']:,.0f} | MoM %: {mom_pct_str}\n"
            f"- Same month last year: {base['LY_same']:,.0f} | YoY Δ: {base['YoY_Δ']:,.0f} | YoY %: {yoy_pct_str}\n"
            f"- Avg {last_year}: {base['LY_avg']:,.0f} | Avg {this_year} (to-date): {base['TY_avg']:,.0f}\n"
            f"- Gap vs LY: {base['Gap_vs_LY']:,.0f} | Gap vs TY: {base['Gap_vs_TY']:,.0f}\n"
            f"- Recency (months since last sale): {recency_str}\n"
            f"- Potential (LY avg): {base['Potential']:,.0f}\n"
            f"- TargetScore: {base['TargetScore']:.3f}\n\n"
            f"Tone: {tone_str}. {risk_hint}\n"
            f"Write 2–3 bullet points, <60 words total. Start with 'Target' or 'Monitor'."
        )

    @st.cache_data(show_spinner=False)
    def _summarize_with_openai(model: str, system_prompt: str, user_prompt: str) -> str:
        """Calls OpenAI; tries Responses API first, then falls back to Chat Completions."""
        if client is None:
            return "(LLM client not available. Set OPENAI_API_KEY and install `openai`.)"
        # Try Responses API
        try:
            resp = client.responses.create(
                model=model,
                input=f"{system_prompt}\n\n{user_prompt}",
                temperature=0.2,
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text.strip()
            if hasattr(resp, "output") and resp.output and resp.output[0].content:
                return getattr(resp.output[0].content[0], "text", "").strip()
        except Exception:
            pass
        # Fallback: Chat Completions
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM error: {e})"

    if st.button("Generate AI summaries"):
        if disp.empty:
            st.info("Nothing to summarize. Adjust filters above.")
        else:
            # Keep the same ordering as the displayed table
            keys = disp[["Salesperson", "CustomerName"]].head(int(topN_for_ai))
            subset = keys.merge(targets, on=["Salesperson", "CustomerName"], how="left")

            rows = []
            with st.spinner("Summarizing…"):
                for _, r in subset.iterrows():
                    user_prompt = _row_to_prompt(r, lm_label, ly, ty, tone, add_risk_flag)
                    summary = _summarize_with_openai(model_name, SYSTEM_PROMPT, user_prompt)
                    rows.append({
                        "Salesperson": r["Salesperson"],
                        "CustomerName": r["CustomerName"],
                        "Summary": summary
                    })

            sum_df = pd.DataFrame(rows)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download AI Summaries (CSV)",
                sum_df.to_csv(index=False).encode("utf-8"),
                file_name=f"ai_summaries_{'all' if sp_pick=='All' else sp_pick}_{anchor.strftime('%Y_%m')}.csv",
                mime="text/csv",
            )

st.markdown("---")

# ---------------- One-click Deep Dive ----------------
st.subheader("🔎 Deep Dive")

if 'disp' not in locals() or disp.empty:
    st.info("Nothing to deep-dive — adjust filters above.")
    st.stop()

c1, c2 = st.columns([1, 2])
with c1:
    dd_sp = st.selectbox(
        "Salesperson",
        sorted(disp["Salesperson"].unique().tolist())
    )
    dd_customer = st.selectbox(
        "Customer",
        sorted(disp[disp["Salesperson"] == dd_sp]["CustomerName"].unique().tolist())
    )

# KPI snapshot for the selected salesperson & CustomerName
def month_value(sp: str, customer: str, ts: pd.Timestamp) -> float:
    rows = monthly_sc[
        (monthly_sc["Salesperson"] == sp) &
        (monthly_sc["CustomerName"] == customer) &
        (monthly_sc["Month"] == pd.Timestamp(ts.year, ts.month, 1))
    ]["Sales"]
    return float(rows.sum()) if not rows.empty else 0.0

m_now   = month_value(dd_sp, dd_customer, anchor)
m_prev  = month_value(dd_sp, dd_customer, prev_m)
m_same  = month_value(dd_sp, dd_customer, same_m_ly)
m_same2 = month_value(dd_sp, dd_customer, anchor - pd.DateOffset(years=2))

# Averages for this customer (per salesperson)
ty_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_customer) &
    (monthly_sc["Month"].dt.year == ty) &
    (monthly_sc["Month"] <= anchor)
]["Sales"]
ly_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_customer) &
    (monthly_sc["Month"].dt.year == ly)
]["Sales"]

avg_ty = float(ty_slice.mean()) if not ty_slice.empty else 0.0
avg_ly = float(ly_slice.mean()) if not ly_slice.empty else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"{anchor.strftime('%b %Y')} Sales", f"{m_now:,.0f}",
          f"{(m_now - m_prev):+,.0f} vs prev mo" if m_prev else "—")
k2.metric(f"Vs {ty} avg (to-date)", f"{(m_now - avg_ty):+,.0f}",
          f"{((m_now - avg_ty)/avg_ty*100):+.1f}%" if avg_ty else "—")
k3.metric(f"Vs {ly} avg", f"{(m_now - avg_ly):+,.0f}",
          f"{((m_now - avg_ly)/avg_ly*100):+.1f}%" if avg_ly else "—")
k4.metric("Same month YoY", f"{(m_now - m_same):+,.0f}",
          f"{((m_now - m_same)/m_same*100):+.1f}%" if m_same else "—")

# Full-year comparison plot (last year vs current year) for this customer & salesperson
fy_curr = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_customer) &
    (monthly_sc["Month"].dt.year == ty)
].copy()
fy_last = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_customer) &
    (monthly_sc["Month"].dt.year == ly)
].copy()

for df in (fy_curr, fy_last):
    if not df.empty:
        df["MonthNum"] = df["Month"].dt.month
        df["MonthLabel"] = df["Month"].dt.strftime("%b")

fy_curr_m = fy_curr.groupby(["MonthNum", "MonthLabel"], as_index=False)["Sales"].sum().sort_values("MonthNum")
fy_last_m = fy_last.groupby(["MonthNum", "MonthLabel"], as_index=False)["Sales"].sum().sort_values("MonthNum")

label_curr = f"{ty} (to-date)" if fy_curr_m["MonthNum"].nunique() < 12 else f"{ty} (Jan–Dec)"
label_last = f"{ly} (Jan–Dec)"

to_plot = []
if not fy_last_m.empty:
    fy_last_m["YearLabel"] = label_last
    to_plot.append(fy_last_m)
if not fy_curr_m.empty:
    fy_curr_m["YearLabel"] = label_curr
    to_plot.append(fy_curr_m)

if to_plot:
    combined_full = pd.concat(to_plot, ignore_index=True)
    fig_full = px.line(
        combined_full, x="MonthNum", y="Sales", color="YearLabel", markers=True,
        hover_data={"Sales": ":,.2f", "MonthNum": False, "MonthLabel": True},
        title=f"{dd_customer} — Full-Year Sales by Month ({ly} vs {ty}) · handled by {dd_sp}"
    )
    fig_full.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]),
        xaxis_title="Month", yaxis_title="Sales Amount", legend_title_text="Series", height=420
    )
    st.plotly_chart(fig_full, use_container_width=True)

    st.download_button(
        "⬇️ Download Full-Year (current vs last) — CSV",
        combined_full.assign(Year=combined_full["YearLabel"])[["Year", "MonthNum", "MonthLabel", "Sales"]]
            .rename(columns={"Sales": "Sales (Monthly)"}).to_csv(index=False).encode("utf-8"),
        file_name=f"full_year_{dd_customer}_{dd_sp}_{ly}_vs_{ty}.csv",
        mime="text/csv",
    )
else:
    st.info("No full-year month-level data available for this selection.")

st.caption("> Score = weighted blend of Gap vs LY/TY, wallet size (LY avg), and recent MoM drop. Tweak weights in code if needed.")
