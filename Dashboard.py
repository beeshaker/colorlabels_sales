# app.py
# Streamlit app: Top 20 clients per salesperson with year-specific interactivity + client improvement tracker
# Run: streamlit run app.py

import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Top 20 Clients by Year", layout="wide")
st.title("📈 Top 20 Clients per Salesperson — Year-Aware & Interactive")

st.sidebar.header("1) Upload data")
uploaded = st.sidebar.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"])

# ------------------------
# Helpers
# ------------------------
MONTH_HEADER_RE = re.compile(r"^[A-Za-z]{3}-\d{2}$")  # e.g., Jan-23

def annualize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given long_df or a slice with Month + Sales,
    return yearly totals with partial years annualized (sales * 12 / months_of_data).
    """
    df = df.copy()
    df["Year"] = df["Month"].dt.year
    grouped = df.groupby("Year").agg(
        Sales_sum=("Sales", "sum"),
        Months=("Month", lambda x: x.dt.month.nunique())
    ).reset_index()
    grouped["AnnualizedSales"] = np.where(
        grouped["Months"] < 12,
        grouped["Sales_sum"] * 12 / grouped["Months"],
        grouped["Sales_sum"]
    )
    return grouped[["Year", "AnnualizedSales", "Months"]]

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 'Sum of Sum of ' prefix and strip whitespace."""
    def _clean(c):
        if isinstance(c, str):
            c = c.strip()
            c = c.replace("Sum of Sum of ", "").replace("Sum of ", "")
        return c
    return df.rename(columns=_clean)

def detect_month_columns(df: pd.DataFrame) -> list:
    """Return columns that look like month headers (e.g., Jan-23), excluding YTD/Projected."""
    cols = []
    for c in df.columns:
        if isinstance(c, str) and MONTH_HEADER_RE.match(c) and ("YTD" not in c and "Projected" not in c):
            cols.append(c)
    return cols

def longify_sales(df: pd.DataFrame, id_cols: list, month_cols: list) -> pd.DataFrame:
    """Melt month columns, parse Month to datetime, add Year, drop NaNs."""
    melted = df.melt(id_vars=id_cols, value_vars=month_cols,
                     var_name="MonthStr", value_name="Sales")
    melted = melted.dropna(subset=["Sales"])
    melted["Month"] = pd.to_datetime(melted["MonthStr"], format="%b-%y", errors="coerce")
    melted = melted.dropna(subset=["Month"])
    melted["Year"] = melted["Month"].dt.year.astype(int)
    return melted

@st.cache_data(show_spinner=False)
def process_file(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=xls.sheet_names[0])
    df = clean_columns(df)

    # Match core columns case-insensitively
    cols_lower = {c.lower(): c for c in df.columns if isinstance(c, str)}
    name_col = cols_lower.get("customername", cols_lower.get("customer name", "CustomerName"))
    sales_col = cols_lower.get("sales man name", cols_lower.get("salesman", cols_lower.get("salesperson", "Sales man name")))

    if name_col not in df.columns or sales_col not in df.columns:
        raise ValueError(
            "Could not find required columns: 'CustomerName' and 'Sales man name'. "
            f"Detected columns: {list(df.columns)}"
        )

    month_cols = detect_month_columns(df)
    if not month_cols:
        raise ValueError("No month columns like 'Jan-23' were found.")

    use_cols = [name_col, sales_col] + month_cols
    df_use = df[use_cols].copy()
    df_use.rename(columns={name_col: "CustomerName", sales_col: "Salesperson"}, inplace=True)

    long_df = longify_sales(df_use, id_cols=["CustomerName", "Salesperson"], month_cols=month_cols)

    # Aggregations (raw sums used only where appropriate)
    total_sc = (
        long_df.groupby(["Salesperson", "CustomerName"], as_index=False)["Sales"].sum()
        .sort_values(["Salesperson", "Sales"], ascending=[True, False])
    )
    yearly_sc = (
        long_df.groupby(["Salesperson", "CustomerName", "Year"], as_index=False)["Sales"].sum()
        .sort_values(["Salesperson", "CustomerName", "Year"])
    )
    monthly_sc = (
        long_df.groupby(["Salesperson", "CustomerName", "Month"], as_index=False)["Sales"].sum()
        .sort_values(["Salesperson", "CustomerName", "Month"])
    )

    years = sorted(long_df["Year"].unique())

    # Raw per-year top 20 (kept for reference; animation will compute annualized on the fly)
    per_year_top20 = (
        yearly_sc
        .sort_values(["Salesperson", "Year", "Sales"], ascending=[True, True, False])
        .groupby(["Salesperson", "Year"], as_index=False, group_keys=False)
        .apply(lambda g: g.head(20))
        .reset_index(drop=True)
    )

    return {
        "raw": df,
        "long": long_df,
        "total_sc": total_sc,
        "yearly_sc": yearly_sc,
        "monthly_sc": monthly_sc,
        "years": years,
        "per_year_top20": per_year_top20,
    }

def compute_top20(total_sc: pd.DataFrame, yearly_sc: pd.DataFrame, salesperson: str,
                  mode: str, year: int | None, long_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Return Top 20 clients for selected salesperson.
    - All years combined: rank by total raw sales across years (unchanged).
    - Specific year: rank by ANNUALIZED sales for that year (so 2025 isn't undercounted).
    """
    if mode == "All years combined":
        top = (total_sc[total_sc["Salesperson"] == salesperson]
               .sort_values("Sales", ascending=False)
               .head(20)
               .reset_index(drop=True))
        return top
    else:
        # Annualized per client for that salesperson-year
        yslice = long_df[(long_df["Salesperson"] == salesperson) & (long_df["Year"] == year)].copy()
        if yslice.empty:
            return yslice.assign(CustomerName=[], Sales=[])
        ann = (
            yslice.groupby("CustomerName", as_index=False)
                  .agg(Sales_sum=("Sales", "sum"),
                       Months=("Month", lambda x: x.dt.month.nunique()))
        )
        ann["Sales"] = np.where(
            ann["Months"] < 12, ann["Sales_sum"] * 12 / ann["Months"], ann["Sales_sum"]
        )
        top = ann.sort_values("Sales", ascending=False).head(20)[["CustomerName", "Sales"]].reset_index(drop=True)
        return top

def compute_yoy(series: pd.Series) -> pd.DataFrame:
    """Given a year-indexed series, return DataFrame with YoY absolute and pct deltas."""
    df = series.sort_index().to_frame("Sales")
    df["YoY Δ"] = df["Sales"].diff()
    df["YoY %"] = (df["YoY Δ"] / df["Sales"].shift(1)) * 100
    return df.reset_index().rename(columns={"index": "Year"})

def compute_cagr(series: pd.Series) -> float | None:
    """CAGR across first and last non-zero years; requires at least 2 points."""
    s = series.sort_index().dropna()
    if len(s) < 2:
        return None
    start_year, end_year = s.index[0], s.index[-1]
    start_val, end_val = float(s.iloc[0]), float(s.iloc[-1])
    periods = end_year - start_year
    if periods <= 0 or start_val <= 0:
        return None
    return (end_val / start_val) ** (1 / periods) - 1.0

# ------------------------
# Main UI
# ------------------------
if not uploaded:
    st.info("Upload an Excel file with columns like **CustomerName**, **Sales man name**, and month columns (e.g., **Jan-23**, **Feb-23** ...).")
    st.stop()

try:
    data = process_file(uploaded.read())
    st.session_state["sales_data"] = data
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

long_df = data["long"]
total_sc = data["total_sc"]
yearly_sc = data["yearly_sc"]
monthly_sc = data["monthly_sc"]
years_all = data["years"]
per_year_top20 = data["per_year_top20"]

st.sidebar.header("2) Filters")
salespeople = total_sc["Salesperson"].dropna().astype(str).unique()
salesperson = st.sidebar.selectbox("Select Salesperson", options=np.sort(salespeople))

mode = st.sidebar.radio(
    "Top 20 Basis",
    options=["All years combined", "Specific year"],
    index=0,
    help="Choose whether Top 20 is computed across all years or for a single year."
)

year_selected = None
if mode == "Specific year":
    year_selected = st.sidebar.selectbox("Year", options=years_all, index=len(years_all)-1)

# Determine Top 20 according to mode/year (annualized for specific year)
top20 = compute_top20(total_sc, yearly_sc, salesperson, mode, year_selected, long_df)

# ------------------------
# Top 20 Table + Bar Chart (Year-Aware)
# ------------------------
title_suffix = "All Years" if mode == "All years combined" else f"Year: {year_selected} (annualized)"
st.subheader(f"🏅 Top 20 Clients — {salesperson} ({title_suffix})")

c1, c2 = st.columns([1.2, 1])
with c1:
    st.dataframe(top20.rename(columns={"Sales": "Total Sales"}), use_container_width=True)

with c2:
    fig_bar = px.bar(
        top20.sort_values("Total Sales" if "Total Sales" in top20.columns else "Sales", ascending=True),
        x="Total Sales" if "Total Sales" in top20.columns else "Sales",
        y="CustomerName",
        orientation="h",
        title=f"Top 20 by Total Sales — {salesperson} ({title_suffix})",
        labels={"Sales": "Total Sales", "CustomerName": "Client"},
        hover_data={"Sales": ":,.2f"}
    )
    fig_bar.update_layout(yaxis={"categoryorder": "total ascending"}, height=650)
    st.plotly_chart(fig_bar, use_container_width=True)

# Download Top 20 slice
csv_top20 = top20.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Top 20 (CSV)",
    csv_top20,
    file_name=f"top20_{salesperson}_{'all_years' if mode=='All years combined' else year_selected}_annualized.csv",
    mime="text/csv",
)

st.markdown("---")

# ------------------------
# Animated per-year Top 20 (ANNUALIZED)
# ------------------------
with st.expander("🎞️ Animated Top 20 per Year (annualized)"):
    st.caption("Watch how the Top 20 clients shift year-to-year (annualized; filtered to the selected salesperson).")
    # Build annualized per-year per-client for this salesperson
    sp_slice = long_df[long_df["Salesperson"] == salesperson].copy()
    if sp_slice.empty:
        st.info("No yearly data available for animation.")
    else:
        ann = (
            sp_slice.groupby(["Year", "CustomerName"], as_index=False)
                    .agg(Sales_sum=("Sales", "sum"),
                         Months=("Month", lambda x: x.dt.month.nunique()))
        )
        ann["AnnualizedSales"] = np.where(
            ann["Months"] < 12, ann["Sales_sum"] * 12 / ann["Months"], ann["Sales_sum"]
        )
        anim_df = (
            ann.sort_values(["Year", "AnnualizedSales"], ascending=[True, False])
               .groupby("Year", as_index=False, group_keys=False)
               .apply(lambda g: g.head(20))
               .reset_index(drop=True)
        )
        fig_anim = px.bar(
            anim_df,
            x="AnnualizedSales",
            y="CustomerName",
            animation_frame="Year",
            orientation="h",
            range_x=[0, anim_df["AnnualizedSales"].max() * 1.1],
            title=f"Animated Top 20 per Year — {salesperson} (annualized)",
            labels={"AnnualizedSales": "Total Sales (Annualized)", "CustomerName": "Client"},
            hover_data={"Sales_sum": ":,.2f", "Months": True, "AnnualizedSales": ":,.2f"}
        )
        fig_anim.update_layout(height=750)
        st.plotly_chart(fig_anim, use_container_width=True)

st.markdown("---")

# ------------------------
# Yearly Performance (for currently selected Top 20) — ANNUALIZED
# ------------------------
st.subheader(f"📅 Yearly Performance — {salesperson} (limited to current Top 20, annualized)")
yr = long_df[
    (long_df["Salesperson"] == salesperson) &
    (long_df["CustomerName"].isin(top20["CustomerName"].tolist()))
].copy()

if yr.empty:
    st.info("No data for the selected scope.")
else:
    yr_annual = (
        yr.groupby(["CustomerName", "Year"])
          .agg(
              Sales_sum=("Sales", "sum"),
              Months=("Month", lambda x: x.dt.month.nunique())
          )
          .reset_index()
    )
    yr_annual["AnnualizedSales"] = np.where(
        yr_annual["Months"] < 12,
        yr_annual["Sales_sum"] * 12 / yr_annual["Months"],
        yr_annual["Sales_sum"]
    )

    fig_year = px.bar(
        yr_annual,
        x="Year",
        y="AnnualizedSales",
        color="CustomerName",
        barmode="group",
        title=f"Yearly Sales by Client — {salesperson} (annualized partial years)",
        labels={"AnnualizedSales": "Total Sales (Annualized)", "Year": "Year", "CustomerName": "Client"},
        hover_data={"Sales_sum": ":,.2f", "Months": True, "AnnualizedSales": ":,.2f"}
    )
    st.plotly_chart(fig_year, use_container_width=True)

    # Download annualized slice
    csv_yr = yr_annual.rename(columns={"AnnualizedSales": "Sales (Annualized)"}).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Yearly Performance (Annualized, CSV)",
        csv_yr,
        file_name=f"yearly_performance_annualized_{salesperson}_{'all_years' if year_selected is None else year_selected}.csv",
        mime="text/csv",
    )

st.markdown("---")

# ------------------------
# Client Improvement Tracker (cross-salesperson view) — ANNUALIZED
# ------------------------
st.subheader("📊 Client Improvement Tracker (across all salespeople, annualized)")
scope = st.radio(
    "Client list scope",
    ["Current Top 20 (above)", "All Clients"],
    index=0,
    horizontal=True,
    help="This view sums a client's sales across all salespeople per year (annualized)."
)

if scope == "Current Top 20 (above)":
    client_pool = top20["CustomerName"].tolist()
else:
    client_pool = sorted(long_df["CustomerName"].dropna().astype(str).unique().tolist())

if len(client_pool) == 0:
    st.info("No clients available for the chosen scope.")
else:
    client_sel = st.selectbox("Select a client", client_pool)

    # Yearly totals for this client across all salespeople (annualized)
    client_slice = long_df[long_df["CustomerName"] == client_sel].copy()
    client_year_annual = (
        client_slice.groupby("Year")
          .agg(
              Sales_sum=("Sales", "sum"),
              Months=("Month", lambda x: x.dt.month.nunique())
          )
          .reset_index()
    )
    client_year_annual["AnnualizedSales"] = np.where(
        client_year_annual["Months"] < 12,
        client_year_annual["Sales_sum"] * 12 / client_year_annual["Months"],
        client_year_annual["Sales_sum"]
    )
    client_year_totals = client_year_annual.set_index("Year")["AnnualizedSales"].sort_index()

    # Yearly split by salesperson (annualized)
    client_by_sp = (
        client_slice.groupby(["Year", "Salesperson"])
          .agg(
              Sales_sum=("Sales", "sum"),
              Months=("Month", lambda x: x.dt.month.nunique())
          )
          .reset_index()
    )
    client_by_sp["AnnualizedSales"] = np.where(
        client_by_sp["Months"] < 12,
        client_by_sp["Sales_sum"] * 12 / client_by_sp["Months"],
        client_by_sp["Sales_sum"]
    )

    colA, colB, colC = st.columns(3)
    # Latest YoY metric (based on annualized totals)
    yoy_df = compute_yoy(client_year_totals)
    if len(yoy_df) >= 2:
        last_row = yoy_df.iloc[-1]
        delta = float(last_row["YoY Δ"]) if pd.notna(last_row["YoY Δ"]) else 0.0
        pct = float(last_row["YoY %"]) if pd.notna(last_row["YoY %"]) else 0.0
        colA.metric("Latest YoY Δ", f"{delta:,.0f}", f"{pct:+.2f}%")
    else:
        colA.metric("Latest YoY Δ", "—", "—")

    # 3-year CAGR (based on annualized totals)
    cagr = compute_cagr(client_year_totals)
    if cagr is not None:
        colB.metric("CAGR (multi-year)", f"{cagr*100:.2f}%")
    else:
        colB.metric("CAGR (multi-year)", "—")

    # Best / Worst year
    if len(client_year_totals) > 0:
        best_year = int(client_year_totals.idxmax())
        worst_year = int(client_year_totals.idxmin())
        colC.metric("Best / Worst Year", f"{best_year} / {worst_year}")
    else:
        colC.metric("Best / Worst Year", "—")

    # Total by year (aggregated across all salespeople, annualized)
    fig_client_total = px.line(
        client_year_annual,
        x="Year",
        y="AnnualizedSales",
        markers=True,
        title=f"Total Sales by Year — {client_sel} (All Salespeople, annualized)",
        labels={"AnnualizedSales": "Total Sales (Annualized)", "Year": "Year"},
        hover_data={"Sales_sum": ":,.2f", "Months": True, "AnnualizedSales": ":,.2f"}
    )
    st.plotly_chart(fig_client_total, use_container_width=True)

    # Yearly split by salesperson (stacked, annualized)
    if client_by_sp.empty:
        st.info("No yearly breakdown by salesperson for this client.")
    else:
        fig_split = px.bar(
            client_by_sp,
            x="Year",
            y="AnnualizedSales",
            color="Salesperson",
            title=f"Yearly Split by Salesperson — {client_sel} (annualized)",
            labels={"AnnualizedSales": "Total Sales (Annualized)", "Year": "Year", "Salesperson": "Salesperson"},
            hover_data={"Sales_sum": ":,.2f", "Months": True, "AnnualizedSales": ":,.2f"},
        )
        st.plotly_chart(fig_split, use_container_width=True)

        # Contribution table (who served which year and their share)
        contrib = client_by_sp.copy()
        totals_per_year = contrib.groupby("Year")["AnnualizedSales"].transform("sum")
        contrib["Share %"] = (contrib["AnnualizedSales"] / totals_per_year) * 100
        st.dataframe(
            contrib.rename(columns={"AnnualizedSales": "Sales (Annualized Year)", "Salesperson": "Handled By"})[
                ["Year", "Handled By", "Sales (Annualized Year)", "Share %"]
            ],
            use_container_width=True
        )

    # Downloads for this client view (annualized)
    st.download_button(
        f"⬇️ Download Client Yearly Totals (Annualized, CSV) — {client_sel}",
        client_year_annual.rename(columns={"AnnualizedSales": "Sales (Annualized)"}).to_csv(index=False).encode("utf-8"),
        file_name=f"client_yearly_totals_annualized_{re.sub(r'[^A-Za-z0-9]+','_', client_sel)}.csv",
        mime="text/csv",
    )
    if not client_by_sp.empty:
        st.download_button(
            f"⬇️ Download Yearly Split by Salesperson (Annualized, CSV) — {client_sel}",
            client_by_sp.to_csv(index=False).encode("utf-8"),
            file_name=f"client_yearly_split_annualized_{re.sub(r'[^A-Za-z0-9]+','_', client_sel)}.csv",
            mime="text/csv",
        )

st.markdown("> **Tip:** Annualization: (sum of months so far ÷ number of months) × 12. Hover tooltips show raw months used. Monthly section remains raw (no annualization).")

# ------------------------
# Client Trend (Monthly) — 3 lines (one per year) overlaid by month (RAW MONTHLY, UNCHANGED)
# ------------------------
st.markdown("---")
st.subheader(f"📈 Client Trend (Monthly by Year Overlay) — {salesperson}")

client_options = top20["CustomerName"].tolist()
client = st.selectbox("Choose a client (from current Top 20)", client_options, key="client_trend_multi_year")

mth_all = monthly_sc[
    (monthly_sc["Salesperson"] == salesperson) &
    (monthly_sc["CustomerName"] == client)
].copy()

if mth_all.empty:
    st.info("No monthly data for this client.")
else:
    years_avail = sorted(mth_all["Month"].dt.year.unique())
    last_three_years = years_avail[-3:] if len(years_avail) >= 3 else years_avail
    mth = mth_all[mth_all["Month"].dt.year.isin(last_three_years)].copy()
    mth["Year"] = mth["Month"].dt.year.astype(int)
    mth["MonthNum"] = mth["Month"].dt.month
    mth["MonthLabel"] = mth["Month"].dt.strftime("%b")
    month_order = list(range(1, 13))
    mth = mth.sort_values(["Year", "MonthNum"])
    st.caption(f"Showing years: {', '.join(str(y) for y in last_three_years)}")

    fig_overlay = px.line(
        mth,
        x="MonthNum",
        y="Sales",
        color="Year",
        markers=True,
        line_group="Year",
        hover_data={"Sales": ":,.2f", "MonthNum": False, "MonthLabel": True},
        title=f"Monthly Trend Overlay — {client} ({salesperson})"
    )
    fig_overlay.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=month_order,
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        ),
        yaxis_title="Sales Amount",
        xaxis_title="Month",
        legend_title_text="Year",
        height=500
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    yearly_client_subset = (
        mth.groupby("Year", as_index=False)["Sales"].sum()
        .sort_values("Year")
    )
    fig_year_bars = px.bar(
        yearly_client_subset,
        x="Year",
        y="Sales",
        title=f"Yearly Totals — {client} (Shown Years)",
        hover_data={"Sales": ":,.2f"}
    )
    st.plotly_chart(fig_year_bars, use_container_width=True)

    csv_overlay = mth[["Year", "MonthNum", "MonthLabel", "Sales"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download Overlay Data (CSV) — {client}",
        csv_overlay,
        file_name=f"monthly_overlay_{re.sub(r'[^A-Za-z0-9]+','_', client)}.csv",
        mime="text/csv",
    )

# ------------------------
# 📉 Losses in Sales — Latest Month (per salesperson)
# ------------------------
st.markdown("---")
st.subheader(f"📉 Losses in Sales — Latest Month (MoM) — {salesperson}")

# Helper: build a quick monthly index for lookup
monthly_idx = monthly_sc.copy()
monthly_idx["Year"] = monthly_idx["Month"].dt.year
monthly_idx["MonthNum"] = monthly_idx["Month"].dt.month

# 1) Identify latest month in the dataset
latest_month = long_df["Month"].max()
if pd.isna(latest_month):
    st.info("No monthly data available.")
else:
    # 2) Slice latest and previous month for this salesperson
    prev_month = latest_month - pd.DateOffset(months=1)

    latest_slice = monthly_sc[
        (monthly_sc["Salesperson"] == salesperson) &
        (monthly_sc["Month"] == latest_month)
    ][["CustomerName", "Sales"]].rename(columns={"Sales": "LatestMonthSales"})

    prev_slice = monthly_sc[
        (monthly_sc["Salesperson"] == salesperson) &
        (monthly_sc["Month"] == prev_month)
    ][["CustomerName", "Sales"]].rename(columns={"Sales": "PrevMonthSales"})

    # 3) Join and compute deltas
    comp = latest_slice.merge(prev_slice, on="CustomerName", how="left")
    comp["MoM Δ"] = comp["LatestMonthSales"].fillna(0) - comp["PrevMonthSales"].fillna(0)
    comp["MoM %"] = np.where(
        comp["PrevMonthSales"].fillna(0) != 0,
        (comp["MoM Δ"] / comp["PrevMonthSales"].replace(0, np.nan)) * 100,
        np.nan
    )

    # 4) Keep only losses (negative deltas)
    losses = comp[comp["MoM Δ"] < 0].copy()
    losses = losses.sort_values("MoM Δ")  # most negative first

    # Pretty titles
    lm_label = latest_month.strftime("%b %Y")
    pm_label = prev_month.strftime("%b %Y")

    if losses.empty:
        st.success(f"No MoM losses found for {salesperson} in {lm_label}.")
    else:
        st.caption(f"Comparing **{lm_label}** vs **{pm_label}** (raw monthly).")
        # Reorder/rename for display
        show_cols = ["CustomerName", "LatestMonthSales", "PrevMonthSales", "MoM Δ", "MoM %"]
        st.dataframe(
            losses[show_cols].rename(columns={
                "CustomerName": "Client",
                "LatestMonthSales": f"Sales ({lm_label})",
                "PrevMonthSales": f"Sales ({pm_label})"
            }),
            use_container_width=True
        )

        # Optional: download
        st.download_button(
            "⬇️ Download Losses (CSV)",
            losses.rename(columns={
                "CustomerName": "Client",
                "LatestMonthSales": f"Sales ({lm_label})",
                "PrevMonthSales": f"Sales ({pm_label})"
            })[["Client", f"Sales ({lm_label})", f"Sales ({pm_label})", "MoM Δ", "MoM %"]]
            .to_csv(index=False).encode("utf-8"),
            file_name=f"losses_{salesperson}_{latest_month.strftime('%Y_%m')}.csv",
            mime="text/csv",
        )

        st.markdown("### 🔍 History snippet for a selected losing client")
        sel_client = st.selectbox(
            "Pick a client from the losses list to see historical context",
            losses["CustomerName"].tolist()
        )

        # 5) For the selected client, show previous years' month-before, month, and month-after
        #    relative to the latest month (raw monthly, no annualization).
        def _history_three_months(month_anchor: pd.Timestamp, year: int) -> dict:
            """For a given 'year', build prev/anchor/next month timestamps relative to anchor's month."""
            # Anchor set to the same month of the given year
            anchor = pd.Timestamp(year=year, month=month_anchor.month, day=1)
            prev_m = anchor - pd.DateOffset(months=1)
            next_m = anchor + pd.DateOffset(months=1)
            return {"prev": prev_m, "anchor": anchor, "next": next_m}

        # Build quick lookup for (salesperson, client, exact month)
        # monthly_sc already has Month as datetime (1st-of-month). We'll sum just in case duplicates exist.
        lookup = (monthly_sc[monthly_sc["Salesperson"] == salesperson]
                    .groupby(["CustomerName", "Month"], as_index=False)["Sales"]
                    .sum())

        latest_year = int(latest_month.year)
        prior_years = sorted(y for y in lookup["Month"].dt.year.unique() if y < latest_year)
        if not prior_years:
            st.info("No prior years available for this client.")
        else:
            rows = []
            for y in prior_years:
                t = _history_three_months(latest_month, y)

                def _val(ts):
                    row = lookup[(lookup["CustomerName"] == sel_client) & (lookup["Month"] == ts)]
                    return float(row["Sales"].iloc[0]) if not row.empty else 0.0

                prev_v  = _val(t["prev"])
                anch_v  = _val(t["anchor"])
                next_v  = _val(t["next"])

                rows.append({
                    "Year": y,
                    f"{t['prev'].strftime('%b')}": prev_v,
                    f"{t['anchor'].strftime('%b')}": anch_v,
                    f"{t['next'].strftime('%b')}": next_v,
                })

            hist_df = pd.DataFrame(rows).sort_values("Year")
            # Keep consistent 3 columns: month-before, anchor-month, month-after
            # Determine the three month labels once from latest anchor
            prev_label  = (latest_month - pd.DateOffset(months=1)).strftime("%b")
            anch_label  = latest_month.strftime("%b")
            next_label  = (latest_month + pd.DateOffset(months=1)).strftime("%b")

            # Rename columns to fixed labels (handles year-crossing edge cases cleanly)
            # Rebuild with those fixed labels:
            rebuilt = []
            for _, r in hist_df.iterrows():
                y = int(r["Year"])
                # compute again to fetch by fixed labels
                t = _history_three_months(latest_month, y)
                row = {
                    "Year": y,
                    prev_label: lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["prev"])
                    ]["Sales"].sum() if not lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["prev"])
                    ].empty else 0.0,
                    anch_label: lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["anchor"])
                    ]["Sales"].sum() if not lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["anchor"])
                    ].empty else 0.0,
                    next_label: lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["next"])
                    ]["Sales"].sum() if not lookup[
                        (lookup["CustomerName"] == sel_client) & (lookup["Month"] == t["next"])
                    ].empty else 0.0,
                }
                rebuilt.append(row)

            hist_fixed = pd.DataFrame(rebuilt).sort_values("Year")
            st.caption(f"Historical view for **{sel_client}** (per **{salesperson}**) — showing **{prev_label} / {anch_label} / {next_label}** for each prior year.")
            st.dataframe(hist_fixed, use_container_width=True)

            # Bar chart for the anchor month across years (quick visual)
            fig_hist = px.bar(
                hist_fixed,
                x="Year",
                y=anch_label,
                title=f"{sel_client} — {anch_label} across years (per {salesperson})",
                hover_data={anch_label: ":,.2f"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
