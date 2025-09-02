# pages/04_Losses_All_Salespeople.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Latest Month Losses — All Salespeople", layout="wide")
st.title("📉 Latest Month Losses — All Salespeople")

# --- Pull processed data from session_state ---
if "sales_data" not in st.session_state:
    st.info("Please go back to the main page, upload your Excel, and return here.")
    st.stop()

data = st.session_state["sales_data"]
long_df      = data["long"]         # CustomerName, Salesperson, Month(datetime), Year, Sales
monthly_sc   = data["monthly_sc"]   # grouped monthly (CustomerName, Salesperson, Month, Sales)
total_sc     = data["total_sc"]

# --- Identify latest month across the dataset ---
latest_month = long_df["Month"].max()
if pd.isna(latest_month):
    st.info("No monthly data available.")
    st.stop()

prev_month = latest_month - pd.DateOffset(months=1)
lm_label   = latest_month.strftime("%b %Y")
pm_label   = prev_month.strftime("%b %Y")

# --- Build latest vs previous month slices across ALL salespeople ---
latest_slice = (
    monthly_sc[monthly_sc["Month"] == latest_month]
    .groupby(["Salesperson", "CustomerName"], as_index=False)["Sales"].sum()
    .rename(columns={"Sales": "LatestMonthSales"})
)

prev_slice = (
    monthly_sc[monthly_sc["Month"] == prev_month]
    .groupby(["Salesperson", "CustomerName"], as_index=False)["Sales"].sum()
    .rename(columns={"Sales": "PrevMonthSales"})
)

comp = latest_slice.merge(prev_slice, on=["Salesperson", "CustomerName"], how="left")
comp["PrevMonthSales"] = comp["PrevMonthSales"].fillna(0.0)
comp["MoM Δ"] = comp["LatestMonthSales"] - comp["PrevMonthSales"]
comp["MoM %"] = np.where(
    comp["PrevMonthSales"] != 0,
    (comp["MoM Δ"] / comp["PrevMonthSales"]) * 100,
    np.nan
)

# --- Losses only (negative MoM) ---
losses = comp[comp["MoM Δ"] < 0].copy().sort_values("MoM Δ")

st.caption(f"Comparing **{lm_label}** vs **{pm_label}** (raw monthly, across all salespeople).")

# Sidebar filters
st.sidebar.header("Filters")
sp_options = ["All"] + sorted(losses["Salesperson"].dropna().astype(str).unique().tolist())
sp_pick = st.sidebar.selectbox("Salesperson", sp_options, index=0)

min_drop_abs = st.sidebar.number_input("Min absolute drop (MoM Δ ≤ …)", value=0.0, step=1000.0, format="%.2f")
min_drop_pct = st.sidebar.number_input("Min % drop (MoM % ≤ …)", value=0.0, step=1.0, format="%.2f")

filt = losses.copy()
if sp_pick != "All":
    filt = filt[filt["Salesperson"] == sp_pick]
if min_drop_abs > 0:
    filt = filt[filt["MoM Δ"] <= -abs(min_drop_abs)]
if min_drop_pct > 0:
    # negative percentage at least as low as -min_drop_pct
    filt = filt[(~filt["MoM %"].isna()) & (filt["MoM %"] <= -abs(min_drop_pct))]

# Display table
display_cols = ["Salesperson", "CustomerName", "LatestMonthSales", "PrevMonthSales", "MoM Δ", "MoM %"]
st.dataframe(
    filt.rename(columns={
        "CustomerName": "Client",
        "LatestMonthSales": f"Sales ({lm_label})",
        "PrevMonthSales": f"Sales ({pm_label})"
    })[["Salesperson", "Client", f"Sales ({lm_label})", f"Sales ({pm_label})", "MoM Δ", "MoM %"]],
    use_container_width=True
)

# Download
st.download_button(
    "⬇️ Download Losses (CSV)",
    filt.rename(columns={
        "CustomerName": "Client",
        "LatestMonthSales": f"Sales ({lm_label})",
        "PrevMonthSales": f"Sales ({pm_label})"
    })[["Salesperson", "Client", f"Sales ({lm_label})", f"Sales ({pm_label})", "MoM Δ", "MoM %"]]
    .to_csv(index=False).encode("utf-8"),
    file_name=f"losses_all_salespeople_{latest_month.strftime('%Y_%m')}.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("🔎 Deeper Dive — Client Snapshot")

# --- Choose a row to dive into ---
# Limit choices to the current filtered losses for a clean flow
if filt.empty:
    st.info("No rows match the current filters.")
    st.stop()



dd_sp = st.selectbox("Salesperson", sorted(filt["Salesperson"].unique().tolist()))

# --- High-level metrics for salesperson (latest month) ---
st.markdown("### 🔑 Salesperson Performance Snapshot")

anchor_month = latest_month
curr_year = anchor_month.year
last_year = curr_year - 1
prev_month = anchor_month - pd.DateOffset(months=1)

# --- Total sales this salesperson, latest month & prev month
sp_latest = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) & (monthly_sc["Month"] == anchor_month)
]["Sales"].sum()

sp_prev = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) & (monthly_sc["Month"] == prev_month)
]["Sales"].sum()

# --- This year avg (to date)
this_year_sales = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["Month"].dt.year == curr_year) &
    (monthly_sc["Month"] <= anchor_month)
]["Sales"]

avg_this_year = this_year_sales.sum() / this_year_sales.count() if not this_year_sales.empty else 0

# --- Last year avg (all months, or clip to same month if you prefer)
last_year_sales = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["Month"].dt.year == last_year)
]["Sales"]

avg_last_year = last_year_sales.sum() / last_year_sales.count() if not last_year_sales.empty else 0

# --- Build metrics row
col1, col2, col3, col4 = st.columns(4)

# MoM delta
mom_delta = sp_latest - sp_prev
col1.metric(
    f"{anchor_month.strftime('%b %Y')} Sales",
    f"{sp_latest:,.0f}",
    f"{mom_delta:+,.0f} vs prev mo" if sp_prev > 0 else "—"
)

# Vs this year avg
delta_vs_this = sp_latest - avg_this_year
pct_vs_this = (delta_vs_this / avg_this_year * 100) if avg_this_year > 0 else None
col2.metric(
    f"Vs {curr_year} avg (to-date)",
    f"{delta_vs_this:+,.0f}",
    f"{pct_vs_this:+.1f}%" if pct_vs_this is not None else "—"
)

# Vs last year avg
delta_vs_last = sp_latest - avg_last_year
pct_vs_last = (delta_vs_last / avg_last_year * 100) if avg_last_year > 0 else None
col3.metric(
    f"Vs {last_year} avg",
    f"{delta_vs_last:+,.0f}",
    f"{pct_vs_last:+.1f}%" if pct_vs_last is not None else "—"
)

# Direction summary
trend_icon = "⬆️" if mom_delta > 0 else ("⬇️" if mom_delta < 0 else "➡️")
col4.metric("Overall Trend", trend_icon, None)


# --- Overall Sales This Year (YTD) for selected salesperson ---
st.markdown("### 🧮 Overall Sales — This Year (YTD)")

# Slice current year up to the latest available month for this salesperson
ytd_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["Month"].dt.year == curr_year) &
    (monthly_sc["Month"] <= anchor_month)
].copy()

if ytd_slice.empty:
    st.info(f"No {curr_year} data found for {dd_sp}.")
else:
    # YTD total
    ytd_total = float(ytd_slice["Sales"].sum())

    # Last year YTD (up to the same anchor month) — optional but handy
    lytd_slice = monthly_sc[
        (monthly_sc["Salesperson"] == dd_sp) &
        (monthly_sc["Month"].dt.year == last_year) &
        (monthly_sc["Month"].dt.month <= anchor_month.month)
    ].copy()
    lytd_total = float(lytd_slice["Sales"].sum()) if not lytd_slice.empty else 0.0

    delta_vs_lytd = ytd_total - lytd_total
    pct_vs_lytd = (delta_vs_lytd / lytd_total * 100.0) if lytd_total > 0 else None

    cY, cT, cD = st.columns(3)
    cY.metric("Year", f"{curr_year}")
    cT.metric(f"YTD Sales (Jan–{anchor_month.strftime('%b')})", f"{ytd_total:,.0f}")
    cD.metric(
        f"Vs {last_year} YTD",
        f"{delta_vs_lytd:+,.0f}",
        f"{pct_vs_lytd:+.1f}%" if pct_vs_lytd is not None else "—"
    )

    # --- Monthly breakdowns (Jan..anchor_month) for current year and last year ---
ytd_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["Month"].dt.year == curr_year) &
    (monthly_sc["Month"] <= anchor_month)
].copy()

lytd_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["Month"].dt.year == last_year) &
    (monthly_sc["Month"].dt.month <= anchor_month.month)
].copy()

# Prepare month columns
for df in (ytd_slice, lytd_slice):
    if not df.empty:
        df["MonthNum"] = df["Month"].dt.month
        df["MonthLabel"] = df["Month"].dt.strftime("%b")

# Aggregate to month level
ytd_monthly = (
    ytd_slice.groupby(["MonthNum", "MonthLabel"], as_index=False)["Sales"].sum()
             .sort_values("MonthNum")
)
lytd_monthly = (
    lytd_slice.groupby(["MonthNum", "MonthLabel"], as_index=False)["Sales"].sum()
              .sort_values("MonthNum")
)

# Build a combined long-form dataframe to plot both lines
ytd_monthly["YearLabel"] = f"{curr_year} YTD"
lytd_monthly["YearLabel"] = f"{last_year} YTD"

combined = pd.concat([ytd_monthly, lytd_monthly], ignore_index=True)

# If last year has no data, keep just current year
if combined.empty:
    st.info(f"No {curr_year} or {last_year} month-level data found for {dd_sp}.")
else:
    import plotly.express as px
    fig_ytd = px.line(
        combined,
        x="MonthNum",
        y="Sales",
        color="YearLabel",
        markers=True,
        hover_data={"Sales": ":,.2f", "MonthNum": False, "MonthLabel": True},
        title=f"{dd_sp} — YTD Sales by Month (through {anchor_month.strftime('%b')})"
    )
    # X-axis ticks: Jan..current anchor month
    tick_nums = list(range(1, anchor_month.month + 1))
    # Use the month labels from the current year if available, else from last year
    if not ytd_monthly.empty:
        tick_text = ytd_monthly["MonthLabel"].tolist()
    else:
        tick_text = lytd_monthly["MonthLabel"].tolist()

    fig_ytd.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=tick_nums,
            ticktext=tick_text
        ),
        xaxis_title="Month",
        yaxis_title="Sales Amount",
        legend_title_text="Series",
        height=420
    )
    st.plotly_chart(fig_ytd, use_container_width=True)

    # Download combined YTD data
    st.download_button(
        "⬇️ Download YTD (current vs last year) — CSV",
        combined.rename(columns={"Sales": "Sales (YTD)"}).to_csv(index=False).encode("utf-8"),
        file_name=f"ytd_vs_last_{dd_sp}_{curr_year}_through_{anchor_month.strftime('%b')}.csv",
        mime="text/csv",
    )




dd_client = st.selectbox(
        "Client",
        sorted(filt[filt["Salesperson"] == dd_sp]["CustomerName"].unique().tolist())
    )

anchor_month = latest_month
anchor_label = anchor_month.strftime("%b")  # e.g., "Aug"

# Helper: fetch monthly value for (sp, client, specific month)
def month_value(sp: str, client: str, ts: pd.Timestamp) -> float:
    rows = monthly_sc[
        (monthly_sc["Salesperson"] == sp) &
        (monthly_sc["CustomerName"] == client) &
        (monthly_sc["Month"] == pd.Timestamp(ts.year, ts.month, 1))
    ]["Sales"]
    return float(rows.sum()) if not rows.empty else 0.0

# Current month value
v_now = month_value(dd_sp, dd_client, anchor_month)

# --- Current year average up to latest month (same salesperson+client) ---
this_year = int(anchor_month.year)
this_year_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_client) &
    (monthly_sc["Month"].dt.year == this_year) &
    (monthly_sc["Month"] <= anchor_month)
].copy()

if this_year_slice.empty:
    avg_this_year = 0.0
    months_this_year = 0
else:
    months_this_year = this_year_slice["Month"].dt.month.nunique()
    avg_this_year = this_year_slice["Sales"].sum() / max(months_this_year, 1)

# --- Last year's average (all months available in that year) ---
last_year = this_year - 1
last_year_slice = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_client) &
    (monthly_sc["Month"].dt.year == last_year)
].copy()

if last_year_slice.empty:
    avg_last_year = 0.0
    months_last_year = 0
else:
    months_last_year = last_year_slice["Month"].dt.month.nunique()
    avg_last_year = last_year_slice["Sales"].sum() / max(months_last_year, 1)

# --- Same month last year & two years ago ---
v_last_year_same_month = month_value(dd_sp, dd_client, anchor_month - pd.DateOffset(years=1))
v_two_years_same_month = month_value(dd_sp, dd_client, anchor_month - pd.DateOffset(years=2))

# --- Metrics row ---
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{anchor_label} {this_year} Sales", f"{v_now:,.0f}")
m2.metric(f"Vs this-year monthly avg\n({months_this_year} mo)", f"{(v_now - avg_this_year):,.0f}", f"{((v_now - avg_this_year)/avg_this_year*100):+.2f}%"
          if avg_this_year else "—")
m3.metric(f"Vs last-year avg\n({months_last_year} mo)", f"{(v_now - avg_last_year):,.0f}", f"{((v_now - avg_last_year)/avg_last_year*100):+.2f}%"
          if avg_last_year else "—")
m4.metric(f"Same month YoY", f"{(v_now - v_last_year_same_month):,.0f}", f"{((v_now - v_last_year_same_month)/v_last_year_same_month*100):+.2f}%"
          if v_last_year_same_month else "—")

# --- Comparison table (quick view) ---
summary_rows = [
    ["Salesperson", dd_sp],
    ["Client", dd_client],
    [f"{anchor_label} {this_year}", v_now],
    [f"Avg / mo — {this_year}", avg_this_year],
    [f"Avg / mo — {last_year}", avg_last_year],
    [f"{anchor_label} {last_year}", v_last_year_same_month],
    [f"{anchor_label} {this_year-2}", v_two_years_same_month],
]
st.table(pd.DataFrame(summary_rows, columns=["Metric", "Value"]))



# --- Mini trend: split by year (last 3 years) with dashed averages ---
growth_pct = st.number_input(
    "Growth % to apply on last year's average",
    value=0.0, step=1.0, format="%.2f"
)

trend_all = monthly_sc[
    (monthly_sc["Salesperson"] == dd_sp) &
    (monthly_sc["CustomerName"] == dd_client)
].copy()

trend_all["Year"] = trend_all["Month"].dt.year
trend_all["MonthNum"] = trend_all["Month"].dt.month
trend_all["MonthLabel"] = trend_all["Month"].dt.strftime("%b")

# Keep last 3 years dynamically (or fewer if not enough data)
years_sorted = sorted(trend_all["Year"].unique())
years_keep = years_sorted[-3:] if len(years_sorted) > 3 else years_sorted
trend_all = trend_all[trend_all["Year"].isin(years_keep)]

# Aggregate in case multiple rows exist for the same month
trend_all = (
    trend_all.groupby(["Year", "MonthNum", "MonthLabel"], as_index=False)["Sales"]
    .sum()
    .sort_values(["Year", "MonthNum"])
)

fig_line_years = px.line(
    trend_all,
    x="MonthNum",
    y="Sales",
    color="Year",
    markers=True,
    line_group="Year",
    hover_data={"Sales": ":,.2f", "MonthLabel": True},
    title=f"Monthly Sales by Year — {dd_client} (handled by {dd_sp})"
)

fig_line_years.update_layout(
    xaxis=dict(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ),
    yaxis_title="Sales Amount",
    xaxis_title="Month",
    legend_title_text="Year",
    height=500
)

# --- Add dashed lines for averages ---
anchor_month = latest_month
current_year = int(anchor_month.year)
last_year = current_year - 1

def year_avg(y: int) -> float:
    dfy = trend_all[trend_all["Year"] == y].copy()
    if y == current_year:
        # average up to the anchor month only
        dfy = dfy[dfy["MonthNum"] <= anchor_month.month]
    months = dfy["MonthNum"].nunique()
    return float(dfy["Sales"].sum() / months) if months else 0.0

avg_curr = year_avg(current_year)
avg_last = year_avg(last_year)
avg_last_proj = avg_last * (1.0 + growth_pct / 100.0)

x_vals = list(range(1, 13))

if avg_curr > 0:
    fig_line_years.add_scatter(
        x=x_vals, y=[avg_curr]*12,
        mode="lines",
        name=f"{current_year} avg (to-date)",
        line=dict(dash="dash"),
        hovertemplate="%{y:,.2f}<extra></extra>"  # keep %{y} for Plotly
    )

if avg_last_proj > 0:
    fig_line_years.add_scatter(
        x=x_vals, y=[avg_last_proj]*12,
        mode="lines",
        name=f"{last_year} avg × (1+{growth_pct:.2f}%)",
        line=dict(dash="dot"),
        hovertemplate="%{y:,.2f}<extra></extra>"
    )


st.plotly_chart(fig_line_years, use_container_width=True)
