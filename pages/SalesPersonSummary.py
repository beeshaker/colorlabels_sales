# pages/02_Salesperson_Growth.py
# New page: Salesperson YoY Growth % (uses the file uploaded on main page via st.session_state)
import io
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Salesperson YoY Growth %", layout="wide")
st.title("📈 Salespersons — Year-over-Year Growth (%)")

# ------------------------
# Helpers (self-contained)
# ------------------------
MONTH_HEADER_RE = re.compile(r"^[A-Za-z]{3}-\d{2}$")  # e.g., Jan-23

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    def _clean(c):
        if isinstance(c, str):
            c = c.strip().replace("Sum of Sum of ", "").replace("Sum of ", "")
        return c
    return df.rename(columns=_clean)

def detect_month_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        if isinstance(c, str) and MONTH_HEADER_RE.match(c) and ("YTD" not in c and "Projected" not in c):
            cols.append(c)
    return cols

def longify_sales(df: pd.DataFrame, id_cols: list, month_cols: list) -> pd.DataFrame:
    melted = df.melt(id_vars=id_cols, value_vars=month_cols,
                     var_name="MonthStr", value_name="Sales")
    melted = melted.dropna(subset=["Sales"])
    melted["Month"] = pd.to_datetime(melted["MonthStr"], format="%b-%y", errors="coerce")
    melted = melted.dropna(subset=["Month"])
    melted["Year"] = melted["Month"].dt.year.astype(int)
    return melted

def annualize_group(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """
    Annualize at the group level (e.g., per salesperson per year).
    AnnualizedSales = (sum Sales) * 12 / (# distinct months) if months < 12 else sum.
    """
    g = (
        df.groupby(group_cols)
          .agg(Sales_sum=("Sales", "sum"),
               Months=("Month", lambda x: x.dt.month.nunique()))
          .reset_index()
    )
    g["AnnualizedSales"] = np.where(g["Months"] < 12, g["Sales_sum"] * 12 / g["Months"], g["Sales_sum"])
    return g

@st.cache_data(show_spinner=False)
def load_long_df(file_bytes: bytes) -> pd.DataFrame:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=xls.sheet_names[0])
    df = clean_columns(df)

    cols_lower = {c.lower(): c for c in df.columns if isinstance(c, str)}
    name_col = cols_lower.get("customername", cols_lower.get("customer name", "CustomerName"))
    sales_col = cols_lower.get("sales man name", cols_lower.get("salesman", cols_lower.get("salesperson", "Sales man name")))

    if name_col not in df.columns or sales_col not in df.columns:
        raise ValueError("Missing required columns: 'CustomerName' and 'Sales man name'.")

    month_cols = detect_month_columns(df)
    if not month_cols:
        raise ValueError("No month columns like 'Jan-23' were found.")

    use_cols = [name_col, sales_col] + month_cols
    df_use = df[use_cols].copy()
    df_use.rename(columns={name_col: "CustomerName", sales_col: "Salesperson"}, inplace=True)

    long_df = longify_sales(df_use, id_cols=["CustomerName", "Salesperson"], month_cols=month_cols)
    return long_df

# ------------------------
# Data intake — reuse upload from main page if available
# ------------------------
file_bytes = st.session_state.get("uploaded_file_bytes", None)

st.sidebar.header("Data source")
if file_bytes is None:
    st.info("No file found from the main page. Upload here or go back to the main page to upload once.")
    uploaded = st.sidebar.file_uploader("Upload the Excel (.xlsx)", type=["xlsx"], key="uploader_growth_fallback")
    if uploaded:
        file_bytes = uploaded.read()
        st.session_state["uploaded_file_bytes"] = file_bytes  # persist for other pages

if file_bytes is None:
    st.stop()

try:
    long_df = load_long_df(file_bytes)
except Exception as e:
    st.error(f"Error processing file: {e}")
    st.stop()

years_all = sorted(long_df["Year"].unique())
salespeople = sorted(long_df["Salesperson"].dropna().astype(str).unique())

# ------------------------
# Build annualized totals per Salesperson per Year (so partial 2025 is comparable)
# ------------------------
sp_year = annualize_group(long_df, ["Year", "Salesperson"])  # columns: Year, Salesperson, Sales_sum, Months, AnnualizedSales

# Compute YoY growth per salesperson (based on *their own* annualized totals)
sp_year = sp_year.sort_values(["Salesperson", "Year"])
sp_year["PrevAnnualized"] = sp_year.groupby("Salesperson")["AnnualizedSales"].shift(1)

# Growth % = (curr / prev - 1) * 100  ; if prev<=0 or NaN -> NaN (can't compute)
valid_prev = (sp_year["PrevAnnualized"].notna()) & (sp_year["PrevAnnualized"] > 0)
sp_year["Growth %"] = np.where(valid_prev, (sp_year["AnnualizedSales"] / sp_year["PrevAnnualized"] - 1) * 100, np.nan)
sp_year["Abs Δ"] = np.where(valid_prev, sp_year["AnnualizedSales"] - sp_year["PrevAnnualized"], np.nan)

# ------------------------
# Controls
# ------------------------
cA, cB = st.columns([1, 1])
with cA:
    year_focus = st.selectbox("Focus Year (leaderboard, bars & YoY table)", options=years_all, index=len(years_all)-1)
with cB:
    pick_people = st.multiselect("Filter Salespeople (optional)", options=salespeople, default=salespeople)

filtered = sp_year[sp_year["Salesperson"].isin(pick_people)].copy()

# ------------------------
# 1) Leaderboard — Best & Worst Growth % for the selected year
# ------------------------
st.subheader(f"🏁 Leaderboard — YoY Growth % in {year_focus} (annualized)")
yr_slice = filtered[filtered["Year"] == year_focus].copy()
yr_slice = yr_slice.sort_values("Growth %", ascending=False)

if yr_slice["Growth %"].notna().sum() == 0:
    st.info("No valid YoY comparison for the selected year (need a prior year per salesperson).")
else:
    # best/worst among those with valid growth values
    valid = yr_slice[yr_slice["Growth %"].notna()]
    best_row = valid.iloc[0]
    worst_row = valid.iloc[-1]

    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    c1.metric("⭐ Best Performing", f"{best_row['Salesperson']}", f"{best_row['Growth %']:.2f}%")
    c2.metric("⚠️ Worst Performing", f"{worst_row['Salesperson']}", f"{worst_row['Growth %']:.2f}%")
    c3.caption("Growth % is relative to each salesperson's *previous* year annualized sales.")

    # table
    tbl = valid[["Salesperson", "PrevAnnualized", "AnnualizedSales", "Abs Δ", "Growth %"]].copy()
    tbl = tbl.rename(columns={
        "PrevAnnualized": "Prev Year Sales (Ann.)",
        "AnnualizedSales": "This Year Sales (Ann.)",
        "Abs Δ": "Δ Sales (Ann.)"
    })
    st.dataframe(tbl, use_container_width=True)

    st.download_button(
        "⬇️ Download Leaderboard CSV",
        tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"salesperson_growth_leaderboard_{year_focus}.csv",
        mime="text/csv"
    )

# ------------------------
# 2) Bar Chart — YoY Growth % by Salesperson for the selected year
# ------------------------
st.subheader(f"📊 YoY Growth % by Salesperson — {year_focus} (annualized)")
bars = yr_slice.dropna(subset=["Growth %"]).sort_values("Growth %", ascending=True)
if bars.empty:
    st.info("No valid YoY growth to plot for this year.")
else:
    fig_bar = px.bar(
        bars,
        x="Growth %",
        y="Salesperson",
        orientation="h",
        title=f"YoY Growth % by Salesperson — {year_focus} (annualized)",
        hover_data={"PrevAnnualized": ":,.2f", "AnnualizedSales": ":,.2f", "Abs Δ": ":,.2f", "Growth %": ":.2f"}
    )
    fig_bar.update_layout(height=650)
    st.plotly_chart(fig_bar, use_container_width=True)

# ------------------------
# 3) Trend — YoY Growth % across years (per salesperson)
# ------------------------
st.subheader("📈 Trend — YoY Growth % by Salesperson (annualized)")
trend_people = st.multiselect(
    "Choose lines to display",
    options=salespeople,
    default=pick_people
)

trend_df = filtered[filtered["Salesperson"].isin(trend_people)].copy()
trend_df = trend_df.sort_values(["Salesperson", "Year"])

if trend_df["Growth %"].notna().sum() == 0:
    st.info("Not enough data to draw growth trends (need consecutive years per salesperson).")
else:
    fig_lines = px.line(
        trend_df.dropna(subset=["Growth %"]),
        x="Year",
        y="Growth %",
        color="Salesperson",
        markers=True,
        title="YoY Growth % Trend (annualized)",
        hover_data={"PrevAnnualized": ":,.2f", "AnnualizedSales": ":,.2f", "Abs Δ": ":,.2f", "Growth %": ":.2f"}
    )
    fig_lines.update_layout(yaxis_title="Growth (%)")
    st.plotly_chart(fig_lines, use_container_width=True)

# ------------------------
# 4) Reference table — Annualized Sales per Salesperson & Year
# ------------------------
with st.expander("🔎 Reference: Annualized sales by year (used for growth calc)"):
    ref = sp_year[["Year", "Salesperson", "AnnualizedSales", "Months"]].copy().sort_values(["Salesperson", "Year"])
    ref = ref.rename(columns={"AnnualizedSales": "Sales (Annualized)"})
    st.dataframe(ref, use_container_width=True)
    st.download_button(
        "⬇️ Download Annualized Sales (CSV)",
        ref.to_csv(index=False).encode("utf-8"),
        file_name="salesperson_annualized_sales.csv",
        mime="text/csv"
    )

st.caption("Note: Growth % uses **annualized** totals (e.g., if 2025 has 8 months, it's scaled to a 12-month equivalent). Monthly views in your main page remain raw.")
