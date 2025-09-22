#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# -------------------------------
# 1. Load the data
# -------------------------------
order_df = pd.read_excel("/Users/MKSBX/Documents/Analytics Tool/TestData.xlsx", sheet_name = "OrderData")   # Columns: Date, Shipment No., Order No., Sku Code, Qty in Cases, Qty in Eaches
sku_df = pd.read_excel("/Users/MKSBX/Documents/Analytics Tool/TestData.xlsx", sheet_name = "SkuMaster")     # Columns: Sku Code, Category, Case Config, Pallet Fit


# In[3]:


# -------------------------------
# 2. Enrich order data with SKU Master
# -------------------------------
# Merge on SKU code
order_merged = order_df.merge(sku_df, on="Sku Code", how="left")

# Calculate total eaches for every order line
order_merged["Total_Eaches"] = (
    order_merged["Qty in Eaches"].fillna(0) +
    order_merged["Qty in Cases"].fillna(0) * order_merged["Case Config"]
)

# Convert to case equivalent
order_merged["Case_Equivalent"] = order_merged["Total_Eaches"] / order_merged["Case Config"]

# Convert to pallet equivalent
order_merged["Pallet_Equivalent"] = order_merged["Case_Equivalent"] / order_merged["Pallet Fit"]



# In[4]:


# -------------------------------
# 3. Order Summary (Date Wise)
# -------------------------------
date_order_summary = order_merged.groupby("Date").agg(
    Distinct_Customers=("Order No.", "nunique"),   # If you have customer ID, replace with that column
    Distinct_Shipments=("Shipment No.", "nunique"),
    Distinct_Orders=("Order No.", "nunique"),
    Distinct_SKUs=("Sku Code", "nunique"),
    Qty_Ordered_Cases=("Qty in Cases", "sum"),
    Qty_Ordered_Eaches=("Qty in Eaches", "sum"),
    Total_Case_Equiv=("Case_Equivalent", "sum"),
    Total_Pallet_Equiv=("Pallet_Equivalent", "sum")
).reset_index()

date_order_summary.head(10)


# In[5]:


# -------------------------------
# 4. Order Profile (SKU Wise)
# -------------------------------
sku_order_summary = order_merged.groupby("Sku Code").agg(
    Order_Lines=("Order No.", "count"),
    Order_Volume_CE=("Case_Equivalent", "sum")
).reset_index()

sku_order_summary.head(10)


# In[6]:


# -------------------------------
# 5. Order Profile (Percentile)
# -------------------------------

# Define the metric columns (excluding Date)
metrics = [
    "Distinct_Customers",
    "Distinct_Shipments",
    "Distinct_Orders",
    "Distinct_SKUs",
    "Qty_Ordered_Cases",
    "Qty_Ordered_Eaches",
    "Total_Case_Equiv",
    "Total_Pallet_Equiv"
]

# Initialize summary dataframe with Percentile labels
percentile_profile = pd.DataFrame({
    "Percentile": ["Max", "95%ile", "90%ile", "85%ile", "Average"]
})

# Loop through each metric and calculate values
for col in metrics:
    percentile_profile[col] = [
        date_order_summary[col].max(),                      # Max
        np.percentile(date_order_summary[col], 95),         # 95th Percentile
        np.percentile(date_order_summary[col], 90),         # 90th Percentile
        np.percentile(date_order_summary[col], 85),         # 85th Percentile
        date_order_summary[col].mean()                      # Average
    ]

percentile_profile.head(10)


# In[7]:


# -------------------------------
# 6. SKU Profile (ABC-FMS)
# -------------------------------

# -------------------------------
# 6.1 Aggregate base metrics per SKU
# -------------------------------
# Total order lines per SKU
sku_lines = (
    order_merged.groupby("Sku Code", sort=False)
    .agg(Total_Order_Lines=("Order No.", "count"))
    .reset_index()
)

# Total case-equivalent volume per SKU
sku_volume = (
    order_merged.groupby("Sku Code", sort=False)
    .agg(Total_Case_Equiv=("Case_Equivalent", "sum"))
    .reset_index()
)

# Merge line and volume metrics into one base table
sku_metrics = sku_lines.merge(sku_volume, on="Sku Code", how="outer").fillna(0)

# -------------------------------
# 6.2 FMS Classification (based on order lines)
# -------------------------------
total_lines_all = sku_metrics["Total_Order_Lines"].sum()

sku_metrics["Pct_of_Total_Order_Lines"] = (
    sku_metrics["Total_Order_Lines"] / (total_lines_all if total_lines_all != 0 else 1) * 100
)

# Sort SKUs by order lines (descending) for cumulative %
sku_fms_sorted = sku_metrics.sort_values(by="Total_Order_Lines", ascending=False).reset_index(drop=True)

# Cumulative % of order lines
sku_fms_sorted["Cumulative_Pct_Lines"] = sku_fms_sorted["Pct_of_Total_Order_Lines"].cumsum()

# FMS classification
def classify_fms(cum_pct):
    if cum_pct < 70.0:
        return "F"
    elif cum_pct <= 90.0:
        return "M"
    else:
        return "S"

sku_fms_sorted["FMS"] = sku_fms_sorted["Cumulative_Pct_Lines"].apply(classify_fms)

# -------------------------------
# 6.3 ABC Classification (based on case-equivalent volume)
# -------------------------------
total_volume_all = sku_metrics["Total_Case_Equiv"].sum()

sku_metrics["Pct_of_Total_Case_Equiv"] = (
    sku_metrics["Total_Case_Equiv"] / (total_volume_all if total_volume_all != 0 else 1) * 100
)

# Sort SKUs by volume (descending) for cumulative %
sku_abc_sorted = sku_metrics.sort_values(by="Total_Case_Equiv", ascending=False).reset_index(drop=True)

# Cumulative % of volume
sku_abc_sorted["Cumulative_Pct_Volume"] = sku_abc_sorted["Pct_of_Total_Case_Equiv"].cumsum()

# ABC classification
def classify_abc(cum_pct):
    if cum_pct < 70.0:
        return "A"
    elif cum_pct <= 90.0:
        return "B"
    else:
        return "C"

sku_abc_sorted["ABC"] = sku_abc_sorted["Cumulative_Pct_Volume"].apply(classify_abc)

# -------------------------------
# 6.4 Combine FMS and ABC into one SKU profile
# -------------------------------
sku_profile_abc_fms = sku_metrics[
    ["Sku Code", "Total_Order_Lines", "Pct_of_Total_Order_Lines", "Total_Case_Equiv", "Pct_of_Total_Case_Equiv"]
].copy()

# Merge FMS results
sku_profile_abc_fms = sku_profile_abc_fms.merge(
    sku_fms_sorted[["Sku Code", "Cumulative_Pct_Lines", "FMS"]],
    on="Sku Code",
    how="left"
)

# Merge ABC results
sku_profile_abc_fms = sku_profile_abc_fms.merge(
    sku_abc_sorted[["Sku Code", "Cumulative_Pct_Volume", "ABC"]],
    on="Sku Code",
    how="left"
)

# -------------------------------
# 6.5 Frequency-of-movement metrics (optional, included by default)
# -------------------------------
distinct_days = (
    order_merged.dropna(subset=["Date"]).groupby("Sku Code", sort=False)
    .agg(Distinct_Movement_Days=("Date", "nunique"))
    .reset_index()
)

sku_profile_abc_fms = sku_profile_abc_fms.merge(distinct_days, on="Sku Code", how="left")
sku_profile_abc_fms["Distinct_Movement_Days"] = sku_profile_abc_fms["Distinct_Movement_Days"].fillna(0).astype(int)

total_unique_days = order_merged["Date"].nunique()
total_unique_days = int(total_unique_days) if not np.isnan(total_unique_days) else 0

sku_profile_abc_fms["FMS_Period_Pct"] = sku_profile_abc_fms["Distinct_Movement_Days"].apply(
    lambda x: (x / total_unique_days * 100) if total_unique_days > 0 else 0
)

sku_profile_abc_fms["Orders_per_Movement_Day"] = sku_profile_abc_fms.apply(
    lambda r: (r["Total_Order_Lines"] / r["Distinct_Movement_Days"]) if r["Distinct_Movement_Days"] > 0 else 0,
    axis=1
)

# -------------------------------
# 6.6 Formatting & column ordering
# -------------------------------
sku_profile_abc_fms["Pct_of_Total_Order_Lines"] = sku_profile_abc_fms["Pct_of_Total_Order_Lines"].round(2)
sku_profile_abc_fms["Cumulative_Pct_Lines"] = sku_profile_abc_fms["Cumulative_Pct_Lines"].round(2)
sku_profile_abc_fms["Pct_of_Total_Case_Equiv"] = sku_profile_abc_fms["Pct_of_Total_Case_Equiv"].round(2)
sku_profile_abc_fms["Cumulative_Pct_Volume"] = sku_profile_abc_fms["Cumulative_Pct_Volume"].round(2)
sku_profile_abc_fms["FMS_Period_Pct"] = sku_profile_abc_fms["FMS_Period_Pct"].round(2)
sku_profile_abc_fms["Orders_per_Movement_Day"] = sku_profile_abc_fms["Orders_per_Movement_Day"].round(2)

sku_profile_abc_fms["Total_Order_Lines"] = sku_profile_abc_fms["Total_Order_Lines"].astype(int)
sku_profile_abc_fms["Distinct_Movement_Days"] = sku_profile_abc_fms["Distinct_Movement_Days"].astype(int)

# Add 2D-Classification (ABC + FMS)
sku_profile_abc_fms["2D-Classification"] = sku_profile_abc_fms["ABC"] + sku_profile_abc_fms["FMS"]

# Arrange columns for readability
sku_profile_abc_fms = sku_profile_abc_fms[
    [
        "Sku Code",
        "Total_Order_Lines",
        "Pct_of_Total_Order_Lines",
        "Cumulative_Pct_Lines",
        "FMS",
        "Total_Case_Equiv",
        "Pct_of_Total_Case_Equiv",
        "Cumulative_Pct_Volume",
        "ABC",
        "2D-Classification",
        "Distinct_Movement_Days",
        "FMS_Period_Pct",
        "Orders_per_Movement_Day"
    ]
]

sku_profile_abc_fms.head(10)


# In[8]:


# -------------------------------
# 7. ABC x FMS Summary (abc_fms_summary)
# -------------------------------
# Expecting sku_profile_abc_fms to exist with columns:
# ["Sku Code", "Total_Order_Lines", "Pct_of_Total_Order_Lines", "Cumulative_Pct_Lines", "FMS",
#  "Total_Case_Equiv", "Pct_of_Total_Case_Equiv", "Cumulative_Pct_Volume", "ABC",
#  "Distinct_Movement_Days", "FMS_Period_Pct", "Orders_per_Movement_Day"]

# Quick guard: required columns check
_required_cols = {"Sku Code", "Total_Order_Lines", "Total_Case_Equiv", "ABC", "FMS"}
missing = _required_cols - set(sku_profile_abc_fms.columns)
if missing:
    raise ValueError(f"sku_profile_abc_fms is missing required columns: {missing}")

# Normalize ABC and FMS values (trim & upper) to be safe
sku_profile_abc_fms["ABC"] = sku_profile_abc_fms["ABC"].astype(str).str.strip().str.upper()
sku_profile_abc_fms["FMS"] = sku_profile_abc_fms["FMS"].astype(str).str.strip().str.upper()

# -------------------------------
# 7.1 SKU counts cross-tab (ABC x FMS)
# -------------------------------
sku_count_crosstab = pd.crosstab(
    sku_profile_abc_fms["ABC"],
    sku_profile_abc_fms["FMS"],
    margins=False
).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)

# Row total for SKU counts
sku_count_crosstab["SKU_Total"] = sku_count_crosstab.sum(axis=1)

# -------------------------------
# 7.2 Volume (case-equivalent) cross-tab (ABC x FMS)
# -------------------------------
volume_crosstab = (
    sku_profile_abc_fms
    .groupby(["ABC", "FMS"], sort=False)["Total_Case_Equiv"]
    .sum()
    .unstack(fill_value=0)
).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)

# Row total for volume
volume_crosstab["Volume_Total"] = volume_crosstab.sum(axis=1)

# Compute grand total (sum of all F, M, S values across all rows)
grand_total_volume = volume_crosstab[["F", "M", "S"]].to_numpy().sum()

# Divide each cell by grand total, multiply by 100
volume_pct = volume_crosstab[["F", "M", "S"]].div(grand_total_volume) * 100

volume_pct = volume_pct.round(0)
volume_pct.columns = ["Volume_F_pct", "Volume_M_pct", "Volume_S_pct"]

# -------------------------------
# 7.3 Lines (order-lines) cross-tab (ABC x FMS)
# -------------------------------
lines_crosstab = (
    sku_profile_abc_fms
    .groupby(["ABC", "FMS"], sort=False)["Total_Order_Lines"]
    .sum()
    .unstack(fill_value=0)
).reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)

# # Row total for lines
lines_crosstab["Line_Total"] = lines_crosstab.sum(axis=1)

# # Row-wise percentage distribution for lines
# lines_pct = lines_crosstab[["F", "M", "S"]].div(lines_crosstab["Line_Total"].replace(0, np.nan), axis=0).fillna(0) * 100
# lines_pct = lines_pct.round(2)

# Compute grand total (sum of all F, M, S values across all rows)
grand_total_lines = lines_crosstab[["F", "M", "S"]].to_numpy().sum()

# Divide each cell by grand total, multiply by 100
lines_pct = lines_crosstab[["F", "M", "S"]].div(grand_total_lines) * 100

lines_pct = lines_pct.round(0)
lines_pct.columns = ["Line_F_pct", "Line_M_pct", "Line_S_pct"]

# -------------------------------
# 7.4 Assemble one wide summary DataFrame
# -------------------------------
abc_rows = ["A", "B", "C"]
summary_rows = []

for abc in abc_rows:
    row = {
        "ABC": abc,
        # SKU counts
        "SKU_F": int(sku_count_crosstab.at[abc, "F"]) if "F" in sku_count_crosstab.columns else 0,
        "SKU_M": int(sku_count_crosstab.at[abc, "M"]) if "M" in sku_count_crosstab.columns else 0,
        "SKU_S": int(sku_count_crosstab.at[abc, "S"]) if "S" in sku_count_crosstab.columns else 0,
        "SKU_Total": int(sku_count_crosstab.at[abc, "SKU_Total"]),
        # Volume absolute
        "Volume_F": float(volume_crosstab.at[abc, "F"]) if "F" in volume_crosstab.columns else 0.0,
        "Volume_M": float(volume_crosstab.at[abc, "M"]) if "M" in volume_crosstab.columns else 0.0,
        "Volume_S": float(volume_crosstab.at[abc, "S"]) if "S" in volume_crosstab.columns else 0.0,
        "Volume_Total": float(volume_crosstab.at[abc, "Volume_Total"]),
        # Volume row %
        "Volume_F_pct": float(volume_pct.at[abc, "Volume_F_pct"]),
        "Volume_M_pct": float(volume_pct.at[abc, "Volume_M_pct"]),
        "Volume_S_pct": float(volume_pct.at[abc, "Volume_S_pct"]),
        # Lines absolute
        "Line_F": float(lines_crosstab.at[abc, "F"]) if "F" in lines_crosstab.columns else 0.0,
        "Line_M": float(lines_crosstab.at[abc, "M"]) if "M" in lines_crosstab.columns else 0.0,
        "Line_S": float(lines_crosstab.at[abc, "S"]) if "S" in lines_crosstab.columns else 0.0,
        "Line_Total": float(lines_crosstab.at[abc, "Line_Total"]),
        # Lines row %
        "Line_F_pct": float(lines_pct.at[abc, "Line_F_pct"]),
        "Line_M_pct": float(lines_pct.at[abc, "Line_M_pct"]),
        "Line_S_pct": float(lines_pct.at[abc, "Line_S_pct"]),
    }
    summary_rows.append(row)

abc_fms_summary = pd.DataFrame(summary_rows)

# -------------------------------
# 7.5 Grand total row & percent-of-grand for Volume/Lines (optional sanity)
# -------------------------------
grand = {
    "ABC": "Grand Total",
    "SKU_F": int(sku_count_crosstab[["F", "M", "S"]].sum().get("F", 0)),
    "SKU_M": int(sku_count_crosstab[["F", "M", "S"]].sum().get("M", 0)),
    "SKU_S": int(sku_count_crosstab[["F", "M", "S"]].sum().get("S", 0)),
    "SKU_Total": int(sku_count_crosstab["SKU_Total"].sum()),
    "Volume_F": float(volume_crosstab[["F", "M", "S"]].sum().get("F", 0.0)),
    "Volume_M": float(volume_crosstab[["F", "M", "S"]].sum().get("M", 0.0)),
    "Volume_S": float(volume_crosstab[["F", "M", "S"]].sum().get("S", 0.0)),
    "Volume_Total": float(volume_crosstab["Volume_Total"].sum()),
    # For grand row, pct is percent of grand total
    "Volume_F_pct": round((volume_crosstab[["F", "M", "S"]].sum().get("F", 0.0) / (volume_crosstab["Volume_Total"].sum() or 1) * 100), 2),
    "Volume_M_pct": round((volume_crosstab[["F", "M", "S"]].sum().get("M", 0.0) / (volume_crosstab["Volume_Total"].sum() or 1) * 100), 2),
    "Volume_S_pct": round((volume_crosstab[["F", "M", "S"]].sum().get("S", 0.0) / (volume_crosstab["Volume_Total"].sum() or 1) * 100), 2),
    "Line_F": float(lines_crosstab[["F", "M", "S"]].sum().get("F", 0.0)),
    "Line_M": float(lines_crosstab[["F", "M", "S"]].sum().get("M", 0.0)),
    "Line_S": float(lines_crosstab[["F", "M", "S"]].sum().get("S", 0.0)),
    "Line_Total": float(lines_crosstab["Line_Total"].sum()),
    "Line_F_pct": round((lines_crosstab[["F", "M", "S"]].sum().get("F", 0.0) / (lines_crosstab["Line_Total"].sum() or 1) * 100), 2),
    "Line_M_pct": round((lines_crosstab[["F", "M", "S"]].sum().get("M", 0.0) / (lines_crosstab["Line_Total"].sum() or 1) * 100), 2),
    "Line_S_pct": round((lines_crosstab[["F", "M", "S"]].sum().get("S", 0.0) / (lines_crosstab["Line_Total"].sum() or 1) * 100), 2),
}

abc_fms_summary = pd.concat([abc_fms_summary, pd.DataFrame([grand])], ignore_index=True, sort=False)

# -------------------------------
# 7.6 Formatting & rounding
# -------------------------------
# Round numeric columns for neatness
_pct_cols = ["Volume_F_pct", "Volume_M_pct", "Volume_S_pct", "Line_F_pct", "Line_M_pct", "Line_S_pct"]
_amt_cols = ["Volume_F", "Volume_M", "Volume_S", "Volume_Total", "Line_F", "Line_M", "Line_S", "Line_Total"]

abc_fms_summary[_pct_cols] = abc_fms_summary[_pct_cols].fillna(0).astype(float).round(2)
abc_fms_summary[_amt_cols] = abc_fms_summary[_amt_cols].fillna(0).astype(float).round(2)

# Ensure integer SKU counts
abc_fms_summary[["SKU_F", "SKU_M", "SKU_S", "SKU_Total"]] = abc_fms_summary[["SKU_F", "SKU_M", "SKU_S", "SKU_Total"]].fillna(0).astype(int)

abc_fms_summary  # dataframe available in the environment


# In[9]:


# -------------------------------
# #. Export results
# -------------------------------
with pd.ExcelWriter("Order_Profiles.xlsx") as writer:
    date_order_summary.to_excel(writer, sheet_name="Date Order Summary", index=False)
    sku_order_summary.to_excel(writer, sheet_name="SKU Order Summary", index=False)
    percentile_profile.to_excel(writer, sheet_name="Order Profile(Percentile)", index=False)
    sku_profile_abc_fms.to_excel(writer, sheet_name="SKU_Profile_ABC_FMS", index=False)
    abc_fms_summary.to_excel(writer, sheet_name="ABC_FMS_Summary", index=False)

print("✅ Analysis complete! Results written to Order_Profiles.xlsx")


# In[27]:


"""
Order Profiles -> HTML Report Generator
Produces: report/Order_Profiles_Analysis.html + charts/ + metadata/cache

Assumptions:
 - The following pandas DataFrames exist in the environment:
   date_order_summary, percentile_profile, sku_order_summary (optional), sku_profile_abc_fms, abc_fms_summary
 - GEMINI API key in env var GEMINI_API_KEY if LLM summaries are desired.

Author: Generated for user (style matched to prior code)
"""
import os
import json
import math
from datetime import datetime
from pathlib import Path
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
import requests

# -------------------------------
# ENV VARIABLES
# -------------------------------


# -------------------------------
# CONFIG
# -------------------------------
REPORT_DIR = Path("report")
CHARTS_DIR = REPORT_DIR / "charts"
ASSETS_DIR = REPORT_DIR / "assets"
CACHE_FILE = REPORT_DIR / "llm_cache.json"
HTML_FILE = REPORT_DIR / "Order_Profiles_Analysis.html"
METADATA_FILE = REPORT_DIR / "metadata.json"

# Gemini config
USE_GEMINI = True                        # set False to skip LLM calls (still builds report)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="  # placeholder; update to your provider/endpoint if different
# NOTE: replace GEMINI_ENDPOINT with the actual Gemini/Google endpoint if required.
# The script will gracefully skip LLM calls if KEY not found or USE_GEMINI=False.

# Report options
TOP_N_TABLE_ROWS = 50  # for large tables show top N rows; link to Excel for full table
OPEN_AFTER_BUILD = False  # if true, attempts to open HTML in default browser after build

# Ensure directories
REPORT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

# -------------------------------
# HELPERS
# -------------------------------
def safe_df_to_html(df: pd.DataFrame, max_rows:int=TOP_N_TABLE_ROWS, float_format="%.2f"):
    """
    Return HTML snippet for a dataframe. Shows top N rows and a note if truncated.
    """
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"
    show_df = df.copy()
    truncated = False
    if max_rows is not None and len(show_df) > max_rows:
        show_df = show_df.head(max_rows)
        truncated = True

    html_table = show_df.to_html(classes="table", index=False, float_format=float_format, border=0)
    note = ""
    if truncated:
        note = f"<p class='muted small'>Showing top {max_rows} rows. Full table available in the Excel workbook.</p>"
    return html_table + note

def save_fig(fig, filename: Path, dpi=150):
    """Save matplotlib figure ensuring no explicit colors are set."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(filename), bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))

# -------------------------------
# 0. Collect DataFrames (adjust names if needed)
# -------------------------------
# Map of expected DataFrames -> variable names in the environment
# If you have different names, change these mappings.
REPORT_DFS = {
    "date_order_summary": globals().get("date_order_summary"),
    "percentile_profile": globals().get("percentile_profile"),
    "sku_order_summary": globals().get("sku_order_summary"),  # optional; if None script will skip related charts
    "sku_profile_abc_fms": globals().get("sku_profile_abc_fms"),
    "abc_fms_summary": globals().get("abc_fms_summary")
}

# Quick guard for mandatory dfs
_mandatory = ["date_order_summary", "percentile_profile", "sku_profile_abc_fms", "abc_fms_summary"]
missing = [k for k in _mandatory if REPORT_DFS.get(k) is None]
if missing:
    raise ValueError(f"Missing required DataFrames in environment: {missing}. "
                     "Make sure those DataFrames exist before running the script.")

# -------------------------------
# 1. Generate Charts (matplotlib)
# -------------------------------
# Rule: one chart per figure; do not set explicit colors; use default matplotlib palette.

# 1.1 Date profile charts
def chart_date_time_series(df: pd.DataFrame):
    """Line chart of Total_Case_Equiv over Date and bar chart of Distinct_Customers."""
    d = df.copy()
    # ensure date sorted
    d = d.sort_values("Date")
    # line chart for Total_Case_Equiv
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d["Date"], d["Total_Case_Equiv"], marker="o", linewidth=1)
    ax.set_title("Total Case Equivalent by Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Case Equivalent")
    ax.grid(True, linestyle=':', linewidth=0.5)
    file_line = CHARTS_DIR / "date_total_case_equiv.png"
    save_fig(fig, file_line)
    # bar chart for Distinct_Customers
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.bar(d["Date"].astype(str), d["Distinct_Customers"])
    ax2.set_title("Distinct Customers by Date")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Distinct Customers")
    plt.xticks(rotation=45, ha="right")
    ax2.grid(False)
    file_bar = CHARTS_DIR / "date_distinct_customers.png"
    save_fig(fig2, file_bar)
    return str(file_line), str(file_bar)

date_line_chart, date_customers_chart = chart_date_time_series(REPORT_DFS["date_order_summary"])

# 1.2 Percentile chart (Total_Case_Equiv)
def chart_percentiles(df: pd.DataFrame):
    p = df.set_index("Percentile")
    # try to plot the Total_Case_Equiv percentiles as horizontal bar
    values = p["Total_Case_Equiv"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(values.index, values.values)
    ax.set_title("Percentiles for Total Case Equivalent (per day)")
    ax.set_xlabel("Case Equivalent")
    filep = CHARTS_DIR / "percentile_total_case_equiv.png"
    save_fig(fig, filep)
    return str(filep)

percentile_chart = chart_percentiles(REPORT_DFS["percentile_profile"])

# 1.3 SKU Pareto (if sku_order_summary is present)
def chart_sku_pareto(df: pd.DataFrame, top_n=50):
    if df is None or df.empty:
        return None
    # expecting sku_order_summary with Sku Code and Order_Volume_CE or similar
    # try common column names
    vol_col = None
    candidates = ["Order_Volume_CE", "Order_Volume", "Case_Equivalent", "Total_Case_Equiv"]
    for c in candidates:
        if c in df.columns:
            vol_col = c
            break
    if vol_col is None:
        # fall back to 'Total_Case_Equiv' in sku_profile_abc_fms if available
        if "Total_Case_Equiv" in REPORT_DFS["sku_profile_abc_fms"].columns:
            vol_col = "Total_Case_Equiv"
            df_use = REPORT_DFS["sku_profile_abc_fms"]
        else:
            return None
    else:
        df_use = df

    df_sorted = df_use.sort_values(vol_col, ascending=False).head(top_n)
    cum = df_sorted[vol_col].cumsum() / df_sorted[vol_col].sum() * 100
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(range(len(df_sorted)), df_sorted[vol_col])
    ax1.set_xlabel("Top SKUs (by volume)")
    ax1.set_ylabel("Volume")
    ax2 = ax1.twinx()
    ax2.plot(range(len(df_sorted)), cum, color=None, marker="o")
    ax2.set_ylabel("Cumulative %")
    ax1.set_title("SKU Pareto (Top {})".format(top_n))
    filep = CHARTS_DIR / "sku_pareto.png"
    save_fig(fig, filep)
    return str(filep)

sku_pareto_chart = chart_sku_pareto(REPORT_DFS.get("sku_order_summary"))

# 1.4 ABC volume stacked by FMS (from sku_profile_abc_fms)
def chart_abc_volume_stacked(df: pd.DataFrame):
    # expects columns: ABC, FMS, Total_Case_Equiv
    if df is None or df.empty:
        return None
    pivot = df.pivot_table(index="ABC", columns="FMS", values="Total_Case_Equiv", aggfunc="sum", fill_value=0)
    # ensure order A,B,C and F,M,S
    pivot = pivot.reindex(index=["A", "B", "C"], columns=["F", "M", "S"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8,4))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Volume by ABC and FMS (stacked)")
    ax.set_ylabel("Total Case Equivalent")
    ax.set_xlabel("ABC Class")
    filep = CHARTS_DIR / "abc_volume_stacked.png"
    save_fig(fig, filep)
    return str(filep)

abc_volume_chart = chart_abc_volume_stacked(REPORT_DFS["sku_profile_abc_fms"])

# 1.5 ABCxFMS heatmap (volume share)
def chart_abc_fms_heatmap(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    # use abc_fms_summary for convenience if present
    summary = REPORT_DFS["abc_fms_summary"]
    # pivot volume percentages by ABC rows and F/M/S columns (Volume_F_pct etc)
    # We will compute the matrix from sku_profile_abc_fms for accuracy
    pivot = df.pivot_table(index="ABC", columns="FMS", values="Total_Case_Equiv", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=["A","B","C"], columns=["F","M","S"], fill_value=0)
    # normalize to row %
    pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)*100
    fig, ax = plt.subplots(figsize=(6,4))
    cax = ax.imshow(pivot_pct.values, aspect='auto')
    ax.set_xticks(range(len(pivot_pct.columns)))
    ax.set_xticklabels(pivot_pct.columns)
    ax.set_yticks(range(len(pivot_pct.index)))
    ax.set_yticklabels(pivot_pct.index)
    ax.set_title("ABC x FMS Volume % (row-wise)")
    # annotate values
    for (i, j), val in np.ndenumerate(pivot_pct.values):
        ax.text(j, i, f"{val:.1f}%", ha='center', va='center', fontsize=9)
    fig.colorbar(cax, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    filep = CHARTS_DIR / "abc_fms_heatmap.png"
    save_fig(fig, filep)
    return str(filep)

abc_fms_heatmap_chart = chart_abc_fms_heatmap(REPORT_DFS["sku_profile_abc_fms"])

# -------------------------------
# 2. Prepare LLM prompts and call Gemini (with caching)
# -------------------------------
llm_cache = load_cache()

def build_prompt(section_name: str, facts: dict, instructions: str) -> str:
    """
    Create a concise prompt for the LLM containing key numeric facts and a clear instruction.
    facts: dict of short key: value pairs to include.
    """
    lines = [f"Section: {section_name}", ""]
    lines.append("Key facts:")
    for k, v in facts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Task:")
    lines.append(instructions)
    return "\n".join(lines)

# -------------------------------
# GEMINI / LLM config (safe handling)
# -------------------------------
# Prefer environment variable. If you temporarily set a literal value (not recommended),
# set it to GEMINI_API_KEY_ENV before running and then remove it afterwards.
GEMINI_API_KEY= "AIzaSyD3-HabX9Oc2Q_0R-wywpRk8QZ03Z7HHds"
# GEMINI_API_KEY_ENV = os.getenv("GEMINI_API_KEY")
# If you previously hard-coded GEMINI_API_KEY in the file, avoid that. Use env var instead.
# GEMINI_API_KEY = GEMINI_API_KEY_ENV

# -------------------------------
# Improved call_gemini wrapper
# -------------------------------
def call_gemini(prompt: str) -> str:
    """
    Call Gemini API using the working generateContent pattern.
    """
    if not USE_GEMINI or not GEMINI_API_KEY:
        return "(LLM summaries disabled or API key missing.)"

    # cache check
    cache_key = str(abs(hash(prompt)))
    if cache_key in llm_cache:
        return llm_cache[cache_key]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code != 200:
            msg = f"(LLM call HTTP {resp.status_code}) {json.dumps(data)[:500]}"
            llm_cache[cache_key] = msg
            save_cache(llm_cache)
            return msg

        # Extract text safely
        text_out = (
            data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
        )

        if not text_out:
            text_out = json.dumps(data)[:500]

    except Exception as e:
        text_out = f"(LLM call failed: {e})"

    # save + return
    llm_cache[cache_key] = text_out
    save_cache(llm_cache)
    return text_out




# Compose prompts for each section
llm_summaries = {}

# Section: Cover (high-level)
cover_facts = {
    "Generated on": datetime.now().isoformat(),
    "Total dates (rows)": len(REPORT_DFS["date_order_summary"]),
    "Total SKUs (unique)": int(REPORT_DFS["sku_profile_abc_fms"]["Sku Code"].nunique()),
    "Total orders (lines)": int(REPORT_DFS["sku_profile_abc_fms"]["Total_Order_Lines"].sum())
}
cover_instr = "Produce a 3-sentence executive summary that highlights dataset scope and top-level operational implications (2 bullets)."
llm_summaries["cover"] = call_gemini(build_prompt("Cover", cover_facts, cover_instr))

# Section: Date Profile
top_dates = REPORT_DFS["date_order_summary"].sort_values("Total_Case_Equiv", ascending=False).head(3)
date_facts = {
    "Date range": f"{REPORT_DFS['date_order_summary']['Date'].min().date()} - {REPORT_DFS['date_order_summary']['Date'].max().date()}",
    "Top dates (by CE)": "; ".join([f"{r['Date'].date()} -> {r['Total_Case_Equiv']:.0f}" for _, r in top_dates.iterrows()]),
    "95th percentile CE/day": float(REPORT_DFS["percentile_profile"].set_index("Percentile").at["95%ile","Total_Case_Equiv"]) if "95%ile" in REPORT_DFS["percentile_profile"]["Percentile"].values else ""
}
date_instr = "Write a 4-sentence description of demand patterns and 3 operational recommendations (staffing, pallet allocation, short-term buffer)."
llm_summaries["date_order_summary"] = call_gemini(build_prompt("Date Profile", date_facts, date_instr))

# Section: Percentiles
pct_facts = {row["Percentile"]: float(row["Total_Case_Equiv"]) for _, row in REPORT_DFS["percentile_profile"].iterrows()}
pct_instr = "Provide a short interpretation (3 sentences) of these percentiles for capacity planning and peak provisioning."
llm_summaries["percentile_profile"] = call_gemini(build_prompt("Percentile Summary", pct_facts, pct_instr))

# Section: SKU Pareto / Profile
top_skus = None
if REPORT_DFS.get("sku_order_summary") is not None:
    sp = REPORT_DFS["sku_order_summary"]
else:
    sp = REPORT_DFS["sku_profile_abc_fms"]

# Get top 5 SKUs by total case equiv (try multiple column names)
vol_col_candidates = ["Order_Volume_CE", "Order_Volume", "Total_Case_Equiv", "Case_Equivalent"]
vol_col = next((c for c in vol_col_candidates if c in sp.columns), None)
if vol_col:
    top_skus = sp.sort_values(vol_col, ascending=False).head(5)
    sku_facts = {"Top SKUs (by volume)": "; ".join([f"{r['Sku Code']}:{r[vol_col]:.0f}" for _, r in top_skus.iterrows()])}
else:
    sku_facts = {"Note": "Volume column not found in sku_order_summary; using sku_profile_abc_fms instead."}

sku_instr = "Summarize the Pareto characteristics (3 sentences) and one inventory slotting recommendation."
llm_summaries["sku_order_summary"] = call_gemini(build_prompt("SKU Profile", sku_facts, sku_instr))

# Section: ABC-FMS
abc_sample = REPORT_DFS["abc_fms_summary"].head(3).to_dict(orient="records")
abc_facts = {
    "Example rows (first 3)": json.dumps(abc_sample),
    "Interpretation": "Cross-tab of ABC (volume) vs FMS (lines)."
}
abc_instr = "Provide a short analysis of distribution and 3 prioritized recommendations for slotting & replenishment cadence."
llm_summaries["abc_fms"] = call_gemini(build_prompt("ABC x FMS Summary", abc_facts, abc_instr))

# -------------------------------
# 3. Build HTML via Jinja2 template
# -------------------------------
# Simple template embedded here for portability; you can move to file if preferred.
TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Order Profiles Analysis</title>
  <style>
    body{font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin:30px; color:#111;}
    header {text-align:center; margin-bottom: 20px;}
    h1{font-size:28px; margin-bottom:5px;}
    h2{font-size:20px; margin-top:30px; border-bottom:1px solid #eee; padding-bottom:6px;}
    .kpi{display:inline-block; margin-right:14px; padding:8px 12px; background:#f7f7f7; border-radius:6px;}
    .section-summary{background:#fcfcfc; padding:12px; border-left:4px solid #ddd; margin:8px 0 14px 0;}
    .table {width:100%; border-collapse:collapse; margin-bottom:10px;}
    .table th, .table td{padding:6px 8px; border:1px solid #eee; font-size:12px;}
    .muted{color:#666;}
    .small{font-size:12px;}
    .chart{margin:14px 0; text-align:center;}
    .toc{margin-bottom:20px;}
    .toc a{display:block; margin:4px 0;}
    footer{margin-top:40px; font-size:12px; color:#666; border-top:1px solid #eee; padding-top:10px;}
    .grid {display:grid; grid-template-columns: 1fr 1fr; gap:18px;}
    .mono{font-family:monospace; font-size:13px; background:#fafafa; padding:6px; border-radius:4px;}
    .note{font-size:12px; color:#555}
  </style>
  <script>
    function toggle(id){var e=document.getElementById(id); if(e.style.display==='none') e.style.display='block'; else e.style.display='none';}
  </script>
</head>
<body>
  <header>
    <h1>Order Profiles Analysis</h1>
    <div class="muted">Generated: {{ generated_on }}</div>
    <div style="margin-top:10px;">
      <span class="kpi">Dates analyzed: {{ meta.total_dates }}</span>
      <span class="kpi">Unique SKUs: {{ meta.total_skus }}</span>
      <span class="kpi">Order lines: {{ meta.total_order_lines }}</span>
    </div>
  </header>

  <div class="toc">
    <strong>Contents</strong>
    <a href="#date">1. Date Profile</a>
    <a href="#percentile">2. Percentile Summary</a>
    <a href="#sku">3. SKU Profile</a>
    <a href="#abc_fms">4. ABC-FMS & 2D</a>
    <a href="#abc_fms_summary">5. ABC×FMS Summary</a>
    <a href="#appendix">Appendix</a>
  </div>

  <section id="date">
    <h2>1. Date Profile</h2>
    <div class="section-summary">
      {{ llm.date_order_summary | safe }}
    </div>
    <div class="chart">
      <img src="{{ charts.date_line }}" alt="Total Case Equivalent by Date" style="max-width:100%; height:auto;">
    </div>
    <div class="chart">
      <img src="{{ charts.date_customers }}" alt="Distinct Customers by Date" style="max-width:100%; height:auto;">
    </div>
    <h3>Data (top rows)</h3>
    <div>{{ tables.date_order_summary | safe }}</div>
  </section>

  <section id="percentile">
    <h2>2. Percentile Summary</h2>
    <div class="section-summary">{{ llm.percentile_profile | safe }}</div>
    <div class="chart">
      <img src="{{ charts.percentile }}" alt="Percentiles" style="max-width:80%; height:auto;">
    </div>
    <div>{{ tables.percentile_profile | safe }}</div>
  </section>

  <section id="sku">
    <h2>3. SKU Profile</h2>
    <div class="section-summary">{{ llm.sku_order_summary | safe }}</div>
    {% if charts.sku_pareto %}
      <div class="chart"><img src="{{ charts.sku_pareto }}" alt="SKU Pareto" style="max-width:100%;"></div>
    {% endif %}
    <div>{{ tables.sku_order_summary | safe }}</div>
  </section>

  <section id="abc_fms">
    <h2>4. SKU ABC & FMS</h2>
    <div class="section-summary">{{ llm.abc_fms | safe }}</div>
    {% if charts.abc_volume %}
      <div class="chart"><img src="{{ charts.abc_volume }}" alt="ABC Volume stacked" style="max-width:90%;"></div>
    {% endif %}
    {% if charts.abc_heatmap %}
      <div class="chart"><img src="{{ charts.abc_heatmap }}" alt="ABCxFMS heatmap" style="max-width:60%;"></div>
    {% endif %}
    <div>{{ tables.sku_profile_abc_fms | safe }}</div>
  </section>

  <section id="abc_fms_summary">
    <h2>5. ABC × FMS Summary</h2>
    <div class="section-summary">Summary cross-tab of SKU counts, Volume and Lines across ABC vs FMS.</div>
    <div>{{ tables.abc_fms_summary | safe }}</div>
  </section>

  <section id="appendix">
    <h2>Appendix</h2>
    <h3>Methodology & Assumptions</h3>
    <p class="note">
      Case equivalent computed using Case Config; pallet equivalence used earlier not shown here.
      ABC cutoffs: &lt;70% → A; 70–90% → B; &gt;90% → C. FMS cutoffs applied similarly on order-line cumulative %.
    </p>

    <h3>Metadata</h3>
    <pre class="mono">{{ meta | tojson }}</pre>

    <h3>Raw tables / downloads</h3>
    <p class="muted small">Full Excel workbook is expected to be alongside this report: <code>Order_Profiles.xlsx</code></p>
  </section>

  <footer>
    Report generated by Analytics Automation Tool. <span class="muted">Script runtime: {{ generated_on }}</span>
  </footer>
</body>
</html>
"""

env = Environment(autoescape=select_autoescape(["html", "xml"]))
template = env.from_string(TEMPLATE)

# -------------------------------
# 4. Render HTML with data, charts & tables
# -------------------------------
generated_on = datetime.now().isoformat()
meta = {
    "total_dates": int(len(REPORT_DFS["date_order_summary"])),
    "total_skus": int(REPORT_DFS["sku_profile_abc_fms"]["Sku Code"].nunique()),
    "total_order_lines": int(REPORT_DFS["sku_profile_abc_fms"]["Total_Order_Lines"].sum()),
    "dataframes": {k: (len(v) if v is not None else 0) for k, v in REPORT_DFS.items()}
}

# Prepare table HTML snippets
# Use explicit None checks instead of 'or' to avoid ambiguous truth tests on DataFrames
sku_order_df = REPORT_DFS.get("sku_order_summary")
if sku_order_df is None:
    sku_order_df = REPORT_DFS["sku_profile_abc_fms"]

tables = {
    "date_order_summary": safe_df_to_html(REPORT_DFS["date_order_summary"]),
    "percentile_profile": safe_df_to_html(REPORT_DFS["percentile_profile"]),
    "sku_order_summary": safe_df_to_html(sku_order_df),
    "sku_profile_abc_fms": safe_df_to_html(REPORT_DFS["sku_profile_abc_fms"]),
    "abc_fms_summary": safe_df_to_html(REPORT_DFS["abc_fms_summary"], max_rows=None)  # show full small summary
}

charts = {
    "date_line": date_line_chart,
    "date_customers": date_customers_chart,
    "percentile": percentile_chart,
    "sku_pareto": sku_pareto_chart,
    "abc_volume": abc_volume_chart,
    "abc_heatmap": abc_fms_heatmap_chart
}

# Provide llm snippets (fall back to cached plain text)
llm_blocks = {
    "cover": llm_summaries.get("cover",""),
    "date_order_summary": llm_summaries.get("date_order_summary",""),
    "percentile_profile": llm_summaries.get("percentile_profile",""),
    "sku_order_summary": llm_summaries.get("sku_order_summary",""),
    "abc_fms": llm_summaries.get("abc_fms","")
}

# --- Ensure chart image paths are relative to the report folder so <img src=> resolves correctly ---
# charts currently contain absolute or report-prefixed paths like "report/charts/xxx.png"
# Convert each to a relative path FROM the HTML file (which is inside REPORT_DIR)
def _rel_to_report(path_str):
    if not path_str:
        return path_str
    try:
        p = Path(path_str)
        # if already relative inside report, just use charts/<name>
        if p.is_absolute():
            # compute relative path from REPORT_DIR
            return os.path.relpath(str(p), start=str(REPORT_DIR))
        else:
            # if path already contains REPORT_DIR prefix, drop it
            txt = str(path_str)
            if txt.startswith(str(REPORT_DIR) + os.sep):
                return txt[len(str(REPORT_DIR))+1:]
            # otherwise just return as-is (likely "charts/...")
            return txt
    except Exception:
        return path_str

# Apply to all chart entries
for k, v in charts.items():
    charts[k] = _rel_to_report(v)


html_out = template.render(
    generated_on=generated_on,
    meta=meta,
    tables=tables,
    charts=charts,
    llm=llm_blocks
)

HTML_FILE.write_text(html_out, encoding="utf-8")

# Save metadata file
METADATA_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Save cache
save_cache(llm_cache)

print(f"✅ HTML report built: {HTML_FILE.resolve()}")
print(f" - Charts directory: {CHARTS_DIR.resolve()}")
print(f" - If you want to view: open {HTML_FILE.resolve()} in a browser or run: cd {REPORT_DIR}; python -m http.server 8000")

# Optionally open in browser (if requested)
if OPEN_AFTER_BUILD:
    import webbrowser
    webbrowser.open_new_tab(HTML_FILE.resolve().as_uri())

# -------------------------------
# OPTIONAL: Convert to PDF (requires WeasyPrint)
# -------------------------------
# Uncomment to create a PDF (weasyprint must be installed and appropriate system libs)
# try:
#     from weasyprint import HTML
#     pdf_path = REPORT_DIR / "Order_Profiles_Analysis.pdf"
#     HTML(filename=str(HTML_FILE)).write_pdf(str(pdf_path))
#     print(f"✅ PDF report built: {pdf_path.resolve()}")
# except Exception as e:
#     print(f"⚠️ PDF conversion skipped/failed: {e}")

# -------------------------------
# END
# -------------------------------


# In[24]:


# Gemini simple test (Python) — run in Jupyter or plain Python
import os, json, requests, textwrap

os.environ["GEMINI_API_KEY"] = "AIzaSyD3-HabX9Oc2Q_0R-wywpRk8QZ03Z7HHds"

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in your environment first")

BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "models/gemini-2.0-flash"   # just the model
URL = f"{BASE}/{MODEL}:generateContent?key={API_KEY}"  # use generateContent, not generateText

payload = {
    "contents": [
        {
            "parts": [
                {"text": "Say one short sentence about warehouse efficiency."}
            ]
        }
    ]
}

print("POST ->", URL)
resp = requests.post(URL, json=payload, timeout=20)
print("Status:", resp.status_code)

try:
    data = resp.json()
    print(json.dumps(data, indent=2)[:2000])
except Exception:
    print("Raw response:", resp.text)

# Extract text like your Apps Script does
if resp.status_code == 200:
    candidates = data.get("candidates", [])
    if candidates:
        text = candidates[0].get("content", {}).get("parts", [])[0].get("text")
        print("\nExtracted text:", text.strip())


# API_KEY = os.getenv("GEMINI_API_KEY")
# if not API_KEY:
#     raise RuntimeError("Set GEMINI_API_KEY in environment before running this cell.")

# LIST_URL = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
# print("GET ->", LIST_URL)
# r = requests.get(LIST_URL, timeout=20)
# print("Status:", r.status_code)
# try:
#     print(json.dumps(r.json(), indent=2)[:4000])
# except Exception:
#     print(r.text)


# In[ ]:




