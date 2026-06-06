
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# PAGE SETUP
# =====================================================

st.set_page_config(
    page_title="Wholesale & Retail Trade in Malaysia Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("iowrt_2d.csv")
    df["date"] = pd.to_datetime(df["date"])

    division_map = {
        45: "45 - Motor Vehicles",
        46: "46 - Wholesale Trade",
        47: "47 - Retail Trade"
    }

    df["division_label"] = df["division"].map(division_map)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    return df


if not os.path.exists("iowrt_2d.csv"):
    st.error("Dataset not found. Please upload iowrt_2d.csv first.")
    st.stop()

df = load_data()


# =====================================================
# FORMAT FUNCTIONS
# =====================================================

def fmt_rm(value, decimals=0):
    if pd.isna(value):
        return "RM 0"
    return f"RM {value:,.{decimals}f}"


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "0"
    return f"{value:,.{decimals}f}"


# =====================================================
# STYLE
# =====================================================

PALETTE = ["#16D9FF", "#22E17F", "#F9D423", "#FF4D6D", "#A78BFA", "#FF9F1C"]

st.markdown("""
<style>

.stApp {
    background: #151538;
    color: #F5F6FF;
}

.block-container {
    padding-top: 3rem;
    padding-left: 1.6rem;
    padding-right: 1.6rem;
    max-width: 100%;
}

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 2px solid #2F335F;
}

section[data-testid="stSidebar"] * {
    color: #F5F6FF;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 900;
    color: #F5F6FF;
    border-left: 8px solid #16D9FF;
    padding-left: 12px;
    margin-bottom: 18px;
}

.main-title {
    font-size: 42px;
    font-weight: 900;
    color: #F5F6FF;
    margin-bottom: 6px;
}

.main-subtitle {
    font-size: 17px;
    color: #C7C8E8;
    margin-bottom: 22px;
}

.top-strip {
    background: #282B5F;
    color: #F5F6FF;
    border-radius: 10px;
    padding: 13px 18px;
    font-size: 21px;
    font-weight: 900;
    margin-bottom: 14px;
}

.panel {
    background: #292B60;
    border-radius: 14px;
    padding: 18px;
    min-height: 150px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0px 8px 24px rgba(0,0,0,0.28);
    margin-bottom: 14px;
}

.card-title {
    font-size: 18px;
    font-weight: 800;
    color: #EDEEFF;
    margin-bottom: 10px;
}

.card-value {
    font-size: 36px;
    font-weight: 900;
    color: #F5F6FF;
    line-height: 1.15;
    word-break: break-word;
}

.card-value-medium {
    font-size: 30px;
    font-weight: 900;
    color: #F5F6FF;
    line-height: 1.15;
    word-break: break-word;
}

.card-label {
    font-size: 15px;
    color: #C7C8E8;
    margin-top: 8px;
}

.green-text {
    color: #22E17F;
    font-weight: 900;
}

.red-text {
    color: #FF4D6D;
    font-weight: 900;
}

button[data-baseweb="tab"] {
    background-color: #24264F;
    color: #F5F6FF;
    border-radius: 8px 8px 0px 0px;
    font-weight: 800;
    margin-right: 4px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #16D9FF;
    color: #151538;
}

.stDownloadButton button {
    background: #16D9FF;
    color: #151538;
    border-radius: 8px;
    border: none;
    font-weight: 900;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.stAlert {
    border-radius: 12px;
}

hr {
    border-color: #383B73;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.markdown('<div class="sidebar-title">Interactive Filters</div>', unsafe_allow_html=True)

selected_series = st.sidebar.multiselect(
    "Select Series",
    options=sorted(df["series"].unique()),
    default=["abs"]
)

selected_divisions = st.sidebar.multiselect(
    "Select Division",
    options=sorted(df["division_label"].dropna().unique()),
    default=sorted(df["division_label"].dropna().unique())
)

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date


filtered_df = df[
    (df["series"].isin(selected_series)) &
    (df["division_label"].isin(selected_divisions)) &
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

abs_filtered = filtered_df[filtered_df["series"] == "abs"].copy()


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-title">Wholesale & Retail Trade in Malaysia Dashboard</div>
<div class="main-subtitle">
A decision-support dashboard for monitoring Malaysia's trade performance, sector comparison, growth patterns, and future sales forecast.
</div>
""", unsafe_allow_html=True)


# =====================================================
# KPI CALCULATION
# =====================================================

if not abs_filtered.empty:
    total_sales = abs_filtered["sales"].sum()
    avg_sales = abs_filtered["sales"].mean()
    avg_volume = abs_filtered["volume"].mean()

    top_division = abs_filtered.groupby("division_label")["sales"].sum().idxmax()

    first_month = abs_filtered["date"].min()
    latest_month = abs_filtered["date"].max()

    first_sales = abs_filtered[abs_filtered["date"] == first_month]["sales"].sum()
    latest_sales = abs_filtered[abs_filtered["date"] == latest_month]["sales"].sum()

    sales_growth = ((latest_sales - first_sales) / first_sales) * 100 if first_sales != 0 else 0
else:
    total_sales = avg_sales = avg_volume = latest_sales = sales_growth = 0
    top_division = "N/A"


growth_yoy = df[
    (df["series"] == "growth_yoy") &
    (df["division_label"].isin(selected_divisions)) &
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

avg_yoy_growth = growth_yoy["sales"].mean() if not growth_yoy.empty else 0


# =====================================================
# KPI CARDS
# =====================================================

st.markdown('<div class="top-strip">Trade Performance Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="panel">
        <div class="card-title">Total Sales</div>
        <div class="card-value">{fmt_rm(total_sales, 0)}</div>
        <div class="card-label">Accumulated selected sales</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="panel">
        <div class="card-title">Average Sales</div>
        <div class="card-value">{fmt_rm(avg_sales, 0)}</div>
        <div class="card-label">Monthly average sales</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="panel">
        <div class="card-title">Average Volume</div>
        <div class="card-value">{fmt_num(avg_volume, 2)}</div>
        <div class="card-label">Volume index</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="panel">
        <div class="card-title">Top Division</div>
        <div class="card-value-medium">{top_division}</div>
        <div class="card-label">Highest total sales</div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive View",
    "Sales Trend",
    "Division Comparison",
    "Growth Analysis",
    "Forecasting"
])


# =====================================================
# TAB 1: EXECUTIVE VIEW
# =====================================================

with tab1:

    st.markdown('<div class="top-strip">Executive Dashboard View</div>', unsafe_allow_html=True)

    left, middle, right = st.columns([1.05, 2.2, 2.2])

    with left:
        growth_class = "green-text" if sales_growth >= 0 else "red-text"

        st.markdown(f"""
        <div class="panel">
            <div class="card-title">Sales Growth</div>
            <div class="card-value {growth_class}">{sales_growth:,.1f}%</div>
            <div class="card-label">First month to latest month</div>
        </div>
        """, unsafe_allow_html=True)

        trade_score = min(max((avg_volume / 200) * 100, 0), 100)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=trade_score,
            number={"font": {"color": "#F5F6FF", "size": 30}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#F5F6FF"},
                "bar": {"color": "#22E17F"},
                "bgcolor": "#292B60",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#3B3D78"},
                    {"range": [40, 70], "color": "#414483"},
                    {"range": [70, 100], "color": "#4B4E95"}
                ],
            }
        ))

        fig_gauge.update_layout(
            title="Trade Health Score",
            height=330,
            margin=dict(l=10, r=10, t=60, b=10),
            paper_bgcolor="#292B60",
            plot_bgcolor="#292B60",
            font=dict(color="#F5F6FF")
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

    with middle:
        if not abs_filtered.empty:
            fig_trend = px.line(
                abs_filtered,
                x="date",
                y="sales",
                color="division_label",
                markers=True,
                color_discrete_sequence=PALETTE,
                template="plotly_dark",
                title="Monthly Sales Trend"
            )

            fig_trend.update_layout(
                height=360,
                paper_bgcolor="#292B60",
                plot_bgcolor="#292B60",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h"),
                xaxis_title="Date",
                yaxis_title="Sales (RM)",
                hovermode="x unified"
            )

            fig_trend.update_traces(
                hovertemplate="Date=%{x}<br>Sales=RM %{y:,.2f}<extra></extra>"
            )

            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown(f"""
        <div class="panel">
            <div class="card-title">Dataset Coverage</div>
            <div class="card-value-medium">{len(filtered_df):,} records</div>
            <div class="card-label">{start_date} to {end_date}</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        if not abs_filtered.empty:
            div_share = abs_filtered.groupby("division_label", as_index=False)["sales"].sum()

            fig_pie = px.pie(
                div_share,
                names="division_label",
                values="sales",
                hole=0.55,
                color_discrete_sequence=PALETTE,
                title="Division Sales Share"
            )

            fig_pie.update_layout(
                height=360,
                paper_bgcolor="#292B60",
                plot_bgcolor="#292B60",
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(orientation="h"),
                font=dict(color="#F5F6FF"),
                template="plotly_dark"
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(f"""
        <div class="panel">
            <div class="card-title">Average YoY Growth</div>
            <div class="card-value-medium">{avg_yoy_growth:,.2f}%</div>
            <div class="card-label">Year-on-year sales growth</div>
        </div>
        """, unsafe_allow_html=True)

    st.info("Executive insight: This page gives a quick view of sales value, trade movement, sector contribution, and growth direction.")


# =====================================================
# TAB 2: SALES TREND
# =====================================================

with tab2:

    st.markdown('<div class="top-strip">Sales and Volume Trend Analysis</div>', unsafe_allow_html=True)

    trend_df = filtered_df[filtered_df["series"] == "abs"].copy()

    if trend_df.empty:
        st.warning("Please select the 'abs' series to view sales trend.")
    else:
        chart_type = st.radio(
            "Choose chart type",
            ["Line Chart", "Area Chart"],
            horizontal=True
        )

        if chart_type == "Line Chart":
            fig_sales = px.line(
                trend_df,
                x="date",
                y="sales",
                color="division_label",
                markers=True,
                title="Monthly Sales Trend by Division",
                color_discrete_sequence=PALETTE,
                template="plotly_dark"
            )
        else:
            fig_sales = px.area(
                trend_df,
                x="date",
                y="sales",
                color="division_label",
                title="Monthly Sales Trend by Division",
                color_discrete_sequence=PALETTE,
                template="plotly_dark"
            )

        fig_sales.update_layout(
            height=520,
            paper_bgcolor="#292B60",
            plot_bgcolor="#292B60",
            hovermode="x unified",
            legend=dict(orientation="h"),
            xaxis_title="Date",
            yaxis_title="Sales (RM)"
        )

        fig_sales.update_traces(
            hovertemplate="Date=%{x}<br>Sales=RM %{y:,.2f}<extra></extra>"
        )

        st.plotly_chart(fig_sales, use_container_width=True)

        selected_volume_division = st.selectbox(
            "Select division for volume analysis",
            options=sorted(trend_df["division_label"].unique())
        )

        volume_df = trend_df[trend_df["division_label"] == selected_volume_division].copy()

        fig_volume = go.Figure()

        fig_volume.add_trace(go.Scatter(
            x=volume_df["date"],
            y=volume_df["volume"],
            mode="lines+markers",
            name="Volume",
            line=dict(color="#16D9FF", width=3)
        ))

        fig_volume.add_trace(go.Scatter(
            x=volume_df["date"],
            y=volume_df["volume_sa"],
            mode="lines+markers",
            name="Volume SA",
            line=dict(color="#F9D423", width=3)
        ))

        fig_volume.update_layout(
            title=f"Volume vs Seasonally Adjusted Volume: {selected_volume_division}",
            height=500,
            template="plotly_dark",
            paper_bgcolor="#292B60",
            plot_bgcolor="#292B60",
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Volume Index"
        )

        st.plotly_chart(fig_volume, use_container_width=True)

        st.success("Insight: Volume SA gives a smoother view of the underlying trend after seasonal effects are reduced.")


# =====================================================
# TAB 3: DIVISION COMPARISON
# =====================================================

with tab3:

    st.markdown('<div class="top-strip">Comparison Between Trade Divisions</div>', unsafe_allow_html=True)

    comparison_df = filtered_df[filtered_df["series"] == "abs"].copy()

    if comparison_df.empty:
        st.warning("Please select the 'abs' series to compare divisions.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            comparison_metric = st.selectbox(
                "Choose comparison metric",
                ["sales", "volume", "volume_sa"]
            )

        with c2:
            aggregation_method = st.selectbox(
                "Choose aggregation method",
                ["sum", "mean", "max", "min"]
            )

        if aggregation_method == "sum":
            division_summary = comparison_df.groupby("division_label", as_index=False)[comparison_metric].sum()
        elif aggregation_method == "mean":
            division_summary = comparison_df.groupby("division_label", as_index=False)[comparison_metric].mean()
        elif aggregation_method == "max":
            division_summary = comparison_df.groupby("division_label", as_index=False)[comparison_metric].max()
        else:
            division_summary = comparison_df.groupby("division_label", as_index=False)[comparison_metric].min()

        col_a, col_b = st.columns(2)

        with col_a:
            fig_bar = px.bar(
                division_summary,
                x="division_label",
                y=comparison_metric,
                color="division_label",
                text_auto=True,
                title=f"{aggregation_method.title()} {comparison_metric.title()} by Division",
                color_discrete_sequence=PALETTE,
                template="plotly_dark"
            )

            y_title = "Sales (RM)" if comparison_metric == "sales" else comparison_metric.title()

            fig_bar.update_layout(
                height=500,
                paper_bgcolor="#292B60",
                plot_bgcolor="#292B60",
                showlegend=False,
                xaxis_title="Division",
                yaxis_title=y_title
            )

            if comparison_metric == "sales":
                fig_bar.update_traces(
                    hovertemplate="Division=%{x}<br>Sales=RM %{y:,.2f}<extra></extra>"
                )

            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            fig_donut = px.pie(
                division_summary,
                names="division_label",
                values=comparison_metric,
                hole=0.48,
                title=f"Division Share Based on {comparison_metric.title()}",
                color_discrete_sequence=PALETTE
            )

            fig_donut.update_layout(
                height=500,
                paper_bgcolor="#292B60",
                plot_bgcolor="#292B60",
                template="plotly_dark",
                font=dict(color="#F5F6FF")
            )

            st.plotly_chart(fig_donut, use_container_width=True)

        st.success("Insight: Division comparison helps identify which trade division contributes the most to the selected indicator.")


# =====================================================
# TAB 4: GROWTH ANALYSIS
# =====================================================

with tab4:

    st.markdown('<div class="top-strip">Growth Analysis</div>', unsafe_allow_html=True)

    growth_type = st.selectbox(
        "Select growth type",
        ["growth_mom", "growth_yoy"]
    )

    growth_df = df[
        (df["series"] == growth_type) &
        (df["division_label"].isin(selected_divisions)) &
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    ].copy()

    if growth_df.empty:
        st.warning("No growth data available for the selected filters.")
    else:
        fig_growth = px.line(
            growth_df,
            x="date",
            y="sales",
            color="division_label",
            markers=True,
            title=f"Sales Growth Analysis: {growth_type}",
            color_discrete_sequence=PALETTE,
            template="plotly_dark"
        )

        fig_growth.update_layout(
            height=520,
            paper_bgcolor="#292B60",
            plot_bgcolor="#292B60",
            hovermode="x unified",
            legend=dict(orientation="h"),
            xaxis_title="Date",
            yaxis_title="Sales Growth (%)"
        )

        st.plotly_chart(fig_growth, use_container_width=True)

        growth_summary = (
            growth_df.groupby("division_label")["sales"]
            .agg(["mean", "max", "min", "std"])
            .reset_index()
        )

        st.subheader("Growth Summary")
        st.dataframe(growth_summary, use_container_width=True)

        st.info("Insight: Growth analysis helps detect recovery, slowdown, and unusual fluctuations.")


# =====================================================
# TAB 5: FORECASTING
# =====================================================

with tab5:

    st.markdown(
        '<div class="top-strip">12 Month Future Baseline Forecast</div>',
        unsafe_allow_html=True
    )

    forecast_file = "forecast_output.csv"

    if not os.path.exists(forecast_file):
        st.error("forecast_output.csv not found. Please upload the forecast output file to GitHub.")
        st.stop()

    saved_forecast = pd.read_csv(forecast_file)

    required_columns = ["division", "division_label", "date", "forecasted_sales"]

    missing_columns = [
        col for col in required_columns
        if col not in saved_forecast.columns
    ]

    if missing_columns:
        st.error(f"forecast_output.csv is missing these columns: {missing_columns}")
        st.stop()

    saved_forecast["date"] = pd.to_datetime(saved_forecast["date"], errors="coerce")

    saved_forecast["division_label"] = (
        saved_forecast["division_label"]
        .astype(str)
        .str.strip()
    )

    saved_forecast["forecasted_sales"] = (
        saved_forecast["forecasted_sales"]
        .astype(str)
        .str.replace("RM", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    saved_forecast["forecasted_sales"] = pd.to_numeric(
        saved_forecast["forecasted_sales"],
        errors="coerce"
    )

    saved_forecast = saved_forecast.dropna(
        subset=["division_label", "date", "forecasted_sales"]
    )

    forecast_base = df[df["series"] == "abs"].copy()

    selected_forecast_division = st.selectbox(
        "Select division for forecast display",
        options=sorted(saved_forecast["division_label"].dropna().unique())
    )

    historical_df = forecast_base[
        forecast_base["division_label"] == selected_forecast_division
    ].copy()

    historical_df = historical_df.sort_values("date")

    forecast_df = saved_forecast[
        saved_forecast["division_label"] == selected_forecast_division
    ].copy()

    forecast_df = forecast_df.sort_values("date")

    if forecast_df.empty:
        st.error(
            f"No forecast values found for {selected_forecast_division}. "
            "Please check forecast_output.csv."
        )
        st.stop()

    forecast_df["forecasted_sales_rm"] = forecast_df["forecasted_sales"].apply(
        lambda x: fmt_rm(x, 2)
    )

    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=historical_df["date"],
        y=historical_df["sales"],
        mode="lines+markers",
        name="Actual Historical Sales",
        line=dict(color="#16D9FF", width=3),
        marker=dict(size=6),
        hovertemplate="Date=%{x}<br>Actual Sales=RM %{y:,.2f}<extra></extra>"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["forecasted_sales"],
        mode="lines+markers",
        name="Baseline Future Forecast",
        line=dict(color="#F9D423", width=3, dash="dash"),
        marker=dict(size=8),
        hovertemplate="Date=%{x}<br>Forecasted Sales=RM %{y:,.2f}<extra></extra>"
    ))

    forecast_start = forecast_df["date"].min()

    fig_forecast.add_vline(
        x=forecast_start,
        line_width=2,
        line_dash="dot",
        line_color="#FF4D6D"
    )

    min_y = min(
        historical_df["sales"].min(),
        forecast_df["forecasted_sales"].min()
    )

    max_y = max(
        historical_df["sales"].max(),
        forecast_df["forecasted_sales"].max()
    )

    y_padding = (max_y - min_y) * 0.15

    fig_forecast.update_layout(
        title=f"{selected_forecast_division} 12 Month Future Baseline Forecast",
        height=520,
        template="plotly_dark",
        paper_bgcolor="#292B60",
        plot_bgcolor="#292B60",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Sales (RM)",
        yaxis=dict(
            range=[min_y - y_padding, max_y + y_padding]
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98
        )
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("Forecasted Sales Table")

    st.dataframe(
        forecast_df[["division_label", "date", "forecasted_sales_rm"]],
        use_container_width=True
    )

    forecast_csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Forecast Result",
        data=forecast_csv,
        file_name="forecasted_sales.csv",
        mime="text/csv"
    )

    st.success(
        "The forecast values are loaded from forecast_output.csv, so the dashboard uses the same forecast values provided by the original forecasting result."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<div class="panel">
    <div class="card-title">Data Product Deployment Value</div>
    <div class="card-label">
        <b>Data Product:</b> Wholesale & Retail Trade in Malaysia Dashboard<br><br>
        <b>Interactive Features:</b><br>
        ✅ Filter by date range, division, and series<br>
        ✅ View executive-style KPI cards with RM sales value<br>
        ✅ Compare trade divisions visually<br>
        ✅ Analyse month-on-month and year-on-year growth<br>
        ✅ Forecast future sales<br>
        ✅ Download forecast results<br><br>
        <b>Decision-Making Value:</b><br>
        This dashboard helps authorities and analysts identify growing or declining sectors, monitor trade performance, 
        and support evidence-based planning under <b>SDG 8: Decent Work and Economic Growth</b>.
    </div>
</div>
""", unsafe_allow_html=True)
