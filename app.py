import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Wind Energy Feasibility Dashboard",
    page_icon="💨",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    font-weight: 700;
}
.block-container {
    padding-top: 2rem;
}
.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}
.sidebar .sidebar-content {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
# 🌬️ Wind Energy Feasibility Dashboard
### Smart Wind Analysis & Energy Forecasting System
""")

st.markdown("---")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📊 Data Configuration")

    data_source = st.radio(
        "Choose Data Source:",
        ["Use Sample Data", "Upload CSV"]
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Upload CSV with 'ds' and 'y' columns")
            st.stop()
    else:
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        wind = np.random.normal(7.5, 2.5, len(dates))
        seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        wind = wind + seasonal + np.random.normal(0, 1, len(dates))

        df = pd.DataFrame({
            "ds": dates,
            "y": np.maximum(wind, 0)
        })

        st.success("Sample dataset loaded")

    st.markdown("---")
    st.header("⚙️ Turbine Settings")

    turbine_model = st.selectbox(
        "Select Turbine Model:",
        ["Generic 1.5 MW", "GE 2.5-120", "Custom"]
    )

    turbine_specs = {
        "Generic 1.5 MW": [3.0, 12.0, 25.0, 1500],
        "GE 2.5-120": [3.5, 12.5, 25.0, 2500]
    }

    if turbine_model != "Custom":
        cut_in_speed, rated_speed, cut_out_speed, rated_power = turbine_specs[turbine_model]
    else:
        cut_in_speed = st.slider("Cut-in Speed", 2.0, 5.0, 3.0)
        rated_speed = st.slider("Rated Speed", 10.0, 15.0, 12.0)
        cut_out_speed = st.slider("Cut-out Speed", 20.0, 25.0, 25.0)
        rated_power = st.number_input("Rated Power (kW)", 1000, 5000, 1500)

    st.markdown("---")
    st.header("🔮 Forecast Settings")
    forecast_days = st.slider("Days to Forecast", 30, 365, 90)

# ------------------ VALIDATION ------------------
if "ds" not in df.columns or "y" not in df.columns:
    st.error("CSV must contain 'ds' and 'y' columns.")
    st.stop()

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds")
df["time_index"] = np.arange(len(df))

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🔍 Forecast", "⚡ Energy Estimation"])

# ================= TAB 1 =================
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Wind Speed", f"{df['y'].mean():.2f} m/s")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Maximum Wind Speed", f"{df['y'].max():.2f} m/s")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Wind Speed Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["ds"],
        y=df["y"],
        mode="lines",
        name="Wind Speed",
        line=dict(width=3)
    ))

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df[["ds", "y"]], use_container_width=True)

# ================= TAB 2 =================
with tab2:

    st.markdown("### Polynomial Regression Forecast")

    if st.button("Generate Forecast"):

        with st.spinner("Training model..."):

            model = Pipeline([
                ("poly", PolynomialFeatures(degree=3)),
                ("lr", LinearRegression())
            ])

            X = df[["time_index"]]
            y = df["y"]
            model.fit(X, y)

            future_index = np.arange(len(df) + forecast_days)
            future_dates = pd.date_range(
                start=df["ds"].iloc[0],
                periods=len(future_index),
                freq="D"
            )

            future_df = pd.DataFrame({"time_index": future_index})
            predictions = model.predict(future_df)
            predictions = np.maximum(predictions, 0)

            forecast = pd.DataFrame({
                "ds": future_dates,
                "yhat": predictions
            })

            st.session_state.forecast = forecast

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df["ds"],
                y=df["y"],
                name="Historical",
                line=dict(width=3)
            ))

            fig.add_trace(go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                name="Forecast",
                line=dict(dash="dash", width=3)
            ))

            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3 =================
with tab3:

    if "forecast" not in st.session_state:
        st.warning("Please generate forecast first.")
    else:
        forecast = st.session_state.forecast

        def calculate_power(ws):
            if ws < cut_in_speed or ws > cut_out_speed:
                return 0
            elif ws < rated_speed:
                return rated_power * ((ws - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
            else:
                return rated_power

        forecast["power_kW"] = forecast["yhat"].apply(calculate_power)

        avg_power = forecast["power_kW"].mean()
        capacity_factor = (avg_power / rated_power) * 100
        annual_energy = avg_power * 24 * 365 / 1000

        c1, c2, c3 = st.columns(3)

        c1.metric("Average Power (kW)", f"{avg_power:.0f}")
        c2.metric("Capacity Factor (%)", f"{capacity_factor:.2f}")
        c3.metric("Annual Energy (MWh)", f"{annual_energy:.0f}")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            name="Wind Speed"
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=forecast["ds"],
            y=forecast["power_kW"],
            name="Power Output"
        ), secondary_y=True)

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=500
        )

        fig.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
        fig.update_yaxes(title_text="Power Output (kW)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Wind Energy Feasibility Dashboard | Built with Streamlit | Python 3.14 Compatible")
