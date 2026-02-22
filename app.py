import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Wind Energy Feasibility Dashboard",
    page_icon="💨",
    layout="wide"
)

# ------------------ TITLE ------------------
st.title("🌬️ Wind Energy Feasibility Dashboard")
st.markdown("""
Analyze historical wind speed data, forecast future trends, and estimate
wind turbine energy production.
""")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("📊 Data Input")

    data_source = st.radio(
        "Choose Data Source:",
        ["Use Sample Data", "Upload CSV"]
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file.")
            st.stop()
    else:
        # Generate sample dataset
        dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
        wind = np.random.normal(7.5, 2.5, len(dates))
        seasonal = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
        wind = wind + seasonal + np.random.normal(0, 1, len(dates))

        df = pd.DataFrame({
            "ds": dates,
            "y": np.maximum(wind, 0)
        })

        st.success("Sample data loaded successfully!")

    # ------------------ TURBINE SETTINGS ------------------
    st.header("⚙️ Turbine Specifications")

    turbine_model = st.selectbox(
        "Select Turbine Model:",
        ["Generic 1.5 MW", "Vestas V90-2.0 MW", "GE 2.5-120", "Custom"]
    )

    turbine_specs = {
        "Generic 1.5 MW": [3.0, 12.0, 25.0, 1500],
        "Vestas V90-2.0 MW": [4.0, 13.0, 25.0, 2000],
        "GE 2.5-120": [3.5, 12.5, 25.0, 2500]
    }

    if turbine_model != "Custom":
        cut_in_speed, rated_speed, cut_out_speed, rated_power = turbine_specs[turbine_model]
    else:
        cut_in_speed = st.slider("Cut-in Speed (m/s)", 2.0, 5.0, 3.0)
        rated_speed = st.slider("Rated Speed (m/s)", 10.0, 15.0, 12.0)
        cut_out_speed = st.slider("Cut-out Speed (m/s)", 20.0, 25.0, 25.0)
        rated_power = st.number_input("Rated Power (kW)", 1000, 5000, 1500)

    # Forecast settings
    st.header("🔮 Forecast Settings")
    forecast_days = st.slider("Days to Forecast", 30, 365, 90)

# ------------------ DATA VALIDATION ------------------
if "ds" not in df.columns or "y" not in df.columns:
    st.error("CSV must contain 'ds' (date) and 'y' (wind speed) columns.")
    st.stop()

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds")

# ------------------ TABS ------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Data Overview",
    "🔍 Forecast",
    "⚡ Energy Estimation"
])

# ================= TAB 1 =================
with tab1:
    st.subheader("Historical Wind Data")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Wind Speed", f"{df['y'].mean():.2f} m/s")
    col2.metric("Maximum Wind Speed", f"{df['y'].max():.2f} m/s")
    col3.metric("Total Data Points", len(df))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"], mode="lines", name="Wind Speed"))
    fig.update_layout(
        title="Historical Wind Speed",
        xaxis_title="Date",
        yaxis_title="Wind Speed (m/s)"
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.checkbox("Show Raw Data"):
        st.dataframe(df)

# ================= TAB 2 =================
with tab2:
    st.subheader("Wind Speed Forecast")

    if st.button("Generate Forecast"):

        with st.spinner("Training Prophet model..."):

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )

            model.fit(df)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            forecast["yhat"] = np.maximum(forecast["yhat"], 0)

            st.session_state.forecast = forecast

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

# ================= TAB 3 =================
with tab3:
    st.subheader("Energy Production Estimation")

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

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Power", f"{avg_power:.0f} kW")
        col2.metric("Capacity Factor", f"{capacity_factor:.2f}%")
        col3.metric("Estimated Annual Energy", f"{annual_energy:.0f} MWh")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Wind Speed"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=forecast["ds"], y=forecast["power_kW"], name="Power Output"),
            secondary_y=True
        )

        fig.update_layout(title="Wind Speed vs Power Output")
        fig.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
        fig.update_yaxes(title_text="Power Output (kW)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.caption("Wind Energy Feasibility Dashboard | Built with Streamlit")
