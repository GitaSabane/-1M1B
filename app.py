import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

warnings.filterwarnings("ignore")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Wind Energy Feasibility Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS FOR BETTER UI ------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient text for headers */
    .gradient-text {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    /* Card styling */
    .css-1r6slb0 {
        border-radius: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 0.5rem;
        border-radius: 50px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    /* Dataframe styling */
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success/Warning/Info messages */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .gradient-text {
            font-size: 1.8rem;
        }
        .metric-card {
            margin-bottom: 1rem;
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea transparent #667eea transparent !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px 50px 10px 10px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER SECTION ------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<h1 class="gradient-text">🌪️ Wind Energy Feasibility Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>
    Analyze historical wind speed data, forecast future trends, and estimate wind turbine energy production with our advanced analytics platform.
    </p>
    """, unsafe_allow_html=True)

with col2:
    # Current time display
    current_time = datetime.now().strftime("%B %d, %Y %H:%M")
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
        <p style='margin: 0; font-size: 0.9rem;'>🕐 Last Updated</p>
        <p style='margin: 0; font-size: 1.1rem; font-weight: 600;'>{current_time}</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------ SIDEBAR WITH ENHANCED UI ------------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #667eea;'>⚙️ Configuration Panel</h2>
        <p style='color: #666; font-size: 0.9rem;'>Customize your analysis</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 Data Source", expanded=True):
        data_source = st.radio(
            "Choose Data Source:",
            ["📁 Use Sample Data", "📤 Upload CSV"],
            help="Select sample data or upload your own CSV file"
        )

        if data_source == "📤 Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type=["csv"],
                help="File must contain 'ds' (date) and 'y' (wind speed) columns"
            )

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
            else:
                st.info("📁 Please upload a CSV file.")
                st.stop()
        else:
            # Generate realistic sample data
            np.random.seed(42)
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
            
            # More realistic wind patterns
            base_wind = 7.5
            seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365 - np.pi/2)
            monthly_pattern = 2 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
            noise = np.random.normal(0, 1.5, len(dates))
            
            wind = base_wind + seasonal + monthly_pattern + noise
            
            df = pd.DataFrame({
                "ds": dates,
                "y": np.maximum(wind, 0.5)  # Ensure minimum wind speed
            })

            st.success("🎯 Sample data loaded successfully!")
            st.info("📊 Data includes seasonal and monthly patterns")

    with st.expander("⚡ Turbine Specifications", expanded=True):
        turbine_model = st.selectbox(
            "Select Turbine Model:",
            ["🎯 Generic 1.5 MW", "🏭 GE 2.5-120", "⚙️ Custom"],
            help="Choose a predefined turbine or customize your own"
        )

        turbine_specs = {
            "🎯 Generic 1.5 MW": [3.0, 12.0, 25.0, 1500],
            "🏭 GE 2.5-120": [3.5, 12.5, 25.0, 2500]
        }

        if turbine_model != "⚙️ Custom":
            cut_in_speed, rated_speed, cut_out_speed, rated_power = turbine_specs[turbine_model]
            
            # Display turbine specs nicely
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cut-in Speed", f"{cut_in_speed} m/s")
                st.metric("Cut-out Speed", f"{cut_out_speed} m/s")
            with col2:
                st.metric("Rated Speed", f"{rated_speed} m/s")
                st.metric("Rated Power", f"{rated_power} kW")
        else:
            cut_in_speed = st.slider("Cut-in Speed (m/s)", 2.0, 5.0, 3.0, 0.1)
            rated_speed = st.slider("Rated Speed (m/s)", 10.0, 15.0, 12.0, 0.1)
            cut_out_speed = st.slider("Cut-out Speed (m/s)", 20.0, 25.0, 25.0, 0.1)
            rated_power = st.number_input("Rated Power (kW)", 1000, 5000, 1500, 100)

    with st.expander("🔮 Forecast Settings", expanded=True):
        forecast_days = st.slider(
            "Forecast Horizon (Days)", 
            30, 365, 90,
            help="Number of days to forecast into the future"
        )
        
        confidence_level = st.select_slider(
            "Confidence Level",
            options=[80, 85, 90, 95, 99],
            value=95,
            help="Confidence interval for predictions"
        )

# ------------------ DATA VALIDATION ------------------
if "ds" not in df.columns or "y" not in df.columns:
    st.error("🚨 CSV must contain 'ds' and 'y' columns.")
    st.stop()

# Data preprocessing
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)
df["time_index"] = np.arange(len(df))
df["month"] = df["ds"].dt.month
df["year"] = df["ds"].dt.year

# ------------------ TABS WITH ENHANCED UI ------------------
tab1, tab2, tab3 = st.tabs(["📈 Data Overview", "🔮 Forecast Analysis", "⚡ Energy Estimation"])

# ================= TAB 1: DATA OVERVIEW =================
with tab1:
    # Key metrics in attractive cards
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Average Wind Speed</h3>
            <p style='margin:0; font-size:2.5rem; font-weight:700;'>{df['y'].mean():.1f}</p>
            <p style='margin:0; font-size:0.9rem;'>m/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #f6b23d 0%, #f5a623 100%);'>
            <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Max Wind Speed</h3>
            <p style='margin:0; font-size:2.5rem; font-weight:700;'>{df['y'].max():.1f}</p>
            <p style='margin:0; font-size:0.9rem;'>m/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #a8e6cf 0%, #3cb371 100%);'>
            <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Min Wind Speed</h3>
            <p style='margin:0; font-size:2.5rem; font-weight:700;'>{df['y'].min():.1f}</p>
            <p style='margin:0; font-size:0.9rem;'>m/s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        std_dev = df['y'].std()
        st.markdown(f"""
        <div class="metric-card" style='background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);'>
            <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Std Deviation</h3>
            <p style='margin:0; font-size:2.5rem; font-weight:700;'>{std_dev:.1f}</p>
            <p style='margin:0; font-size:0.9rem;'>m/s</p>
        </div>
        """, unsafe_allow_html=True)

    # Time series plot with enhanced styling
    st.markdown("### 📈 Historical Wind Speed Analysis")
    
    fig1 = go.Figure()
    
    # Add main time series
    fig1.add_trace(go.Scatter(
        x=df["ds"], 
        y=df["y"], 
        mode="lines",
        name="Wind Speed",
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='<b>Date</b>: %{x}<br><b>Wind Speed</b>: %{y:.2f} m/s<extra></extra>'
    ))
    
    # Add rolling average
    rolling_mean = df['y'].rolling(window=30).mean()
    fig1.add_trace(go.Scatter(
        x=df["ds"], 
        y=rolling_mean,
        mode="lines",
        name="30-Day Average",
        line=dict(color='#f6b23d', width=2, dash='dash'),
        hovertemplate='<b>Date</b>: %{x}<br><b>30-Day Avg</b>: %{y:.2f} m/s<extra></extra>'
    ))
    
    fig1.update_layout(
        title=dict(
            text="Wind Speed Time Series with Moving Average",
            x=0.5,
            font=dict(size=20, color='#333')
        ),
        xaxis_title="Date",
        yaxis_title="Wind Speed (m/s)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Dataset preview with styling
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Dataset Preview")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        
        # Style the dataframe
        styled_df = df[['ds', 'y']].copy()
        styled_df.columns = ['Date', 'Wind Speed (m/s)']
        styled_df['Date'] = styled_df['Date'].dt.date
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Wind Speed (m/s)": st.column_config.NumberColumn("Wind Speed (m/s)", format="%.2f")
            }
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Summary Statistics")
        
        # Create summary statistics
        stats = df['y'].describe().round(2)
        stats_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [stats['count'], stats['mean'], stats['std'], 
                     stats['min'], stats['25%'], stats['50%'], 
                     stats['75%'], stats['max']]
        })
        
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Statistic": "Statistic",
                "Value": st.column_config.NumberColumn("Value", format="%.2f")
            }
        )
        
        # Download button for dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name=f"wind_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# ================= TAB 2: FORECAST ANALYSIS =================
with tab2:
    st.markdown("### 🔮 Advanced Wind Speed Forecasting")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        model_degree = st.selectbox(
            "Polynomial Degree",
            options=[2, 3, 4],
            index=1,
            help="Higher degree may overfit the data"
        )
    
    with col2:
        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    
    with col3:
        forecast_button = st.button("🚀 Generate Forecast", use_container_width=True)

    if forecast_button:
        with st.spinner("🔄 Generating forecast... This may take a moment."):
            # Polynomial regression model
            model = Pipeline([
                ("poly", PolynomialFeatures(degree=model_degree)),
                ("lr", LinearRegression())
            ])

            X = df[["time_index"]]
            y = df["y"]

            model.fit(X, y)

            # Create future time indices
            future_index = np.arange(len(df) + forecast_days)
            future_dates = pd.date_range(
                start=df["ds"].iloc[0],
                periods=len(future_index),
                freq="D"
            )

            # Make predictions
            future_df = pd.DataFrame({"time_index": future_index})
            predictions = model.predict(future_df)
            predictions = np.maximum(predictions, 0)

            # Calculate confidence intervals (simplified)
            residuals = y - model.predict(X)
            std_residuals = np.std(residuals)
            z_score = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
            
            confidence_upper = predictions + z_score * std_residuals
            confidence_lower = predictions - z_score * std_residuals
            confidence_lower = np.maximum(confidence_lower, 0)

            # Create forecast dataframe
            forecast = pd.DataFrame({
                "ds": future_dates,
                "yhat": predictions,
                "yhat_lower": confidence_lower,
                "yhat_upper": confidence_upper
            })

            # Store in session state
            st.session_state.forecast = forecast
            st.session_state.forecast_generated = True
            
            st.success("✅ Forecast generated successfully!")

    # Check if forecast exists
    if "forecast" in st.session_state:
        forecast = st.session_state.forecast
        
        # Forecast metrics
        st.markdown("### 📊 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Forecast Mean</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{forecast['yhat'].mean():.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>m/s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #f6b23d 0%, #f5a623 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Forecast Max</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{forecast['yhat'].max():.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>m/s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            trend = forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[len(df)]
            trend_color = "green" if trend > 0 else "red"
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #a8e6cf 0%, #3cb371 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Trend</h3>
                <p style='margin:0; font-size:2rem; font-weight:700; color: {trend_color};'>{trend:+.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>m/s change</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            uncertainty = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Uncertainty</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>±{uncertainty:.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>m/s (95% CI)</p>
            </div>
            """, unsafe_allow_html=True)

        # Enhanced forecast plot
        fig2 = go.Figure()
        
        # Historical data
        fig2.add_trace(go.Scatter(
            x=df["ds"], 
            y=df["y"], 
            mode="lines",
            name="Historical",
            line=dict(color='#667eea', width=3),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Wind Speed: %{y:.2f} m/s<extra></extra>'
        ))
        
        # Forecast data
        fig2.add_trace(go.Scatter(
            x=forecast["ds"], 
            y=forecast["yhat"], 
            mode="lines",
            name="Forecast",
            line=dict(color='#f6b23d', width=3),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Wind Speed: %{y:.2f} m/s<extra></extra>'
        ))
        
        # Confidence intervals
        if show_confidence:
            fig2.add_trace(go.Scatter(
                x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
                y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(246, 178, 61, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Add vertical line
        fig2.add_vline(
            x=df["ds"].iloc[-1], 
            line_width=2, 
            line_dash="dash", 
            line_color="#666",
            annotation_text="Forecast Start",
            annotation_position="top right",
            annotation_font=dict(size=12)
        )
        
        fig2.update_layout(
            title=dict(
                text=f"Wind Speed Forecast (Polynomial Degree {model_degree})",
                x=0.5,
                font=dict(size=20, color='#333')
            ),
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)",
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Forecast dataset
        st.markdown("### 📋 Forecast Dataset")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            forecast_display = forecast.copy()
            forecast_display['ds'] = forecast_display['ds'].dt.date
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            
            st.dataframe(
                forecast_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Forecast": st.column_config.NumberColumn("Forecast (m/s)", format="%.2f"),
                    "Lower Bound": st.column_config.NumberColumn(f"Lower {confidence_level}%", format="%.2f"),
                    "Upper Bound": st.column_config.NumberColumn(f"Upper {confidence_level}%", format="%.2f")
                }
            )
        
        with col2:
            st.markdown("### ⬇️ Download")
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="📥 Forecast CSV",
                data=csv,
                file_name=f"wind_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if st.button("🔄 Reset Forecast", use_container_width=True):
                del st.session_state.forecast
                st.rerun()
    else:
        # Placeholder for forecast tab
        st.info("👆 Click 'Generate Forecast' to see advanced predictions with confidence intervals")
        
        # Show sample forecast visualization
        fig_placeholder = go.Figure()
        fig_placeholder.add_annotation(
            text="Generate a forecast to see predictions<br>with confidence intervals",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="#666")
        )
        fig_placeholder.update_layout(
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig_placeholder, use_container_width=True)

# ================= TAB 3: ENERGY ESTIMATION =================
with tab3:
    st.markdown("### ⚡ Energy Production Estimation")
    
    if "forecast" not in st.session_state:
        st.warning("⚠️ Please generate a forecast first in the Forecast Analysis tab")
        
        # Show power curve explanation
        st.markdown("""
        <div style='background: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h4>📈 Power Curve Analysis</h4>
            <p>After generating a forecast, you'll see:</p>
            <ul style='list-style-type: none; padding: 0;'>
                <li>✅ Real-time power output calculations</li>
                <li>✅ Capacity factor estimation</li>
                <li>✅ Daily and annual energy projections</li>
                <li>✅ Interactive wind speed vs power plots</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        forecast = st.session_state.forecast
        
        # Power calculation function with cubic curve
        def calculate_power(ws):
            if ws < cut_in_speed or ws > cut_out_speed:
                return 0
            elif ws < rated_speed:
                # Cubic power curve
                return rated_power * ((ws - cut_in_speed) / (rated_speed - cut_in_speed)) ** 3
            else:
                return rated_power

        # Calculate power for each forecast point
        forecast["power_kW"] = forecast["yhat"].apply(calculate_power)
        
        # Calculate confidence intervals for power
        forecast["power_lower"] = forecast["yhat_lower"].apply(calculate_power)
        forecast["power_upper"] = forecast["yhat_upper"].apply(calculate_power)

        # Energy metrics
        avg_power = forecast["power_kW"].mean()
        capacity_factor = (avg_power / rated_power) * 100 if rated_power > 0 else 0
        daily_energy = avg_power * 24 / 1000  # MWh
        annual_energy = avg_power * 24 * 365 / 1000  # MWh
        total_energy = forecast["power_kW"].sum() * 24 / 1000  # Total MWh over forecast period

        # Energy metrics in attractive cards
        st.markdown("### 📊 Energy Production Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Average Power</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{avg_power:.0f}</p>
                <p style='margin:0; font-size:0.9rem;'>kW</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #f6b23d 0%, #f5a623 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Capacity Factor</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{capacity_factor:.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #a8e6cf 0%, #3cb371 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Daily Energy</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{daily_energy:.1f}</p>
                <p style='margin:0; font-size:0.9rem;'>MWh</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style='background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);'>
                <h3 style='margin:0; font-size:1rem; opacity:0.9;'>Annual Energy</h3>
                <p style='margin:0; font-size:2rem; font-weight:700;'>{annual_energy:.0f}</p>
                <p style='margin:0; font-size:0.9rem;'>MWh/year</p>
            </div>
            """, unsafe_allow_html=True)

        # Power curve visualization
        st.markdown("### 📈 Wind Speed vs Power Output Analysis")
        
        # Create power curve for visualization
        ws_range = np.linspace(0, cut_out_speed + 5, 100)
        power_curve = [calculate_power(ws) for ws in ws_range]
        
        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Power Curve", "Time Series Analysis"),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Power curve plot
        fig3.add_trace(
            go.Scatter(
                x=ws_range,
                y=power_curve,
                mode="lines",
                name="Power Curve",
                line=dict(color='#667eea', width=3),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='<b>Wind Speed</b>: %{x:.1f} m/s<br><b>Power</b>: %{y:.0f} kW<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add vertical lines for cut-in, rated, cut-out
        fig3.add_vline(x=cut_in_speed, line_dash="dash", line_color="green", 
                      annotation_text="Cut-in", row=1, col=1)
        fig3.add_vline(x=rated_speed, line_dash="dash", line_color="orange", 
                      annotation_text="Rated", row=1, col=1)
        fig3.add_vline(x=cut_out_speed, line_dash="dash", line_color="red", 
                      annotation_text="Cut-out", row=1, col=1)
        
        # Time series of power output
        fig3.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["power_kW"],
                mode="lines",
                name="Power Output",
                line=dict(color='#3cb371', width=2),
                fill='tozeroy',
                fillcolor='rgba(60, 179, 113, 0.1)',
                hovertemplate='<b>Date</b>: %{x}<br><b>Power</b>: %{y:.0f} kW<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add confidence bands for power
        fig3.add_trace(
            go.Scatter(
                x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
                y=forecast["power_upper"].tolist() + forecast["power_lower"].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(60, 179, 113, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Power Uncertainty',
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        fig3.update_layout(
            height=500,
            showlegend=True,
            template='plotly_white',
            title=dict(
                text=f"Energy Production Analysis - {turbine_model}",
                x=0.5,
                font=dict(size=16)
            )
        )
        
        fig3.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
        fig3.update_yaxes(title_text="Power Output (kW)", row=1, col=1)
        fig3.update_xaxes(title_text="Date", row=1, col=2)
        fig3.update_yaxes(title_text="Power Output (kW)", row=1, col=2)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Energy production dataset
        st.markdown("### 📋 Energy Production Dataset")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            energy_data = forecast[['ds', 'yhat', 'power_kW', 'power_lower', 'power_upper']].copy()
            energy_data['ds'] = energy_data['ds'].dt.date
            energy_data.columns = ['Date', 'Wind Speed (m/s)', 'Power (kW)', 'Power Lower (kW)', 'Power Upper (kW)']
            
            st.dataframe(
                energy_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date"),
                    "Wind Speed (m/s)": st.column_config.NumberColumn("Wind Speed (m/s)", format="%.2f"),
                    "Power (kW)": st.column_config.NumberColumn("Power (kW)", format="%.0f"),
                    "Power Lower (kW)": st.column_config.NumberColumn("Lower Bound (kW)", format="%.0f"),
                    "Power Upper (kW)": st.column_config.NumberColumn("Upper Bound (kW)", format="%.0f")
                }
            )
        
        with col2:
            st.markdown("### 📊 Summary")
            
            # Additional metrics
            operating_hours = (forecast['power_kW'] > 0).sum() / len(forecast) * 100
            peak_power_hours = (forecast['power_kW'] > rated_power * 0.9).sum()
            
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                <p><b>⏱️ Operating Time:</b> {operating_hours:.1f}%</p>
                <p><b>⚡ Peak Hours:</b> {peak_power_hours} days</p>
                <p><b>📊 Total Energy:</b> {total_energy:.0f} MWh</p>
                <p><b>💰 Est. Revenue:</b> ${(total_energy * 50):,.0f}</p>
                <p style='font-size:0.8rem; color:#666;'>*Assuming $50/MWh</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download button
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download Energy Data",
                data=csv,
                file_name=f"energy_production_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    <p style='margin:0; font-size:1.2rem;'>🌬️ Wind Energy Feasibility Dashboard</p>
    <p style='margin:0; font-size:0.9rem; opacity:0.9;'>Powered by Advanced Analytics | Real-time Predictions | Energy Optimization</p>
    <p style='margin:0; font-size:0.8rem; opacity:0.8; margin-top:1rem;'>© 2026 | Built with Streamlit | Python 3.14 Compatible</p>
</div>
""", unsafe_allow_html=True)

# Add auto-refresh capability (optional)
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()
