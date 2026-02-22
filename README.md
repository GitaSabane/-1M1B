🌬️ Wind Energy Feasibility Dashboard

A Streamlit web app that analyzes wind speed data, predicts future wind trends, and estimates wind turbine power generation.
🔗 Live App: https://1m1b.onrender.com/

📌 About the Project
This dashboard helps evaluate whether a location is suitable for wind energy production.
It combines wind data analysis, forecasting, and turbine energy estimation in one interactive interface.
Users can upload their own dataset or use sample data to:
   Analyze historical wind speed
   Forecast future wind speeds
   Estimate turbine power output
   Calculate capacity factor and annual energy generation

Built With
   Python 3.14
   Streamlit
   Prophet (time series forecasting)
   Pandas & NumPy
   Plotly & Matplotlib

⚙️ How It Works

   Data Input – Upload CSV file (ds for date, y for wind speed) or use sample data.
   Forecasting – Prophet model predicts future wind speed trends.
   Energy Estimation – Power is calculated using turbine specifications (cut-in, rated, cut-out speeds).
   The dashboard then shows:
     Average wind speed
     Forecast graphs
     Estimated power output
     Capacity factor
     Annual energy production

🚀 Deployment

Hosted on Render using:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

📂 Project Structure
    app.py
    requirements.txt
    runtime.txt
    README.md
    
👩‍💻 Author
    Gitabai Sabane
