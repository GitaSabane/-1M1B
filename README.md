🌬️ Wind Energy Feasibility Dashboard
📌 Project Overview

The Wind Energy Feasibility Dashboard is a Streamlit-based web application that analyzes historical wind speed data, forecasts future wind trends using time series modeling, and estimates potential energy production from wind turbines.

This tool helps evaluate whether a location is suitable for wind energy generation by combining:

📊 Data analysis

🔮 Time series forecasting

⚡ Power output estimation

📈 Interactive visualizations

🎯 Project Objectives

Analyze historical wind speed data

Forecast future wind speeds using machine learning

Estimate wind turbine power generation

Calculate capacity factor and annual energy production

Provide interactive and visual insights through a web dashboard

🛠️ Technologies Used

Python 3.14

Streamlit – Web application framework

Prophet – Time series forecasting model

Pandas & NumPy – Data processing

Plotly & Matplotlib – Data visualization

🧠 How the Project Works
1️⃣ Data Input

Users can:

Upload their own CSV file (with ds and y columns)

Or use generated sample wind speed data

Where:

ds = Date

y = Wind speed (m/s)

2️⃣ Forecasting (Machine Learning)

The app uses Prophet, a time series forecasting model, to:

Learn seasonal patterns

Detect trends

Predict future wind speeds

Forecast values are adjusted to prevent negative wind speeds.

3️⃣ Wind Turbine Energy Calculation

Power output is calculated using a simplified wind turbine power curve:

Below cut-in speed → 0 power

Between cut-in and rated speed → Cubic growth

Above rated speed → Constant rated power

Above cut-out speed → 0 power

Then the app calculates:

⚡ Average Power Output (kW)

📊 Capacity Factor (%)

🔋 Estimated Annual Energy Production (MWh)

📈 Dashboard Features
📊 Data Overview

Average wind speed

Maximum wind speed

Interactive wind speed chart

Raw data table

🔮 Forecast Section

Future wind speed prediction

Trend & seasonality breakdown

Interactive forecast visualization

⚡ Energy Estimation

Turbine model selection

Custom turbine configuration

Power vs Wind speed visualization

Capacity factor calculation

🚀 Deployment

The application is deployed using:

Render (Cloud hosting platform)

Streamlit server configuration:

streamlit run app.py --server.port $PORT --server.address 0.0.0.0
📂 Project Structure
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
🌍 Real-World Use Case

This project can be useful for:

Renewable energy feasibility studies

Academic research projects

Wind farm pre-analysis

Engineering simulations

Data science portfolio demonstration

🔥 Future Improvements

Add multiple forecasting model comparison

Include real weather API integration

Add economic cost-benefit analysis

Improve UI/UX design

Add downloadable PDF report generation

👨‍💻 Author

Your Name
Wind Energy & Data Science Project
Built with ❤️ using Python & Streamlit
