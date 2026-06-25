from utils.forecasting.forecasting_engine import forecast

results = forecast(
    df=df,
    target="Sales",
    model="Prophet",
    periods=30
)
