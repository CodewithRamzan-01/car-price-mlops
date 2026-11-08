import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

cars = pd.read_csv("cars.csv")
features = cars[["age_years", "mileage_km", "engine_cc"]]
prices = cars["price_lakh"]

features_train, features_test, prices_train, prices_test = train_test_split(
    features, prices, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("car-price-prediction")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(features_train, prices_train)
        predictions = model.predict(features_test)
        mae = mean_absolute_error(prices_test, predictions)

        mlflow.log_param("model_type", name)
        mlflow.log_metric("mae_lakh", mae)
        print(f"{name}: {mae:.2f} lakh MAE")
