import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

cars = pd.read_csv("cars.csv")
features = cars[["age_years", "mileage_km", "engine_cc"]]
prices = cars["price_lakh"]

features_train, features_test, prices_train, prices_test = train_test_split(
    features, prices, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(features_train, prices_train)

predictions = model.predict(features_test)
mae = mean_absolute_error(prices_test, predictions)

print(f"Training cars: {len(features_train)}")
print(f"Test cars: {len(features_test)}")
print(f"Mean absolute error: {mae:.2f} lakh")
if mae > 3.0:
    raise SystemExit(f"Quality check failed: {mae:.2f} lakh is above 3.00 lakh.")

print("Quality check passed.")
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
