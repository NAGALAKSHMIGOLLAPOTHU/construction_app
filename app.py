import os
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load ML model (only for material quantity)
try:
    model = joblib.load("model.pkl")
except:
    model = None


# City-wise realistic middle-class rates
CITY_RATES = {
    "Hyderabad": 1700,
    "Chennai": 1800,
    "Bangalore": 2000
}


@app.route("/")
def home():
    # ✅ FIX: Pass cities to template
    return render_template("index.html", cities=CITY_RATES.keys())


@app.route("/predict", methods=["POST"])
def predict():
    city = request.form["city"]
    area = int(request.form["area"])        # area per floor
    floors = int(request.form["floors"])

    total_area = area * floors
    rate_per_sqft = CITY_RATES.get(city, 1700)

    # ---------- ML FOR MATERIAL ONLY ----------
    if model:
        cement, sand, bricks, steel = model.predict([[total_area]])[0]
        cement = int(cement)
        sand = int(sand)
        bricks = int(bricks)
        steel = int(steel)
    else:
        # fallback logic (SAFE)
        cement = int(total_area * 0.8)
        sand = int(total_area * 2.5)
        bricks = int(total_area * 8)
        steel = int(total_area * 3)

    # ---------- RATE-BASED COSTING ----------
    material_cost = total_area * rate_per_sqft * 0.55
    labour_cost = total_area * rate_per_sqft * 0.20
    interior_cost = total_area * rate_per_sqft * 0.15
    electrical_cost = total_area * rate_per_sqft * 0.07
    approval_cost = total_area * rate_per_sqft * 0.03

    total_cost = (
        material_cost
        + labour_cost
        + interior_cost
        + electrical_cost
        + approval_cost
    )

    return render_template(
        "result.html",
        city=city,
        area=area,
        floors=floors,
        total_area=total_area,
        rate_per_sqft=rate_per_sqft,
        cement=cement,
        sand=sand,
        bricks=bricks,
        steel=steel,
        material_cost=int(material_cost),
        labour_cost=int(labour_cost),
        interior_cost=int(interior_cost),
        electrical_cost=int(electrical_cost),
        approval_cost=int(approval_cost),
        total_cost=int(total_cost)
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
