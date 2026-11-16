import os
import math
import time
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, Response, request, jsonify
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge

app = Flask(__name__)

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_HEADERS = {"User-Agent": "PlanetPulse/1.0 (NASA Space Apps 2026 student project)"}
_place_cache = {}
_nominatim_lock = threading.Lock()
_last_nominatim_call = [0.0]


def _wrap_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _format_coords(lat, lon):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def describe_location(lat, lon):
    """Return the area name for a coordinate using the Nominatim API."""
    lon = _wrap_lon(lon)
    key = (round(lat, 3), round(lon, 3))
    if key in _place_cache:
        return _place_cache[key]

    coords = _format_coords(lat, lon)
    try:
        with _nominatim_lock:  
            wait = 1.0 - (time.time() - _last_nominatim_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_nominatim_call[0] = time.time()
            resp = requests.get(
                NOMINATIM_REVERSE_URL,
                params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10,
                        "addressdetails": 1, "accept-language": "en"},
                headers=NOMINATIM_HEADERS,
                timeout=8,
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # API unreachable: show coordinates only, and don't cache so it retries next time
        return {"label": coords, "name": "", "region": "", "country": "",
                "precision": "unavailable", "coords": coords, "error": str(e)[:80]}

    if "error" in data:  # Nominatim returns an error for oceans and empty areas
        result = {"label": "Open water or unnamed area", "name": "", "region": "", "country": "",
                  "precision": "remote", "coords": coords}
    else:
        addr = data.get("address", {})
        name = next((addr[k] for k in ("city", "town", "village", "hamlet", "municipality",
                                       "suburb", "county", "state_district") if addr.get(k)), "")
        region = addr.get("state") or addr.get("province") or addr.get("region") or ""
        country = addr.get("country", "")

        parts = []
        for p in (name, region, country):
            if p and p.lower() not in [x.lower() for x in parts]:
                parts.append(p)
        label = ", ".join(parts) or data.get("display_name", coords)
        result = {"label": label, "name": name, "region": region, "country": country,
                  "precision": "local", "coords": coords}

    _place_cache[key] = result
    return result


def place_for(lat, lon):
    return describe_location(float(lat), float(lon))

NASA_POWER_MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

VARIABLES_CATALOG = [
    {
        "id": "T2M",
        "name": "Surface Air Temperature (2m)",
        "unit": "°C",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "How hot or cold the air feels just above the ground.",
        "audience_science": "2-meter air temperature derived from NASA's Modern-Era Retrospective analysis for Research and Applications, Version 2.",
        "anomaly_example": "Sudden polar vortex dips or unprecedented urban heatwaves."
    },
    {
        "id": "PRECTOTCORR",
        "name": "Precipitation (Rain & Snow)",
        "unit": "mm/day",
        "agency": "NASA POWER (MERRA-2, bias-corrected)",
        "audience_simple": "Total water falling from the sky as rain, sleet, or snow.",
        "audience_science": "Bias-corrected total surface precipitation from the MERRA-2 reanalysis, as provided by NASA POWER.",
        "anomaly_example": "Extreme cloudbursts in arid deserts (e.g., Dubai) or desert snowfall events."
    },
    {
        "id": "SNODP",
        "name": "Snow Depth & Accumulation",
        "unit": "cm",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "How deep the snowpack is sitting on the ground.",
        "audience_science": "Land surface snow depth from the MERRA-2 reanalysis, as provided by NASA POWER.",
        "anomaly_example": "Rare snowfall in Sahara/Arabian deserts or rapid spring snowmelt in the Arctic."
    },
    {
        "id": "RH2M",
        "name": "Relative Humidity (2m)",
        "unit": "%",
        "agency": "NASA POWER (MERRA-2)",
        "audience_simple": "How full of moisture the air is compared to the maximum it can hold.",
        "audience_science": "Ratio of actual water vapor pressure to saturation vapor pressure at 2 meters above surface.",
        "anomaly_example": "Atmospheric rivers injecting tropical moisture into dry inland regions."
    },
    {
        "id": "ALLSKY_SFC_SW_DWN",
        "name": "Solar Irradiance (Sunlight Energy)",
        "unit": "kW-hr/m²/day",
        "agency": "NASA POWER (CERES / SRB)",
        "audience_simple": "How much solar energy hits the ground each day.",
        "audience_science": "All-sky surface downward shortwave irradiance from NASA's SRB (1984–2000) and CERES SYN1deg / FLASHFlux (2001 onward) products, as provided by NASA POWER.",
        "anomaly_example": "Dimming caused by wildfire smoke plumes or volcanic aerosols."
    },
    {
        "id": "GWETPROF",
        "name": "Root-Zone Soil Moisture",
        "unit": "0 to 1 (Wetness Index)",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "How wet or dry the soil is deep down where plant roots drink.",
        "audience_science": "Profile soil moisture wetness fraction (0 = completely dry, 1 = fully saturated) from the MERRA-2 land model, as provided by NASA POWER.",
        "anomaly_example": "Flash droughts in agricultural belts or severe flood saturation."
    },
    {
        "id": "WS10M",
        "name": "Surface Wind Speed (10m)",
        "unit": "m/s",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "How fast the wind is blowing near the surface.",
        "audience_science": "10-meter wind speed from the MERRA-2 atmospheric reanalysis, as provided by NASA POWER.",
        "anomaly_example": "Jet stream meanders stalling storm systems over a single region."
    },
    {
        "id": "T2M_MAX",
        "name": "Maximum Air Temperature (2m)",
        "unit": "°C",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "The hottest air temperature just above the ground in each month.",
        "audience_science": "2-meter maximum air temperature from MERRA-2 as provided in the NASA POWER monthly product.",
        "anomaly_example": "Record-breaking daytime heatwaves."
    },
    {
        "id": "T2M_MIN",
        "name": "Minimum Air Temperature (2m)",
        "unit": "°C",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "The coldest air temperature just above the ground in each month, usually at night.",
        "audience_science": "2-meter minimum air temperature from MERRA-2 as provided in the NASA POWER monthly product.",
        "anomaly_example": "Unusually warm nights during heatwaves or severe cold snaps."
    },
    {
        "id": "QV2M",
        "name": "Specific Humidity (2m)",
        "unit": "g/kg",
        "agency": "NASA POWER (MERRA-2 Reanalysis)",
        "audience_simple": "The actual amount of water vapour in the air: grams of water in each kilogram of air.",
        "audience_science": "Mass of water vapor per unit mass of moist air at 2 meters above the surface, from MERRA-2.",
        "anomaly_example": "Moisture surges from atmospheric rivers or an early monsoon onset."
    },
    {
        "id": "CLOUD_AMT",
        "name": "Cloud Amount",
        "unit": "%",
        "agency": "NASA POWER",
        "audience_simple": "How much of the sky is covered by cloud.",
        "audience_science": "Average percentage of cloud cover over the grid cell, as provided in the NASA POWER monthly product.",
        "anomaly_example": "Persistent cloud cover during monsoons or unusually clear, dry spells."
    }
]


# NASA POWER monthly availability: MERRA-2 meteorology from 1981, CERES/SRB solar & cloud from 1984,
# up to the last complete calendar year.
DEFAULT_FIRST_YEAR = 1981
VARIABLE_FIRST_YEAR = {"ALLSKY_SFC_SW_DWN": 1984, "CLOUD_AMT": 1984}
MIN_SPAN_YEARS = 3


def latest_complete_year():
    return datetime.utcnow().year - 1


def first_year_for(parameter):
    return VARIABLE_FIRST_YEAR.get(parameter, DEFAULT_FIRST_YEAR)


def clamp_years(parameter, start_year, end_year):
    """Keep the requested period inside the years NASA POWER provides for this variable."""
    lo, hi = first_year_for(parameter), latest_complete_year()
    start_year = max(lo, min(int(start_year), hi - MIN_SPAN_YEARS + 1))
    end_year = min(hi, max(int(end_year), start_year + MIN_SPAN_YEARS - 1))
    return start_year, end_year


for _v in VARIABLES_CATALOG:
    _v["first_year"] = first_year_for(_v["id"])


def mann_kendall_test(x):
    n = len(x)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            s += np.sign(x[j] - x[k])

    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    if p_value < 0.05:
        direction = "Increasing (Significant)" if z > 0 else "Decreasing (Significant)"
    else:
        direction = "No Statistically Significant Trend"

    return direction, round(float(p_value), 5), round(float(z), 3), round(float(tau), 3)


def fetch_nasa_power_data(lat, lon, parameter, start_year=2000, end_year=2025):
    """Monthly series from the NASA POWER API. Raises RuntimeError if NASA POWER can't provide it
    (no synthetic data is ever substituted)."""
    params = {
        "parameters": parameter,
        "community": "AG",
        "longitude": round(float(lon), 4),
        "latitude": round(float(lat), 4),
        "start": start_year,
        "end": end_year,
        "format": "JSON"
    }
    try:
        resp = requests.get(NASA_POWER_MONTHLY_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        raw_dict = payload["properties"]["parameter"][parameter]
    except Exception as e:
        raise RuntimeError(f"NASA POWER could not be reached or returned an error ({str(e)[:80]}). Please try again.")

    dates, values = [], []
    for ym, val in sorted(raw_dict.items()):
        if ym.endswith("13") or val is None or val <= -900:
            continue
        dates.append(f"{ym[:4]}-{ym[4:]}")
        values.append(float(val))
    if len(values) < 24:
        raise RuntimeError("NASA POWER has fewer than 24 valid months for this point and period. Try a longer period or another location.")
    return dates, values, "NASA POWER API (monthly point, community AG)"


def generate_anomaly_diagnostics_and_precautions(parameter, z_val, direction):
    is_spike = direction == "Spike Above Normal"
    
    if parameter == "PRECTOTCORR":
        if is_spike:
            cause = "Atmospheric River / Exceptional Cloudburst: Unusually high moisture transport injected into regional airmass, overwhelming natural soil infiltration capacity."
            precautions = "• Flash Flood Advisory: Move to higher ground immediately.\n• Clear urban storm drains and culverts.\n• Avoid driving through standing water or flooded underpasses."
        else:
            cause = "Prolonged Meteorological Drought: High pressure blocking atmospheric moisture convergence over the region."
            precautions = "• Implement strict agricultural water rationing.\n• Inspect irrigation systems for leaks.\n• Enforce wildfire burn bans due to dry vegetation."
            
    elif parameter == "T2M":
        if is_spike:
            cause = "Extreme Heat Dome / Subsidence Inversion: Trapped high-pressure airmass causing intense radiative heating and stagnant air."
            precautions = "• Stay indoors in air-conditioned spaces during peak daylight hours.\n• Hydrate frequently and check on elderly or vulnerable neighbors.\n• Postpone strenuous outdoor labor."
        else:
            cause = "Polar Vortex Disruption / Arctic Air Intrusion: Rapid displacement of high-latitude freezing air masses into lower latitudes."
            precautions = "• Insulate exposed water pipes to prevent freezing/bursting.\n• Wear layered thermal clothing and protect skin from frostbite.\n• Ensure safe auxiliary heating and check carbon monoxide detectors."
            
    elif parameter == "SNODP" or parameter == "SNOW":
        cause = "Anomalous Sub-Freezing Precipitation Front: Deep cold air wedge meeting winter moisture currents over unaccustomed terrain."
        precautions = "• Exercise extreme caution on icy roadways (black ice hazard).\n• Protect sensitive outgoing crops and livestock.\n• Stock up on emergency rations and backup power."
            
    else:
        cause = "Statistically Significant Outlier: Environmental variable surged beyond multi-year monthly baseline."
        precautions = "• Monitor local meteorological agency bulletins.\n• Secure loose outdoor equipment against potential wind/storm surges.\n• Review regional disaster preparedness plans."
            
    return cause, precautions


def run_ml_and_anomalies(dates, values, parameter, forecast_months=36):
    y = np.array(values, dtype=float)
    n = len(y)
    t = np.arange(n)

    def make_features(t_arr):
        month_idx = t_arr % 12
        return np.column_stack([
            t_arr,
            np.sin(2 * np.pi * month_idx / 12.0),
            np.cos(2 * np.pi * month_idx / 12.0),
            np.sin(4 * np.pi * month_idx / 12.0),
            np.cos(4 * np.pi * month_idx / 12.0)
        ])

    X = make_features(t)

    monthly_means = np.array([np.mean(y[i::12]) for i in range(12)])
    monthly_stds = np.array([np.std(y[i::12]) + 1e-6 for i in range(12)])
    deseasonalized = np.array([y[i] - monthly_means[i % 12] for i in range(n)])
    z_scores = np.array([deseasonalized[i] / monthly_stds[i % 12] for i in range(n)])

    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    iso_preds = iso_forest.fit_predict(deseasonalized.reshape(-1, 1))

    anomalies = []
    for i in range(n):
        if iso_preds[i] == -1 and abs(z_scores[i]) >= 2.2:
            severity = "Extreme" if abs(z_scores[i]) >= 3.2 else "Unusual"
            direction = "Spike Above Normal" if z_scores[i] > 0 else "Drop Below Normal"
            
            explanation, precautions = generate_anomaly_diagnostics_and_precautions(parameter, z_scores[i], direction)
            
            anomalies.append({
                "date": dates[i],
                "value": round(float(y[i]), 2),
                "expected": round(float(monthly_means[i % 12]), 2),
                "z_score": round(float(z_scores[i]), 2),
                "severity": severity,
                "type": direction,
                "explanation": explanation,
                "precautions": precautions
            })

    anomalies = sorted(anomalies, key=lambda a: abs(a["z_score"]), reverse=True)[:6]

    t_future = np.arange(n, n + forecast_months)
    X_future = make_features(t_future)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    pred_ridge = ridge.predict(X_future)

    slope, intercept, _, _, _ = stats.linregress(t, y)
    y_detrended = y - (slope * t + intercept)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X[:, 1:], y_detrended)
    pred_rf = rf.predict(X_future[:, 1:]) + (slope * t_future + intercept)

    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    gbr.fit(X[:, 1:], y_detrended)
    pred_gbr = gbr.predict(X_future[:, 1:]) + (slope * t_future + intercept)

    ensemble = (pred_ridge + pred_rf + pred_gbr) / 3.0
    residual_std = np.std(y - ridge.predict(X))
    upper_ci = ensemble + 1.96 * residual_std
    lower_ci = ensemble - 1.96 * residual_std

    last_year, last_month = map(int, dates[-1].split("-"))
    future_dates = []
    curr_y, curr_m = last_year, last_month
    for _ in range(forecast_months):
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1
        future_dates.append(f"{curr_y}-{curr_m:02d}")

    return {
        "anomalies": anomalies,
        "future_dates": future_dates,
        "predictions": {
            "ensemble": np.round(ensemble, 2).tolist(),
            "random_forest": np.round(pred_rf, 2).tolist(),
            "gradient_boosting": np.round(pred_gbr, 2).tolist(),
            "seasonal_ridge": np.round(pred_ridge, 2).tolist(),
            "upper_95": np.round(upper_ci, 2).tolist(),
            "lower_95": np.round(lower_ci, 2).tolist()
        },
        "linear_trendline": np.round(slope * t + intercept, 2).tolist()
    }


# ---------------------------------------------------------------
# Discover: automatic trend screening + location comparison
# ---------------------------------------------------------------
POWER_PARAMETERS = [v["id"] for v in VARIABLES_CATALOG]
CATALOG_BY_ID = {v["id"]: v for v in VARIABLES_CATALOG}


def fetch_power_multi(lat, lon, parameters, start_year, end_year):
    """One NASA POWER request for several parameters.
    Returns {parameter: (dates, values)}; parameters without enough data are left out."""
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "longitude": round(float(lon), 4),
        "latitude": round(float(lat), 4),
        "start": start_year,
        "end": end_year,
        "format": "JSON"
    }
    resp = requests.get(NASA_POWER_MONTHLY_URL, params=params, timeout=30)
    resp.raise_for_status()
    block = resp.json()["properties"]["parameter"]

    out = {}
    for p in parameters:
        raw = block.get(p)
        if not raw:
            continue
        dates, values = [], []
        for ym, val in sorted(raw.items()):
            if ym.endswith("13") or val is None or val <= -900:
                continue
            dates.append(f"{ym[:4]}-{ym[4:]}")
            values.append(float(val))
        if len(values) >= 24:
            out[p] = (dates, values)
    return out


def compute_trend_stats(values):
    """Linear trend, Mann-Kendall test and a standardized effect size for one monthly series."""
    y = np.array(values, dtype=float)
    n = len(y)
    t = np.arange(n)

    lr = stats.linregress(t, y)
    slope_per_decade = lr.slope * 120.0
    mk_direction, mk_p, mk_z, mk_tau = mann_kendall_test(y)

    # Effect size: trend per decade divided by the spread of deseasonalized (anomaly) values
    monthly_means = np.array([np.mean(y[i::12]) for i in range(12)])
    resid = y - monthly_means[t % 12]
    resid_sd = float(np.std(resid))
    effect = slope_per_decade / resid_sd if resid_sd > 1e-9 else 0.0
    mk_hr_p = mann_kendall_hamed_rao(t, resid)["p"] if n >= 10 else None

    return {
        "mean": round(float(np.mean(y)), 3),
        "min": round(float(np.min(y)), 3),
        "max": round(float(np.max(y)), 3),
        "slope_per_decade": round(float(slope_per_decade), 4),
        "ci_95_decade": round(float(1.96 * lr.stderr * 120.0), 4),
        "r_squared": round(float(lr.rvalue ** 2), 4),
        "ttest_p_value": round(float(lr.pvalue), 5),
        "mann_kendall_p_value": mk_p,
        "mann_kendall_z": mk_z,
        "mann_kendall_tau": mk_tau,
        "mann_kendall_trend": mk_direction,
        "effect_size_sd_per_decade": round(float(effect), 3),
        "mk_hamed_rao_p_deseasonalized": round(float(mk_hr_p), 5) if mk_hr_p is not None else None,
        "is_significant": bool(lr.pvalue < 0.05 or mk_p < 0.05)
    }


def benjamini_hochberg(pvals):
    """False-discovery-rate adjusted p-values (Benjamini-Hochberg)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running_min = min(running_min, p[i] * m / (rank + 1))
        adjusted[i] = running_min
    return adjusted.tolist()


def annual_means(dates, values):
    """Average of each calendar year that has all 12 months."""
    df = pd.DataFrame({"year": [int(d[:4]) for d in dates], "value": values})
    g = df.groupby("year")["value"].agg(["mean", "count"])
    g = g[g["count"] == 12]
    return [int(y) for y in g.index], [round(float(v), 3) for v in g["mean"]]


def _read_point_request(data):
    lat = float(data.get("lat", 25.2048))
    lon = float(data.get("lon", 55.2708))
    if not -90 <= lat <= 90:
        raise ValueError("lat must be between -90 and 90")
    lon = ((lon + 180.0) % 360.0) - 180.0
    start_year, end_year = clamp_years(data.get("parameter"),
                                       int(data.get("start_year", 2001)),
                                       int(data.get("end_year", latest_complete_year())))
    return lat, lon, start_year, end_year


@app.route("/api/discover", methods=["POST"])
def discover_trends():
    data = request.get_json() or {}
    try:
        lat, lon, start_year, end_year = _read_point_request(data)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    power_error = None
    try:
        series = fetch_power_multi(lat, lon, POWER_PARAMETERS, start_year, end_year)
    except Exception as e:
        series = {}
        power_error = f"NASA POWER could not be reached ({str(e)[:60]})"

    results = []
    for v in VARIABLES_CATALOG:
        row = {"id": v["id"], "name": v["name"], "unit": v["unit"], "agency": v["agency"]}
        if v["id"] in series:
            dates, values = series[v["id"]]
            source = "NASA POWER API"
            row["modelled"] = False
        else:
            row["status"] = "unavailable"
            row["reason"] = power_error or "Not enough NASA POWER data for this point and period"
            results.append(row)
            continue

        row.update(compute_trend_stats(values))
        row.update({"status": "ok", "source": source, "n_months": len(values),
                    "first": dates[0], "last": dates[-1]})
        results.append(row)

    ok_rows = [r for r in results if r["status"] == "ok"]
    adjusted = benjamini_hochberg([r["mann_kendall_p_value"] for r in ok_rows])
    for r, adj in zip(ok_rows, adjusted):
        r["adjusted_p_value"] = round(float(adj), 5)
        r["significant_after_adjustment"] = bool(adj < 0.05)

    results.sort(key=lambda r: (
        r["status"] != "ok",
        not r.get("significant_after_adjustment", False),
        -abs(r.get("effect_size_sd_per_decade", 0.0))
    ))

    return jsonify({
        "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "place": place_for(lat, lon),
        "start_year": start_year,
        "end_year": end_year,
        "power_error": power_error,
        "tested": len(ok_rows),
        "results": results
    })


@app.route("/api/compare", methods=["POST"])
def compare_locations():
    data = request.get_json() or {}
    parameter = data.get("parameter", "T2M")
    if parameter not in CATALOG_BY_ID:
        return jsonify({"error": "Unknown parameter"}), 400
    locations = data.get("locations") or []
    if len(locations) != 2:
        return jsonify({"error": "Send exactly two locations"}), 400

    out = []
    for loc in locations:
        try:
            lat, lon, start_year, end_year = _read_point_request({**data, **loc})
        except (TypeError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        try:
            dates, values, source = fetch_nasa_power_data(lat, lon, parameter, start_year, end_year)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 502
        years, yearly = annual_means(dates, values)
        out.append({
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "place": place_for(lat, lon),
            "source": source,
            "fallback": False,
            "modelled": False,
            "first": dates[0],
            "last": dates[-1],
            "stats": compute_trend_stats(values),
            "annual": {"years": years, "values": yearly}
        })

    v = CATALOG_BY_ID[parameter]
    return jsonify({"parameter": parameter, "name": v["name"], "unit": v["unit"], "locations": out})


# ===============================================================
# SCIENTIST MODE: rigorous trend statistics
# References:
#   Mann (1945); Kendall (1975) - Mann-Kendall trend test
#   Sen (1968) - Sen's slope estimator; Gilbert (1987) - its confidence interval
#   Hirsch, Slack & Smith (1982) - Seasonal Mann-Kendall test
#   Hamed & Rao (1998) - Mann-Kendall variance correction for autocorrelation
#   Santer et al. (2000) - effective sample size for regression trends
#   Weatherhead et al. (1998) - years of data needed to detect a trend
#   Durbin & Watson (1950); Shapiro & Wilk (1965) - residual diagnostics
# ===============================================================
from concurrent.futures import ThreadPoolExecutor

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _r(x, d=5):
    """Round to a JSON-safe float (None for NaN/inf)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, d) if math.isfinite(x) else None


def _lag1(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _mk_s_and_var(x):
    """Mann-Kendall S statistic and its variance (with tie correction)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    diff = x[None, :] - x[:, None]
    s = float(np.sign(diff[np.triu_indices(n, 1)]).sum())
    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    return s, float(var_s)


def _z_from_s(s, var_s):
    if var_s <= 0:
        return 0.0
    if s > 0:
        return (s - 1) / math.sqrt(var_s)
    if s < 0:
        return (s + 1) / math.sqrt(var_s)
    return 0.0


def _p_from_z(z):
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def sens_slope(t, x, alpha=0.05):
    """Sen's slope (per unit of t) with Gilbert (1987) confidence interval."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    i, j = np.triu_indices(n, 1)
    dt = t[j] - t[i]
    keep = dt != 0
    slopes = np.sort((x[j] - x[i])[keep] / dt[keep])
    m = len(slopes)
    slope = float(np.median(slopes))
    intercept = float(np.median(x - slope * t))
    _, var_s = _mk_s_and_var(x)
    c = stats.norm.ppf(1 - alpha / 2) * math.sqrt(var_s)
    lo_idx = int(np.clip(round((m - c) / 2) - 1, 0, m - 1))
    hi_idx = int(np.clip(round((m + c) / 2), 0, m - 1))
    return slope, intercept, float(slopes[lo_idx]), float(slopes[hi_idx])


def mann_kendall_full(x):
    s, var_s = _mk_s_and_var(x)
    z = _z_from_s(s, var_s)
    n = len(x)
    return {"S": s, "var_S": var_s, "Z": z, "p": _p_from_z(z), "tau": s / (0.5 * n * (n - 1))}


def mann_kendall_hamed_rao(t, x, alpha=0.05):
    """Modified Mann-Kendall test (Hamed & Rao 1998): inflates Var(S) for significant
    autocorrelation in the ranks of the Sen-detrended series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s, var_s = _mk_s_and_var(x)
    slope, _, _, _ = sens_slope(t, x, alpha)
    ranks = stats.rankdata(x - slope * np.asarray(t, dtype=float))
    r = ranks - ranks.mean()
    denom = np.sum(r * r)
    acf = np.array([np.sum(r[:n - k] * r[k:]) / denom for k in range(n)])
    bound = stats.norm.ppf(1 - alpha / 2) / math.sqrt(n)
    total = 0.0
    for k in range(1, n - 1):
        if abs(acf[k]) > bound:
            total += (n - k) * (n - k - 1) * (n - k - 2) * acf[k]
    correction = 1 + 2.0 / (n * (n - 1) * (n - 2)) * total
    if correction <= 0:
        correction = 1.0
    var_mod = var_s * correction
    z = _z_from_s(s, var_mod)
    return {"S": s, "var_S": var_mod, "Z": z, "p": _p_from_z(z), "variance_correction": correction}


def seasonal_mann_kendall(years, months, x):
    """Seasonal Mann-Kendall (Hirsch et al. 1982) + seasonal Sen slope (per year)."""
    years, months, x = np.asarray(years), np.asarray(months), np.asarray(x, dtype=float)
    s_total, var_total, all_slopes = 0.0, 0.0, []
    for m in range(1, 13):
        sel = months == m
        if sel.sum() < 3:
            continue
        xm, ym = x[sel], years[sel].astype(float)
        s, v = _mk_s_and_var(xm)
        s_total += s
        var_total += v
        i, j = np.triu_indices(len(xm), 1)
        dy = ym[j] - ym[i]
        keep = dy != 0
        all_slopes.extend(((xm[j] - xm[i])[keep] / dy[keep]).tolist())
    z = _z_from_s(s_total, var_total)
    return {"S": s_total, "Z": z, "p": _p_from_z(z),
            "sen_slope_per_year": float(np.median(all_slopes)) if all_slopes else 0.0}


def ols_trend(t, x, alpha=0.05):
    """Ordinary least squares trend plus the Santer et al. (2000) effective-sample-size version."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    lr = stats.linregress(t, x)
    resid = x - (lr.slope * t + lr.intercept)
    r1 = _lag1(resid)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 2)
    out = {"slope": lr.slope, "intercept": lr.intercept, "se": lr.stderr, "p": lr.pvalue,
           "t": lr.slope / lr.stderr if lr.stderr > 0 else 0.0, "r2": lr.rvalue ** 2,
           "ci_low": lr.slope - tcrit * lr.stderr, "ci_high": lr.slope + tcrit * lr.stderr,
           "lag1": r1, "residuals": resid}
    n_eff = n * (1 - r1) / (1 + r1) if r1 > 0 else float(n)
    n_eff = max(n_eff, 3.0)
    se_adj = lr.stderr * math.sqrt((n - 2) / (n_eff - 2)) if n_eff > 2 else float("inf")
    t_adj = lr.slope / se_adj if se_adj > 0 else 0.0
    df = max(n_eff - 2, 1.0)
    tcrit_adj = stats.t.ppf(1 - alpha / 2, df)
    out.update({"n_eff": n_eff, "se_adj": se_adj, "t_adj": t_adj,
                "p_adj": float(2 * (1 - stats.t.cdf(abs(t_adj), df))),
                "ci_low_adj": lr.slope - tcrit_adj * se_adj, "ci_high_adj": lr.slope + tcrit_adj * se_adj})
    return out


def weatherhead_years(slope_per_year, noise_sd, phi):
    """Years of monthly data needed to detect a trend with 90% probability at the 5% level
    (Weatherhead et al. 1998): n* = [(3.3 sigma_N / |omega|) * sqrt((1+phi)/(1-phi))]^(2/3)."""
    if abs(slope_per_year) < 1e-12 or noise_sd <= 0:
        return None
    phi = float(np.clip(phi, -0.95, 0.95))
    return float(((3.3 * noise_sd / abs(slope_per_year)) * math.sqrt((1 + phi) / (1 - phi))) ** (2.0 / 3.0))


def _yearly_series(years, months, values, season_months):
    """Mean of the given months for each year (DJF: December counts toward the next year)."""
    acc = {}
    for y, m, v in zip(years, months, values):
        if m not in season_months:
            continue
        key = int(y) + 1 if (season_months == SEASONS["DJF"] and m == 12) else int(y)
        acc.setdefault(key, []).append(v)
    yrs = sorted(k for k, vals in acc.items() if len(vals) == len(season_months))
    return np.array(yrs, dtype=float), np.array([np.mean(acc[k]) for k in yrs])


def _quick_trend(t, x, alpha):
    if len(x) < 5:
        return None
    o = ols_trend(t, x, alpha)
    mk = mann_kendall_full(x)
    sen, _, _, _ = sens_slope(t, x, alpha)
    return {"n": int(len(x)), "ols_per_decade": _r(o["slope"] * 10, 4), "ols_p": _r(o["p"]),
            "sen_per_decade": _r(sen * 10, 4), "mk_p": _r(mk["p"]), "mk_tau": _r(mk["tau"], 3),
            "significant": bool(min(o["p"], mk["p"]) < alpha)}


def science_analysis(dates, values, alpha=0.05, aggregation="monthly"):
    years = np.array([int(d[:4]) for d in dates])
    months = np.array([int(d[5:7]) for d in dates])
    y = np.array(values, dtype=float)
    climatology = {m: float(np.mean(y[months == m])) for m in range(1, 13) if np.any(months == m)}
    anomalies = y - np.array([climatology[m] for m in months])
    t_month = years + (months - 0.5) / 12.0

    # ---- choose the series to test ----
    if aggregation == "monthly":
        t, x = t_month, anomalies
        labels = list(dates)
        series_desc = "Deseasonalized monthly anomalies (each month minus its long-term mean for that calendar month)"
    elif aggregation == "annual":
        t, x = _yearly_series(years, months, y, tuple(range(1, 13)))
        labels = [str(int(v)) for v in t]
        series_desc = "Annual means (calendar years with all 12 months)"
    elif aggregation in SEASONS:
        t, x = _yearly_series(years, months, y, SEASONS[aggregation])
        labels = [str(int(v)) for v in t]
        series_desc = f"{aggregation} seasonal means" + (" (December counted with the following January-February)" if aggregation == "DJF" else "")
    else:
        raise ValueError("Unknown aggregation")
    if len(x) < 8:
        raise ValueError("Not enough data points for this aggregation; choose a longer period.")

    ols = ols_trend(t, x, alpha)
    raw_ols = ols_trend(t_month, y, alpha)
    sen, sen_icpt, sen_lo, sen_hi = sens_slope(t, x, alpha)
    mk = mann_kendall_full(x)
    mk_hr = mann_kendall_hamed_rao(t, x, alpha)
    smk = seasonal_mann_kendall(years, months, y) if aggregation == "monthly" else None

    def dec(v):
        return _r(v * 10, 5)

    estimates = [
        {"method": "OLS on raw monthly values", "note": "Seasonal cycle left in; shown for comparison",
         "per_decade": dec(raw_ols["slope"]), "ci_low": dec(raw_ols["ci_low"]), "ci_high": dec(raw_ols["ci_high"]), "p": _r(raw_ols["p"])},
        {"method": "OLS (least squares)", "note": series_desc,
         "per_decade": dec(ols["slope"]), "ci_low": dec(ols["ci_low"]), "ci_high": dec(ols["ci_high"]), "p": _r(ols["p"])},
        {"method": "OLS, autocorrelation-adjusted", "note": f"Effective sample size n_eff = {ols['n_eff']:.0f} of {len(x)} (Santer et al. 2000)",
         "per_decade": dec(ols["slope"]), "ci_low": dec(ols["ci_low_adj"]), "ci_high": dec(ols["ci_high_adj"]), "p": _r(ols["p_adj"])},
        {"method": "Sen's slope", "note": "Median of all pairwise slopes; CI from Gilbert (1987)",
         "per_decade": dec(sen), "ci_low": dec(sen_lo), "ci_high": dec(sen_hi), "p": _r(mk["p"])},
    ]
    if smk:
        estimates.append({"method": "Seasonal Sen's slope", "note": "Median of within-calendar-month slopes",
                          "per_decade": dec(smk["sen_slope_per_year"]), "ci_low": None, "ci_high": None, "p": _r(smk["p"])})

    tests = [
        {"test": "t-test on OLS slope", "statistic": "t", "value": _r(ols["t"], 3), "p": _r(ols["p"])},
        {"test": "t-test, effective sample size", "statistic": "t", "value": _r(ols["t_adj"], 3), "p": _r(ols["p_adj"])},
        {"test": "Mann-Kendall", "statistic": "Z", "value": _r(mk["Z"], 3), "p": _r(mk["p"]), "tau": _r(mk["tau"], 4)},
        {"test": "Mann-Kendall, Hamed-Rao corrected", "statistic": "Z", "value": _r(mk_hr["Z"], 3), "p": _r(mk_hr["p"]),
         "note": f"Var(S) inflated ×{mk_hr['variance_correction']:.2f} for autocorrelation"},
    ]
    if smk:
        tests.append({"test": "Seasonal Mann-Kendall", "statistic": "Z", "value": _r(smk["Z"], 3), "p": _r(smk["p"]),
                      "note": "Hirsch et al. (1982); compares each calendar month only with itself"})
    for row in tests:
        row["significant"] = bool(row["p"] is not None and row["p"] < alpha)

    # ---- detectability (always from monthly anomalies) ----
    det_fit = ols_trend(t_month, anomalies, alpha)
    noise = det_fit["residuals"]
    sigma_n, phi = float(np.std(noise, ddof=1)), _lag1(noise)
    years_have = len(y) / 12.0
    years_needed = weatherhead_years(det_fit["slope"], sigma_n, phi)
    detect = {"years_have": _r(years_have, 1), "years_needed": _r(years_needed, 1) if years_needed else None,
              "noise_sd": _r(sigma_n, 4), "phi": _r(phi, 3), "slope_per_decade": dec(det_fit["slope"]),
              "detectable_now": bool(years_needed is not None and years_have >= years_needed)}

    # ---- residual diagnostics ----
    resid = ols["residuals"]
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)) if np.sum(resid ** 2) > 0 else None
    shapiro_p = float(stats.shapiro(resid).pvalue) if 3 <= len(resid) <= 5000 else None
    diagnostics = {"n": int(len(x)), "n_eff": _r(ols["n_eff"], 1), "lag1_autocorrelation": _r(ols["lag1"], 3),
                   "durbin_watson": _r(dw, 3), "shapiro_p": _r(shapiro_p), "r_squared": _r(ols["r2"], 4)}

    # ---- seasonal & monthly breakdown (from monthly values) ----
    by_season = []
    for name, mset in SEASONS.items():
        ts, xs = _yearly_series(years, months, y, mset)
        q = _quick_trend(ts, xs, alpha)
        if q:
            by_season.append({"period": name, **q})
    by_month = []
    for m in range(1, 13):
        ts, xs = _yearly_series(years, months, y, (m,))
        q = _quick_trend(ts, xs, alpha)
        if q:
            by_month.append({"period": MONTH_NAMES[m - 1], **q})

    return {
        "alpha": alpha,
        "aggregation": aggregation,
        "series_description": series_desc,
        "series": {"labels": labels, "values": [_r(v, 4) for v in x],
                   "ols_line": [_r(ols["slope"] * ti + ols["intercept"], 4) for ti in t],
                   "sen_line": [_r(sen * ti + sen_icpt, 4) for ti in t]},
        "raw": {"dates": list(dates), "values": [_r(v, 4) for v in y], "anomalies": [_r(v, 4) for v in anomalies]},
        "climatology": [_r(climatology.get(m), 4) for m in range(1, 13)],
        "estimates": estimates,
        "tests": tests,
        "detectability": detect,
        "diagnostics": diagnostics,
        "by_season": by_season,
        "by_month": by_month,
    }


def _power_request_url(lat, lon, parameter, start_year, end_year):
    params = {"parameters": parameter, "community": "AG", "longitude": round(float(lon), 4),
              "latitude": round(float(lat), 4), "start": start_year, "end": end_year, "format": "JSON"}
    return requests.Request("GET", NASA_POWER_MONTHLY_URL, params=params).prepare().url, params


def fetch_power_with_meta(lat, lon, parameter, start_year, end_year):
    """NASA POWER point request that also returns units/long name from the response. Raises on failure."""
    url, params = _power_request_url(lat, lon, parameter, start_year, end_year)
    resp = requests.get(NASA_POWER_MONTHLY_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    raw = payload["properties"]["parameter"][parameter]
    dates, values = [], []
    for ym, val in sorted(raw.items()):
        if ym.endswith("13") or val is None or val <= -900:
            continue
        dates.append(f"{ym[:4]}-{ym[4:]}")
        values.append(float(val))
    if len(values) < 24:
        raise ValueError("Insufficient data points")
    meta = (payload.get("parameters") or {}).get(parameter, {})
    return dates, values, {"url": url, "units": meta.get("units"), "longname": meta.get("longname")}


def _science_inputs(data):
    lat, lon, start_year, end_year = _read_point_request(data)
    parameter = data.get("parameter", "T2M")
    if parameter not in CATALOG_BY_ID:
        raise ValueError("Unknown parameter")
    alpha = float(data.get("alpha", 0.05))
    if alpha not in (0.01, 0.05, 0.10, 0.1):
        raise ValueError("alpha must be 0.01, 0.05 or 0.10")
    aggregation = data.get("aggregation", "monthly")
    if aggregation not in ("monthly", "annual", *SEASONS.keys()):
        raise ValueError("Unknown aggregation")
    return lat, lon, start_year, end_year, parameter, alpha, aggregation


@app.route("/api/science", methods=["POST"])
def science_point():
    data = request.get_json() or {}
    try:
        lat, lon, start_year, end_year, parameter, alpha, aggregation = _science_inputs(data)
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    v = CATALOG_BY_ID[parameter]
    provenance = {"variable": v["name"], "parameter": parameter, "catalog_unit": v["unit"], "agency": v["agency"],
                  "grid": "Nearest NASA POWER grid cell (MERRA-2 meteorology ≈ 0.5° × 0.625°; solar ≈ 1° × 1°)",
                  "retrieved_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                  "fallback": False, "modelled": False}
    try:
        dates, values, meta = fetch_power_with_meta(lat, lon, parameter, start_year, end_year)
    except Exception as e:
        return jsonify({"error": f"NASA POWER could not provide this series ({str(e)[:80]}). Please try again."}), 502
    provenance.update({"source": "NASA POWER API, monthly point, community AG", "url": meta["url"],
                       "api_units": meta["units"], "api_longname": meta["longname"]})
    try:
        result = science_analysis(dates, values, alpha, aggregation)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    result.update({"mode": "point", "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
                   "place": place_for(lat, lon), "provenance": provenance,
                   "period": {"start": dates[0], "end": dates[-1]}})
    return jsonify(result)


@app.route("/api/science/region", methods=["POST"])
def science_region():
    """Trend map + area-weighted regional average from a grid of NASA POWER point requests."""
    data = request.get_json() or {}
    try:
        _, _, start_year, end_year, parameter, alpha, aggregation = _science_inputs(data)
        box = data.get("box") or {}
        south, north = sorted([float(box["south"]), float(box["north"])])
        west, east = sorted([float(box["west"]), float(box["east"])])
        grid = int(data.get("grid", 4))
    except (TypeError, ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid request: {e}"}), 400
    if not (-90 <= south < north <= 90) or not (-180 <= west < east <= 180):
        return jsonify({"error": "Invalid box"}), 400
    if north - south > 10 or east - west > 10:
        return jsonify({"error": "Region is too large: keep it within 10° × 10°."}), 400
    if north - south < 0.5 or east - west < 0.5:
        return jsonify({"error": "Region is too small: make it at least 0.5° on each side."}), 400
    if grid not in (3, 4, 5):
        return jsonify({"error": "grid must be 3, 4 or 5"}), 400

    dlat, dlon = (north - south) / grid, (east - west) / grid
    cells = [{"row": r, "col": c,
              "lat": south + (r + 0.5) * dlat, "lon": west + (c + 0.5) * dlon,
              "south": south + r * dlat, "north": south + (r + 1) * dlat,
              "west": west + c * dlon, "east": west + (c + 1) * dlon}
             for r in range(grid) for c in range(grid)]

    def load(cell):
        try:
            d, vals, _ = fetch_power_with_meta(cell["lat"], cell["lon"], parameter, start_year, end_year)
            return cell, d, vals, None
        except Exception as e:
            return cell, None, None, str(e)[:60]

    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load, cells))

    out_cells, series_by_cell, errors = [], [], []
    for cell, d, vals, err in loaded:
        row = {k: _r(cell[k], 4) for k in ("lat", "lon", "south", "north", "west", "east")}
        if err:
            row["status"] = "unavailable"
            errors.append(err)
            out_cells.append(row)
            continue
        years = np.array([int(x[:4]) for x in d])
        months = np.array([int(x[5:7]) for x in d])
        y = np.array(vals, dtype=float)
        clim = {m: np.mean(y[months == m]) for m in range(1, 13) if np.any(months == m)}
        anom = y - np.array([clim[m] for m in months])
        t = years + (months - 0.5) / 12.0
        o = ols_trend(t, anom, alpha)
        hr = mann_kendall_hamed_rao(t, anom, alpha)
        row.update({"status": "ok", "per_decade": _r(o["slope"] * 10, 4), "p_ols": _r(o["p"]),
                    "p_mk_hr": _r(hr["p"]), "significant": bool(hr["p"] < alpha)})
        out_cells.append(row)
        series_by_cell.append((cell["lat"], dict(zip(d, vals))))

    if not series_by_cell:
        return jsonify({"error": "No grid cells could be loaded from NASA POWER. " + (errors[0] if errors else "")}), 502

    common = sorted(set.intersection(*[set(s.keys()) for _, s in series_by_cell]))
    weights = np.array([math.cos(math.radians(lat)) for lat, _ in series_by_cell])
    matrix = np.array([[s[dd] for dd in common] for _, s in series_by_cell])
    regional = (weights[:, None] * matrix).sum(axis=0) / weights.sum()
    try:
        result = science_analysis(common, regional.tolist(), alpha, aggregation)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    v = CATALOG_BY_ID[parameter]
    sample_url = _power_request_url(cells[0]["lat"], cells[0]["lon"], parameter, start_year, end_year)[0]
    result.update({
        "mode": "region",
        "box": {"south": south, "north": north, "west": west, "east": east},
        "grid": grid,
        "cells": out_cells,
        "cells_loaded": len(series_by_cell),
        "place": place_for((south + north) / 2, (west + east) / 2),
        "period": {"start": common[0], "end": common[-1]},
        "provenance": {"variable": v["name"], "parameter": parameter, "catalog_unit": v["unit"], "agency": v["agency"],
                       "source": "NASA POWER API, monthly point requests, community AG",
                       "grid": f"{grid}×{grid} sample points at cell centres; regional series = cos(latitude)-weighted mean of {len(series_by_cell)} loaded points",
                       "url": sample_url, "fallback": False, "modelled": False,
                       "retrieved_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                       "note": (f"{len(cells) - len(series_by_cell)} of {len(cells)} points could not be loaded." if errors else None)}
    })
    return jsonify(result)


# ---------------------------------------------------------------
# Linked indicators computed from real NASA POWER monthly data
# (replaces the earlier random-number modules)
# ---------------------------------------------------------------
INDICATOR_PARAMS = ["T2M", "T2M_MAX", "PRECTOTCORR", "RH2M", "WS10M", "GWETPROF"]


def compute_power_indicators(lat, lon, start_year, end_year):
    """Heat & drought stress, growing conditions and fire-weather conditions from one
    NASA POWER monthly request. Compares the first and last part of the selected period."""
    series = fetch_power_multi(lat, lon, INDICATOR_PARAMS, start_year, end_year)
    missing = [p for p in INDICATOR_PARAMS if p not in series]
    if missing:
        raise RuntimeError(f"NASA POWER returned no data for {', '.join(missing)}")

    df = pd.DataFrame({p: pd.Series(v, index=pd.PeriodIndex(d, freq="M")) for p, (d, v) in series.items()}).dropna()
    df["year"], df["month"], df["days"] = df.index.year, df.index.month, df.index.days_in_month
    counts = df.groupby("year").size()
    df = df[df["year"].isin(counts[counts == 12].index)]
    years = sorted(df["year"].unique())
    if len(years) < 3:
        raise RuntimeError("Not enough complete years for the indicators")

    seg = max(1, min(10, len(years) // 3))
    early_years, late_years = years[:seg], years[-seg:]
    early, late = df[df["year"].isin(early_years)], df[df["year"].isin(late_years)]
    periods = {"early": f"{early_years[0]}–{early_years[-1]}", "late": f"{late_years[0]}–{late_years[-1]}"}

    def r(x, d=2):
        return round(float(x), d)

    # ---- 1) Heat & drought stress ----
    hottest = df.groupby("year")["T2M_MAX"].max()
    lr = stats.linregress(np.array(hottest.index, dtype=float), hottest.values)
    dry_cut = df.groupby("month")["PRECTOTCORR"].quantile(0.2)
    df["dry"] = df["PRECTOTCORR"] <= df["month"].map(dry_cut)
    dry_early = 100 * df.loc[early.index, "dry"].mean()
    dry_late = 100 * df.loc[late.index, "dry"].mean()
    soil_early, soil_late = early["GWETPROF"].mean(), late["GWETPROF"].mean()
    signals = int(lr.slope > 0 and lr.pvalue < 0.05) + int(dry_late > dry_early) + int(soil_late < soil_early)
    heat = {
        "hottest_month_trend_per_decade": r(lr.slope * 10, 3),
        "hottest_month_trend_p": r(lr.pvalue, 4),
        "dry_month_share_early": r(dry_early, 1),
        "dry_month_share_late": r(dry_late, 1),
        "soil_wetness_early": r(soil_early, 3),
        "soil_wetness_late": r(soil_late, 3),
        "headline": ["No clear increase in heat or drought stress", "Mixed signals for heat and drought stress",
                     "Heat and drought stress are increasing", "Heat and drought stress are clearly increasing"][signals]
    }

    # ---- 2) Growing conditions ----
    df["warm"] = df["T2M"] >= 5.0
    df["gdd"] = (df["T2M"] - 10.0).clip(lower=0) * df["days"]
    df["rain_mm"] = df["PRECTOTCORR"] * df["days"]
    yearly = df.groupby("year").agg(season=("warm", "sum"), gdd=("gdd", "sum"), rain=("rain_mm", "sum"))
    yearly["season_rain"] = df[df["warm"]].groupby("year")["rain_mm"].sum().reindex(yearly.index, fill_value=0.0)
    ye, yl = yearly.loc[early_years], yearly.loc[late_years]
    annual_rain = yl["rain"].mean()
    if yl["season"].mean() < 1:
        outlook = "Too cold for field crops: no month averages 5 °C or more"
    elif annual_rain < 250:
        outlook = "Arid: crops would depend on irrigation"
    elif annual_rain < 500:
        outlook = "Semi-arid: drought-tolerant crops or irrigation"
    else:
        outlook = "Rain-fed farming is plausible"
    rain_change = 100 * (yl["season_rain"].mean() - ye["season_rain"].mean()) / ye["season_rain"].mean() if ye["season_rain"].mean() > 0 else None
    grow = {
        "season_months_early": r(ye["season"].mean(), 1),
        "season_months_late": r(yl["season"].mean(), 1),
        "gdd_early": int(round(ye["gdd"].mean())),
        "gdd_late": int(round(yl["gdd"].mean())),
        "season_rain_early_mm": int(round(ye["season_rain"].mean())),
        "season_rain_late_mm": int(round(yl["season_rain"].mean())),
        "season_rain_change_pct": r(rain_change, 1) if rain_change is not None else None,
        "annual_rain_late_mm": int(round(annual_rain)),
        "headline": outlook
    }

    # ---- 3) Fire-weather conditions (hot + dry month) ----
    hot_cut = df.groupby("month")["T2M_MAX"].quantile(0.75)
    rh_cut = df.groupby("month")["RH2M"].quantile(0.25)
    df["fire_wx"] = (df["T2M_MAX"] >= df["month"].map(hot_cut)) & (
        (df["PRECTOTCORR"] <= df["month"].map(dry_cut)) | (df["RH2M"] <= df["month"].map(rh_cut)))
    per_year_early = df.loc[early.index, "fire_wx"].sum() / len(early_years)
    per_year_late = df.loc[late.index, "fire_wx"].sum() / len(late_years)
    flagged = df[df["fire_wx"]]
    if per_year_late > per_year_early * 1.25 and per_year_late - per_year_early >= 0.3:
        fire_headline = "Fire-weather months are becoming more frequent"
    elif per_year_late < per_year_early * 0.8 and per_year_early - per_year_late >= 0.3:
        fire_headline = "Fire-weather months are becoming less frequent"
    else:
        fire_headline = "No clear change in fire-weather months"
    fire = {
        "months_per_year_early": r(per_year_early, 2),
        "months_per_year_late": r(per_year_late, 2),
        "wind_in_fire_months": r(flagged["WS10M"].mean(), 2) if len(flagged) else None,
        "wind_all_months": r(df["WS10M"].mean(), 2),
        "headline": fire_headline
    }

    return {"source": "NASA POWER monthly API (MERRA-2)", "periods": periods,
            "heat_drought": heat, "growing": grow, "fire_weather": fire}


# ---------------------------------------------------------------
# Public "case file": deseasonalized trend, evidence level and the
# data behind the visualizations (yearly trend + band, stripes,
# early-vs-late monthly comparison)
# ---------------------------------------------------------------
def public_summary(dates, values):
    years = np.array([int(d[:4]) for d in dates])
    months = np.array([int(d[5:7]) for d in dates])
    y = np.array(values, dtype=float)
    clim = {m: float(np.mean(y[months == m])) for m in range(1, 13) if np.any(months == m)}
    anom = y - np.array([clim[m] for m in months])
    t = years + (months - 0.5) / 12.0

    sen, _, sen_lo, sen_hi = sens_slope(t, anom, 0.05)
    hr = mann_kendall_hamed_rao(t, anom, 0.05)
    p = float(hr["p"])
    span_years = len(y) / 12.0
    if p < 0.01:
        evidence = "strong"
    elif p < 0.05:
        evidence = "moderate"
    elif p < 0.10:
        evidence = "weak"
    else:
        evidence = "none"

    # yearly means + least-squares line with 95% confidence band for the line
    ann_years, ann_vals = _yearly_series(years, months, y, tuple(range(1, 13)))
    yearly = {"years": [int(v) for v in ann_years], "values": [_r(v, 4) for v in ann_vals]}
    if len(ann_vals) >= 4:
        lr = stats.linregress(ann_years, ann_vals)
        fit = lr.intercept + lr.slope * ann_years
        n = len(ann_vals)
        s = math.sqrt(float(np.sum((ann_vals - fit) ** 2)) / (n - 2))
        sxx = float(np.sum((ann_years - ann_years.mean()) ** 2))
        half = stats.t.ppf(0.975, n - 2) * s * np.sqrt(1.0 / n + (ann_years - ann_years.mean()) ** 2 / sxx)
        yearly.update({"fit": [_r(v, 4) for v in fit], "band_low": [_r(v, 4) for v in fit - half],
                       "band_high": [_r(v, 4) for v in fit + half],
                       "stripes": [_r(v, 4) for v in ann_vals - ann_vals.mean()]})

    # early vs late average for each calendar month
    full_years = sorted(set(int(v) for v in ann_years))
    seg = max(1, min(10, len(full_years) // 3)) if full_years else 0
    month_compare = None
    if seg:
        early_set, late_set = set(full_years[:seg]), set(full_years[-seg:])
        early = [_r(np.mean(y[(months == m) & np.isin(years, list(early_set))]), 4) for m in range(1, 13)]
        late = [_r(np.mean(y[(months == m) & np.isin(years, list(late_set))]), 4) for m in range(1, 13)]
        month_compare = {"early_label": f"{full_years[0]}–{full_years[seg - 1]}",
                         "late_label": f"{full_years[-seg]}–{full_years[-1]}",
                         "early": early, "late": late}

    return {
        "sen_per_decade": _r(sen * 10, 5),
        "sen_ci_low": _r(sen_lo * 10, 5),
        "sen_ci_high": _r(sen_hi * 10, 5),
        "total_change": _r(sen * span_years, 4),
        "span_years": _r(span_years, 1),
        "p_value": _r(p, 6),
        "evidence": evidence,
        "direction": "up" if sen > 0 else "down" if sen < 0 else "flat",
        "period_mean": _r(float(np.mean(y)), 4),
        "yearly": yearly,
        "month_compare": month_compare,
        "test": "Mann-Kendall (Hamed-Rao autocorrelation-corrected) on deseasonalized monthly anomalies; slope = Sen's slope"
    }


# ---------------------------------------------------------------
# LIVE weather & air quality (Open-Meteo, free, no API key).
# Used only for the live temperature and wind map layers;
# all trend analysis still uses NASA POWER.
# Attribution required: "Weather data by Open-Meteo.com" (CC BY 4.0).
# ---------------------------------------------------------------
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_live_cache = {}
LIVE_CACHE_SECONDS = 600


def _cached(key, fn):
    now = time.time()
    hit = _live_cache.get(key)
    if hit and now - hit[0] < LIVE_CACHE_SECONDS:
        return hit[1]
    value = fn()
    _live_cache[key] = (now, value)
    if len(_live_cache) > 500:
        for k in sorted(_live_cache, key=lambda k: _live_cache[k][0])[:100]:
            _live_cache.pop(k, None)
    return value


@app.route("/api/live/grid", methods=["GET"])
def live_grid():
    """Hourly temperature and wind on a coarse grid covering the visible map (next 24 h)."""
    try:
        south = max(-85.0, float(request.args.get("south")))
        north = min(85.0, float(request.args.get("north")))
        west = float(request.args.get("west"))
        east = float(request.args.get("east"))
        nx = int(np.clip(int(request.args.get("nx", 12)), 3, 14))
        ny = int(np.clip(int(request.args.get("ny", 8)), 3, 10))
    except (TypeError, ValueError):
        return jsonify({"error": "south, north, west, east are required numbers"}), 400
    if east - west >= 360:
        west, east = -180.0, 180.0
    if north <= south or east <= west:
        return jsonify({"error": "Invalid bounds"}), 400

    # round the grid so nearby views share the cache
    step_lat, step_lon = (north - south) / (ny - 1), (east - west) / (nx - 1)
    lats = [round(south + i * step_lat, 2) for i in range(ny)]
    lons = [round(west + j * step_lon, 2) for j in range(nx)]
    pts = [(la, ((lo + 180.0) % 360.0) - 180.0) for la in lats for lo in lons]

    def load():
        resp = requests.get(OPEN_METEO_FORECAST_URL, params={
            "latitude": ",".join(f"{p[0]:.2f}" for p in pts),
            "longitude": ",".join(f"{p[1]:.2f}" for p in pts),
            "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
            "forecast_hours": 25, "wind_speed_unit": "ms", "timezone": "GMT"}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            payload = [payload]
        times = payload[0]["hourly"]["time"]
        temp, u, v = [], [], []
        for item in payload:
            h = item["hourly"]
            temp.append(h["temperature_2m"])
            sp = np.array([x if x is not None else np.nan for x in h["wind_speed_10m"]], dtype=float)
            di = np.radians(np.array([x if x is not None else np.nan for x in h["wind_direction_10m"]], dtype=float))
            # meteorological direction = where the wind blows FROM
            u.append([_r(x, 2) for x in -sp * np.sin(di)])
            v.append([_r(x, 2) for x in -sp * np.cos(di)])
        return {"lats": lats, "lons": lons, "times": times, "temperature": temp, "u": u, "v": v,
                "source": "Open-Meteo.com (weather model data, CC BY 4.0)"}

    try:
        key = ("grid", lats[0], lats[-1], lons[0], lons[-1], nx, ny)
        return jsonify(_cached(key, load))
    except Exception as e:
        return jsonify({"error": f"Live map data unavailable ({str(e)[:80]})"}), 502


@app.route("/")
def index():
    # Served as plain HTML (no Jinja templating needed)
    return Response(HTML_TEMPLATE, mimetype="text/html")


@app.route("/api/place", methods=["GET"])
def get_place():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon are required numbers"}), 400
    if not (-90 <= lat <= 90):
        return jsonify({"error": "lat must be between -90 and 90"}), 400
    return jsonify(place_for(lat, lon))


@app.route("/api/variables", methods=["GET"])
def get_variables():
    return jsonify({"variables": VARIABLES_CATALOG, "max_year": latest_complete_year(),
                    "min_span_years": MIN_SPAN_YEARS})


@app.route("/api/analyze", methods=["POST"])
def analyze_region():
    data = request.get_json() or {}
    lat = float(data.get("lat", 25.2048))
    lon = float(data.get("lon", 55.2708))
    parameter = data.get("parameter", "T2M")
    start_year = int(data.get("start_year", 2001))
    end_year = int(data.get("end_year", 2025))
    start_year, end_year = clamp_years(parameter, start_year, end_year)

    try:
        dates, values, source_name = fetch_nasa_power_data(lat, lon, parameter, start_year, end_year)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    t = np.arange(len(values))
    y = np.array(values)
    slope_per_month, intercept, r_value, p_value_ttest, std_err = stats.linregress(t, y)
    slope_per_decade = slope_per_month * 120.0
    ci_95_decade = 1.96 * std_err * 120.0

    mk_direction, mk_p_value, mk_z, mk_tau = mann_kendall_test(y)
    is_significant = bool(p_value_ttest < 0.05 or mk_p_value < 0.05)

    ml_results = run_ml_and_anomalies(dates, values, parameter, forecast_months=36)
    try:
        challenge_modules = compute_power_indicators(lat, lon, start_year, end_year)
    except Exception as e:
        challenge_modules = {"error": f"Indicators unavailable: {str(e)[:120]}"}

    change_word = "increasing" if slope_per_decade > 0 else "decreasing"
    sig_word = "statistically significant" if is_significant else "not yet statistically significant (natural variability dominates)"

    verdict = (
        f"Over {start_year}–{end_year}, this variable is {change_word} at a rate of "
        f"{slope_per_decade:+.3f} units/decade. Scientist Significance Check: {sig_word} "
        f"(t-test p={p_value_ttest:.4f}, Mann-Kendall p={mk_p_value:.4f})."
    )

    return jsonify({
        "location": {"lat": round(lat, 4), "lon": round(lon, 4)},
        "place": place_for(lat, lon),
        "parameter": parameter,
        "source": source_name,
        "dates": dates,
        "values": values,
        "statistics": {
            "mean": round(float(np.mean(y)), 2),
            "min": round(float(np.min(y)), 2),
            "max": round(float(np.max(y)), 2),
            "slope_per_decade": round(float(slope_per_decade), 4),
            "ci_95_decade": round(float(ci_95_decade), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "ttest_p_value": round(float(p_value_ttest), 5),
            "mann_kendall_p_value": mk_p_value,
            "mann_kendall_z": mk_z,
            "mann_kendall_trend": mk_direction,
            "is_significant": is_significant,
            "verdict": verdict
        },
        "ml": ml_results,
        "challenges_integration": challenge_modules,
        "public": public_summary(dates, values)
    })


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Planet Pulse | Detecting Earth's Environmental Trends Through NASA Earth Observations.</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --bg-space: #030712;
      --panel-bg: rgba(15, 23, 42, 0.85);
      --border-glow: rgba(56, 189, 248, 0.3);
      --accent-cyan: #38bdf8;
      --accent-purple: #c084fc;
      --accent-emerald: #34d399;
      --accent-amber: #fbbf24;
      --accent-rose: #fb7185;
      --up: #ef4444;
      --down: #3b82f6;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Inter', sans-serif; background-color: var(--bg-space); color: var(--text-main); min-height: 100vh; padding: 24px; }
    body.locked { overflow: hidden; }
    .cosmic-bg {
      position: fixed; inset: 0; z-index: -1;
      background: radial-gradient(circle at 20% 15%, rgba(56, 189, 248, 0.15), transparent 45%),
                  radial-gradient(circle at 85% 80%, rgba(192, 132, 252, 0.15), transparent 50%),
                  linear-gradient(180deg, #030712 0%, #0f172a 100%);
    }
    .hidden { display: none !important; }
    .glass-panel { background: var(--panel-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--border-glow); border-radius: 20px; box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6); }
    .accent { color: var(--accent-cyan); }
    .kicker { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; color: var(--text-muted); text-transform: uppercase; }
    .muted { color: var(--text-muted); }
    button { font-family: 'Inter', sans-serif; }
    .primary-glow-btn {
      width: 100%; padding: 15px; border: none; border-radius: 14px;
      background: linear-gradient(135deg, #0284c7, #9333ea); color: white;
      font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem; cursor: pointer;
      box-shadow: 0 6px 24px rgba(56, 189, 248, 0.4); transition: transform 0.2s, box-shadow 0.2s, opacity 0.2s;
    }
    .primary-glow-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(147, 51, 234, 0.6); }
    .primary-glow-btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
    .ghost-btn { background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.45); color: var(--text-main); border-radius: 10px; padding: 9px 14px; cursor: pointer; font-size: 0.82rem; font-weight: 600; }
    .ghost-btn:hover { background: rgba(56, 189, 248, 0.22); }
    .ghost-btn.active { background: var(--accent-cyan); color: #030712; }
    .space-select, .space-input { width: 100%; background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(56, 189, 248, 0.4); color: var(--text-main); padding: 10px 12px; border-radius: 12px; font-size: 0.9rem; outline: none; }
    .space-select:focus, .space-input:focus { border-color: var(--accent-cyan); }
    .mini-label { display: block; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }
    .seg { display: flex; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 3px; gap: 2px; }
    .seg button { flex: 1; background: transparent; border: none; color: var(--text-muted); padding: 8px 10px; border-radius: 9px; cursor: pointer; font-size: 0.8rem; font-weight: 600; }
    .seg button.active { background: rgba(56, 189, 248, 0.2); color: var(--text-main); }
    .warn-box { font-size: 0.82rem; color: #fde68a; background: rgba(251, 191, 36, 0.1); border-left: 3px solid var(--accent-amber); border-radius: 8px; padding: 10px 12px; line-height: 1.45; }
    .err-box { font-size: 0.85rem; color: #fecdd3; background: rgba(251, 113, 133, 0.12); border-left: 3px solid var(--accent-rose); border-radius: 8px; padding: 12px 14px; line-height: 1.45; }

    /* ---------- Welcome (mode chooser) ---------- */
    .welcome-overlay { position: fixed; inset: 0; z-index: 3000; background: rgba(3, 7, 18, 0.88); backdrop-filter: blur(10px); display: flex; align-items: center; justify-content: center; padding: 24px; overflow-y: auto; }
    .welcome-card { max-width: 820px; width: 100%; padding: 38px 40px; text-align: center; animation: rise 0.5s ease; }
    @keyframes rise { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
    .welcome-card h1 { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 500; margin: 14px 0 10px; line-height: 1.15; }
    .welcome-card > p { color: var(--text-muted); line-height: 1.6; max-width: 640px; margin: 0 auto; }
    .welcome-q { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; margin: 28px 0 16px; }
    .mode-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; text-align: left; }
    @media(max-width: 700px) { .mode-cards { grid-template-columns: 1fr; } }
    .mode-card { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 18px; padding: 22px; cursor: pointer; color: var(--text-main); transition: all 0.2s; text-align: left; }
    .mode-card:hover { border-color: var(--accent-cyan); transform: translateY(-3px); box-shadow: 0 10px 30px rgba(56, 189, 248, 0.25); }
    .mode-card.sci:hover { border-color: var(--accent-purple); box-shadow: 0 10px 30px rgba(192, 132, 252, 0.25); }
    .mode-card .mc-icon { font-size: 1.8rem; }
    .mode-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; margin: 8px 0 6px; }
    .mode-card p { font-size: 0.86rem; color: var(--text-muted); line-height: 1.5; }
    .mode-card ul { margin: 10px 0 0 18px; font-size: 0.8rem; color: var(--text-muted); line-height: 1.6; }
    .welcome-demo { margin-top: 22px; background: transparent; border: none; color: var(--accent-cyan); font-size: 0.95rem; font-weight: 600; cursor: pointer; }
    .welcome-demo:hover { text-decoration: underline; }
    .welcome-note { margin-top: 10px; font-size: 0.78rem; color: var(--text-muted); }

    /* ---------- Header ---------- */
    .navbar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px; padding: 16px 24px; margin-bottom: 18px; }
    .brand { display: flex; align-items: center; gap: 14px; }
    .pulse-orb { width: 16px; height: 16px; border-radius: 50%; background: var(--accent-cyan); box-shadow: 0 0 18px var(--accent-cyan); animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.4); opacity: 0.6; } }
    .brand h1 { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; letter-spacing: 0.04em; }
    .subtitle { font-size: 0.8rem; color: var(--text-muted); margin-top: 2px; }
    .tab-nav { display: flex; flex-wrap: wrap; gap: 4px; background: rgba(30, 41, 59, 0.6); padding: 5px; border-radius: 14px; border: 1px solid rgba(148, 163, 184, 0.2); }
    .tab-btn { background: transparent; border: 1px solid transparent; color: var(--text-muted); padding: 8px 14px; border-radius: 10px; cursor: pointer; font-size: 0.86rem; font-weight: 600; transition: all 0.2s; }
    .tab-btn:hover { color: var(--text-main); background: rgba(56, 189, 248, 0.08); }
    .tab-btn.active { color: var(--text-main); background: rgba(56, 189, 248, 0.18); border-color: rgba(56, 189, 248, 0.55); }
    .mode-toggle { display: flex; align-items: center; gap: 8px; background: rgba(30, 41, 59, 0.6); padding: 5px 10px; border-radius: 999px; border: 1px solid rgba(148, 163, 184, 0.2); }
    .mode-toggle span { font-size: 0.78rem; color: var(--text-muted); }
    .mode-btn { background: transparent; border: none; color: var(--text-muted); padding: 6px 14px; border-radius: 999px; cursor: pointer; font-size: 0.82rem; font-weight: 600; }
    .mode-btn.active { background: var(--accent-cyan); color: #030712; }
    .mode-btn.active.sci { background: var(--accent-purple); }

    .tab-intro { padding: 2px 4px 16px; }
    .tab-intro h2 { font-family: 'Space Grotesk', sans-serif; font-weight: 500; font-size: 1.7rem; line-height: 1.2; margin: 4px 0 6px; }
    .tab-intro p { color: var(--text-muted); font-size: 0.92rem; max-width: 820px; line-height: 1.5; }
    .tab-intro.swap { animation: rise 0.4s ease; }

    /* ---------- Workspace ---------- */
    .workspace { display: grid; grid-template-columns: 430px 1fr; gap: 20px; margin-bottom: 22px; align-items: start; }
    @media(max-width: 1100px) { .workspace { grid-template-columns: 1fr; } }
    .side { display: flex; flex-direction: column; gap: 16px; }
    .card { padding: 22px; }
    .card h2 { font-family: 'Space Grotesk', sans-serif; font-size: 1.12rem; margin-bottom: 6px; }
    .helper { font-size: 0.84rem; color: var(--text-muted); line-height: 1.45; margin-bottom: 14px; }

    .step { padding: 14px 0 16px; border-bottom: 1px solid rgba(148, 163, 184, 0.14); }
    .step:first-child { padding-top: 0; }
    .step-head { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .step-num { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.95rem; border: 2px solid rgba(56, 189, 248, 0.5); color: var(--accent-cyan); flex-shrink: 0; transition: all 0.3s; }
    .step.done .step-num { background: var(--accent-emerald); border-color: var(--accent-emerald); color: #030712; }
    .step-head h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.02rem; }
    .step-head p { font-size: 0.8rem; color: var(--text-muted); }
    .var-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; }
    .var-card { background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 10px 6px; cursor: pointer; text-align: center; color: var(--text-main); transition: all 0.15s; }
    .var-card:hover { border-color: rgba(56, 189, 248, 0.6); }
    .var-card.active { border-color: var(--accent-cyan); background: rgba(56, 189, 248, 0.16); box-shadow: 0 0 14px rgba(56, 189, 248, 0.3); }
    .var-card .vi { font-size: 1.3rem; display: block; }
    .var-card .vn { font-size: 0.72rem; font-weight: 600; line-height: 1.25; display: block; margin-top: 4px; }
    .var-def { margin-top: 10px; font-size: 0.82rem; line-height: 1.45; background: rgba(56, 189, 248, 0.07); border-left: 3px solid var(--accent-cyan); padding: 9px 11px; border-radius: 8px; }
    .var-def small { display: block; color: var(--text-muted); font-size: 0.72rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
    .where-box { display: flex; gap: 10px; align-items: flex-start; background: rgba(30, 41, 59, 0.5); border: 1px dashed rgba(148, 163, 184, 0.35); border-radius: 12px; padding: 11px 13px; font-size: 0.86rem; }
    .step.done .where-box { border-style: solid; border-color: rgba(52, 211, 153, 0.45); }
    .where-box small { display: block; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; margin-top: 2px; }
    .go-wrap { padding-top: 16px; }
    .go-hint { font-size: 0.78rem; color: var(--text-muted); text-align: center; margin-top: 8px; }

    /* timeline */
    .tl-range { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
    .tl-range strong { font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: var(--accent-cyan); }
    .tl-range small { color: var(--text-muted); font-size: 0.74rem; }
    .tl-slider { position: relative; height: 34px; margin: 0 9px; }
    .tl-track, .tl-fill { position: absolute; top: 15px; height: 4px; border-radius: 2px; }
    .tl-track { left: 0; right: 0; background: rgba(148, 163, 184, 0.25); }
    .tl-fill { background: linear-gradient(90deg, #38bdf8, #c084fc); box-shadow: 0 0 10px rgba(56, 189, 248, 0.5); }
    .tl-slider input[type=range] { position: absolute; left: -9px; top: 6px; width: calc(100% + 18px); height: 22px; margin: 0; background: transparent; pointer-events: none; -webkit-appearance: none; appearance: none; outline: none; }
    .tl-slider input[type=range]::-webkit-slider-runnable-track { background: transparent; height: 22px; }
    .tl-slider input[type=range]::-moz-range-track { background: transparent; }
    .tl-slider input[type=range]::-webkit-slider-thumb { pointer-events: auto; -webkit-appearance: none; width: 18px; height: 18px; margin-top: 2px; border-radius: 50%; background: #f8fafc; border: 3px solid #38bdf8; cursor: grab; box-shadow: 0 0 10px rgba(56, 189, 248, 0.6); }
    .tl-slider input[type=range]::-moz-range-thumb { pointer-events: auto; width: 14px; height: 14px; border-radius: 50%; background: #f8fafc; border: 3px solid #38bdf8; cursor: grab; }
    .tl-slider input.tl-end::-webkit-slider-thumb { border-color: #c084fc; }
    .tl-slider input.tl-end::-moz-range-thumb { border-color: #c084fc; }
    .tl-scale { position: relative; height: 16px; margin: 2px 9px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-muted); }
    .tl-scale span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
    .tl-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .tl-chip { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(56, 189, 248, 0.3); color: var(--text-main); padding: 5px 10px; border-radius: 999px; font-size: 0.74rem; cursor: pointer; }
    .tl-chip:hover { background: rgba(56, 189, 248, 0.2); }
    .tl-note { font-size: 0.72rem; color: var(--text-muted); margin-top: 8px; line-height: 1.4; }

    /* map */
    .map-card { padding: 14px; position: relative; }
    #world-map { height: 520px; width: 100%; border-radius: 14px; border: 1px solid rgba(148, 163, 184, 0.25); z-index: 1; cursor: crosshair; }
    .map-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; padding: 2px 4px 12px; }
    .map-place { font-size: 0.92rem; font-weight: 600; }
    .map-place small { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 0.74rem; margin-left: 6px; }
    .map-tools { display: flex; gap: 8px; flex-wrap: wrap; }
    .map-legend { position: absolute; left: 28px; bottom: 28px; z-index: 500; background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 12px; padding: 10px 12px; font-size: 0.74rem; max-width: 260px; }
    .legend-bar { height: 10px; border-radius: 5px; background: linear-gradient(90deg, #2563eb, #e2e8f0, #dc2626); margin: 6px 0 3px; }
    .legend-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: var(--text-muted); }
    .leaflet-popup-content { font-family: 'Inter', sans-serif; font-size: 0.84rem; }

    /* ---------- Results / case file ---------- */
    .results { position: relative; min-height: 120px; }
    .loader { position: absolute; inset: 0; z-index: 50; background: rgba(3, 7, 18, 0.85); border-radius: 20px; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding-top: 80px; gap: 14px; min-height: 260px; }
    .radar-spinner { width: 50px; height: 50px; border: 4px solid rgba(56, 189, 248, 0.2); border-top-color: var(--accent-cyan); border-radius: 50%; animation: spin 0.9s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .empty-state { padding: 34px; text-align: center; color: var(--text-muted); }
    .empty-state .big { font-size: 2.2rem; }
    .empty-state h3 { font-family: 'Space Grotesk', sans-serif; color: var(--text-main); margin: 8px 0 6px; font-size: 1.15rem; }
    .cf-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; padding: 0 4px; }
    .cf-head h2 { font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem; font-weight: 500; }
    .cf-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; }
    .cf-card { padding: 22px; }
    .span-12 { grid-column: span 12; } .span-7 { grid-column: span 7; } .span-5 { grid-column: span 5; } .span-6 { grid-column: span 6; }
    @media(max-width: 1100px) { .span-7, .span-5, .span-6 { grid-column: span 12; } }
    .cf-q { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; color: var(--accent-cyan); text-transform: uppercase; margin-bottom: 6px; }
    .cf-card h3 { font-family: 'Space Grotesk', sans-serif; font-size: 1.08rem; margin-bottom: 4px; }
    .cf-card .helper { margin-bottom: 12px; }
    .headline-num { display: flex; align-items: center; gap: 16px; margin: 10px 0 12px; }
    .headline-arrow { width: 64px; height: 64px; border-radius: 18px; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 700; flex-shrink: 0; }
    .headline-arrow.up { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }
    .headline-arrow.down { background: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.5); }
    .headline-arrow.flat { background: rgba(148, 163, 184, 0.15); color: var(--text-muted); border: 1px solid rgba(148, 163, 184, 0.4); }
    .headline-value { font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 700; line-height: 1; }
    .headline-value small { display: block; font-size: 0.9rem; font-weight: 500; color: var(--text-muted); margin-top: 6px; }
    .headline-text { font-size: 0.95rem; line-height: 1.55; }
    .headline-sub { font-size: 0.8rem; color: var(--text-muted); margin-top: 8px; }
    .meter { position: relative; margin: 18px 0 10px; }
    .meter-track { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
    .meter-seg { height: 14px; border-radius: 4px; background: rgba(148, 163, 184, 0.15); }
    .meter-seg.on.l0 { background: #64748b; } .meter-seg.on.l1 { background: #fbbf24; } .meter-seg.on.l2 { background: #34d399; } .meter-seg.on.l3 { background: #10b981; box-shadow: 0 0 12px rgba(16, 185, 129, 0.6); }
    .meter-labels { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; font-size: 0.7rem; color: var(--text-muted); margin-top: 6px; text-align: center; }
    .meter-labels .cur { color: var(--text-main); font-weight: 700; }
    .evidence-word { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; margin-top: 6px; }
    .chart-box { position: relative; height: 330px; }
    .chart-box.short { height: 260px; }
    .stripes { display: flex; height: 92px; border-radius: 12px; overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.2); }
    .stripes div { flex: 1; position: relative; }
    .stripes div:hover { outline: 2px solid #fff; z-index: 2; }
    .stripe-axis { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-muted); margin-top: 6px; }
    .stripe-legend { display: flex; align-items: center; gap: 8px; font-size: 0.74rem; color: var(--text-muted); margin-top: 8px; }
    .stripe-legend span.sw { width: 90px; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #08306b, #6baed6, #f7f7f7, #fb6a4a, #67000d); }
    .stripe-tip { font-size: 0.8rem; margin-top: 6px; min-height: 1.2em; }
    .unusual-list { display: flex; flex-direction: column; gap: 8px; }
    .unusual-item { display: grid; grid-template-columns: 90px 1fr; gap: 10px; align-items: center; background: rgba(30, 41, 59, 0.5); border-radius: 10px; padding: 9px 12px; font-size: 0.82rem; }
    .unusual-item b { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--accent-rose); }
    .seg-mini { display: inline-flex; margin-bottom: 10px; }
    .seg-mini button { padding: 6px 12px; font-size: 0.76rem; }

    /* ---------- Side panels: discover / examples ---------- */
    .subtab-row { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; background: rgba(15, 23, 42, 0.8); padding: 4px; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.2); margin-bottom: 16px; }
    .subtab-btn { background: transparent; border: 1px solid transparent; color: var(--text-muted); padding: 9px 6px; border-radius: 9px; cursor: pointer; font-size: 0.78rem; font-weight: 600; line-height: 1.25; }
    .subtab-btn.active { color: var(--text-main); background: rgba(56, 189, 248, 0.16); border-color: rgba(56, 189, 248, 0.45); }
    .loc-box { background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px; padding: 12px; margin-bottom: 12px; }
    .loc-box h4 { font-size: 0.74rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.06em; margin-bottom: 6px; }
    .loc-a h4 { color: var(--accent-cyan); } .loc-b h4 { color: var(--accent-rose); }
    .loc-name { font-size: 0.86rem; font-weight: 600; }
    .loc-inputs { display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; margin-top: 10px; }
    .loc-inputs .space-input { padding: 8px 10px; font-size: 0.84rem; }
    .pick-btn { background: rgba(251, 113, 133, 0.12); border: 1px solid rgba(251, 113, 133, 0.45); color: var(--text-main); border-radius: 10px; padding: 0 12px; cursor: pointer; font-size: 0.78rem; font-weight: 600; white-space: nowrap; }
    .pick-btn.picking { background: var(--accent-rose); color: #030712; }
    .example-list { display: flex; flex-direction: column; gap: 10px; }
    .example { background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; padding: 14px; }
    .example-top { display: flex; gap: 10px; align-items: flex-start; }
    .example-icon { font-size: 1.4rem; }
    .example h4 { font-size: 0.92rem; margin-bottom: 3px; }
    .example p { font-size: 0.8rem; color: var(--text-muted); line-height: 1.4; }
    .example-actions { display: flex; gap: 8px; margin-top: 10px; }
    .example-actions button { flex: 1; padding: 8px; font-size: 0.78rem; }
    .play-btn { background: linear-gradient(135deg, #0284c7, #9333ea); border: none; color: #fff; border-radius: 10px; cursor: pointer; font-weight: 700; }

    /* discover results */
    .disc-list { display: flex; flex-direction: column; gap: 8px; margin-top: 14px; }
    .disc-row { display: grid; grid-template-columns: 26px 1fr auto; gap: 10px; align-items: center; background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(56, 189, 248, 0.14); border-radius: 12px; padding: 10px 12px; font-size: 0.84rem; }
    .disc-row.sig { border-color: rgba(52, 211, 153, 0.45); }
    .disc-row.na { opacity: 0.5; }
    .disc-rank { font-family: 'JetBrains Mono', monospace; color: var(--accent-cyan); font-weight: 700; text-align: center; }
    .disc-meta { font-size: 0.76rem; color: var(--text-muted); margin-top: 2px; line-height: 1.4; }
    .tag { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; font-weight: 700; padding: 2px 7px; border-radius: 6px; margin-left: 4px; vertical-align: middle; }
    .tag.ok { background: rgba(52, 211, 153, 0.18); color: var(--accent-emerald); }
    .tag.no { background: rgba(148, 163, 184, 0.15); color: var(--text-muted); }
    .cmp-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
    .cmp-table th, .cmp-table td { padding: 8px 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); text-align: left; }
    .cmp-table th { color: var(--text-muted); font-size: 0.76rem; font-weight: 600; }
    .cmp-a { color: var(--accent-cyan); } .cmp-b { color: var(--accent-rose); }

    /* ---------- Scientist panel ---------- */
    .sci-panel { padding: 22px; margin-top: 18px; border-color: rgba(192, 132, 252, 0.45); }
    .sci-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; padding: 5px 10px; border-radius: 8px; background: rgba(192, 132, 252, 0.15); color: var(--accent-purple); border: 1px solid rgba(192, 132, 252, 0.4); }
    .sci-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
    .sci-top h2 { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; }
    .sci-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; align-items: end; margin-bottom: 14px; }
    .sci-tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); margin-bottom: 16px; }
    .sci-tab { background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--text-muted); padding: 10px 12px; cursor: pointer; font-size: 0.84rem; font-weight: 600; }
    .sci-tab.active { color: var(--text-main); border-bottom-color: var(--accent-purple); }
    .sci-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media(max-width: 1100px) { .sci-grid { grid-template-columns: 1fr; } }
    .sci-block { background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 14px; padding: 16px; overflow-x: auto; }
    .sci-block.wide { grid-column: 1 / -1; }
    .sci-block h3 { font-family: 'Space Grotesk', sans-serif; font-size: 0.98rem; margin-bottom: 4px; color: var(--accent-cyan); }
    .sci-block .helper { font-size: 0.78rem; margin-bottom: 10px; }
    .sci-summary { font-size: 0.92rem; line-height: 1.55; background: rgba(192, 132, 252, 0.08); border-left: 4px solid var(--accent-purple); padding: 13px 15px; border-radius: 12px; margin-bottom: 16px; }
    .sci-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    .sci-table th { text-align: left; color: var(--text-muted); font-weight: 600; font-size: 0.72rem; padding: 6px; border-bottom: 1px solid rgba(148, 163, 184, 0.25); white-space: nowrap; }
    .sci-table td { padding: 7px 6px; border-bottom: 1px solid rgba(148, 163, 184, 0.1); vertical-align: top; }
    .sci-table td.num { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; white-space: nowrap; }
    .sci-table small { display: block; color: var(--text-muted); font-size: 0.7rem; margin-top: 2px; }
    .pill-yes { color: var(--accent-emerald); font-weight: 700; } .pill-no { color: var(--text-muted); }
    .diag-row { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1); font-size: 0.8rem; }
    .diag-row small { display: block; color: var(--text-muted); font-size: 0.72rem; }
    .diag-ok { color: var(--accent-emerald); font-weight: 700; font-size: 0.72rem; } .diag-warn { color: var(--accent-amber); font-weight: 700; font-size: 0.72rem; }
    .bar-cell { position: relative; height: 10px; background: rgba(148, 163, 184, 0.12); border-radius: 5px; min-width: 90px; }
    .bar-cell span { position: absolute; top: 0; height: 10px; border-radius: 5px; }
    .bar-cell .mid { position: absolute; left: 50%; top: -2px; width: 1px; height: 14px; background: rgba(248, 250, 252, 0.5); }
    .detect-big { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700; color: var(--accent-cyan); }
    .det-meter { height: 10px; background: rgba(148, 163, 184, 0.15); border-radius: 5px; overflow: hidden; margin: 10px 0 6px; }
    .det-meter span { display: block; height: 100%; background: linear-gradient(90deg, #0284c7, #34d399); }
    .prov-table td:first-child { color: var(--text-muted); width: 30%; }
    .prov-table a { color: var(--accent-cyan); word-break: break-all; }
    .sci-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .methods-text { font-size: 0.8rem; color: var(--text-muted); line-height: 1.55; margin-top: 12px; background: rgba(3, 7, 18, 0.5); padding: 12px; border-radius: 10px; }
    .anom-card { background: rgba(251, 113, 133, 0.07); border: 1px solid rgba(251, 113, 133, 0.28); border-radius: 12px; padding: 14px; }
    .anom-card .ad { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--accent-rose); font-weight: 700; }
    .anom-card .av { font-size: 1.1rem; font-weight: 700; margin: 4px 0; }
    .anom-card details { margin-top: 8px; font-size: 0.8rem; }
    .anom-card summary { cursor: pointer; color: var(--accent-amber); font-weight: 600; }
    .anom-card .prec { white-space: pre-line; color: #fde68a; margin-top: 6px; line-height: 1.45; }
    .ind-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    @media(max-width: 1100px) { .ind-grid { grid-template-columns: 1fr; } }
    .ind-box { background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 14px; padding: 14px; font-size: 0.82rem; line-height: 1.55; color: var(--text-muted); }
    .ind-box h4 { color: var(--accent-cyan); font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; margin-bottom: 6px; }
    .ind-box strong { color: var(--text-main); }
    .region-bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px; }

    /* ---------- Learn ---------- */
    .learn-view { padding: 26px; }
    .learn-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
    @media(max-width: 1000px) { .learn-cols { grid-template-columns: 1fr; } }
    .learn-list details { border-bottom: 1px solid rgba(148, 163, 184, 0.15); padding: 12px 2px; }
    .learn-list summary { cursor: pointer; font-size: 0.92rem; font-weight: 600; list-style: none; display: flex; justify-content: space-between; gap: 10px; }
    .learn-list summary::after { content: '+'; color: var(--accent-cyan); font-weight: 700; }
    .learn-list details[open] summary::after { content: '−'; }
    .learn-list summary::-webkit-details-marker { display: none; }
    .learn-body { font-size: 0.85rem; color: var(--text-muted); line-height: 1.6; margin-top: 8px; }
    .learn-body b { color: var(--text-main); font-weight: 600; }
    .learn-section { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; color: var(--accent-cyan); margin: 6px 0 4px; text-transform: uppercase; }

    /* ---------- Demo tour ---------- */
    .demo-focus { outline: 3px solid var(--accent-cyan) !important; outline-offset: 4px; box-shadow: 0 0 0 8px rgba(56, 189, 248, 0.18), 0 0 40px rgba(56, 189, 248, 0.45) !important; transition: box-shadow 0.3s; }
    .demo-bar { position: fixed; left: 50%; bottom: 22px; transform: translateX(-50%); z-index: 2500; width: min(680px, calc(100vw - 32px)); padding: 16px 18px; border-color: rgba(56, 189, 248, 0.6); animation: rise 0.35s ease; }
    .demo-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
    .demo-text { font-size: 0.95rem; line-height: 1.5; min-height: 2.9em; }
    .demo-progress { height: 4px; background: rgba(148, 163, 184, 0.2); border-radius: 2px; overflow: hidden; margin: 10px 0; }
    .demo-progress span { display: block; height: 100%; background: linear-gradient(90deg, #38bdf8, #c084fc); width: 0; }
    .demo-controls { display: flex; gap: 8px; justify-content: flex-end; }

    /* ================= LANDING (3D night Earth) ================= */
    #landing { position: fixed; inset: 0; z-index: 3000; background: #000; overflow: hidden; }
    #globe { position: absolute; inset: 0; }
    #globe-fallback { position: absolute; width: 900px; height: 900px; right: -180px; top: 50%; transform: translateY(-50%); border-radius: 50%;
      background: radial-gradient(circle at 35% 35%, #1e3a5f 0%, #0b1a33 45%, #020617 70%); box-shadow: 0 0 120px 30px rgba(56, 189, 248, 0.35), inset -60px -40px 120px rgba(0, 0, 0, 0.9); }
    .landing-shade { position: absolute; inset: 0; pointer-events: none; background: linear-gradient(90deg, rgba(0, 0, 0, 0.85) 0%, rgba(0, 0, 0, 0.55) 38%, rgba(0, 0, 0, 0) 65%); }
    .landing-content { position: absolute; left: 7vw; top: 50%; transform: translateY(-50%); max-width: 640px; z-index: 2; animation: rise 0.8s ease; }
    .landing-content h1 { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.4rem, 5vw, 4.2rem); font-weight: 700; line-height: 1.05; letter-spacing: -0.02em; margin: 16px 0 18px; }
    .landing-content h1 .glow { background: linear-gradient(90deg, #38bdf8, #c084fc); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .landing-content > p { font-size: 1.08rem; line-height: 1.6; color: #cbd5e1; max-width: 560px; }
    .landing-buttons { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 30px; }
    .landing-btn { padding: 16px 26px; border-radius: 14px; font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.02rem; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; text-align: left; }
    .landing-btn small { display: block; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.74rem; opacity: 0.85; margin-top: 3px; }
    .landing-btn.primary { background: linear-gradient(135deg, #0284c7, #9333ea); border: none; color: #fff; box-shadow: 0 8px 30px rgba(56, 189, 248, 0.45); }
    .landing-btn.outline { background: rgba(15, 23, 42, 0.55); border: 1.5px solid rgba(248, 250, 252, 0.7); color: #fff; backdrop-filter: blur(6px); }
    .landing-btn:hover { transform: translateY(-3px); }
    .landing-demo { margin-top: 18px; background: none; border: none; color: var(--accent-cyan); font-weight: 600; font-size: 0.95rem; cursor: pointer; }
    .landing-demo:hover { text-decoration: underline; }
    .landing-stats { display: flex; gap: 22px; flex-wrap: wrap; margin-top: 34px; }
    .landing-stats div { font-size: 0.78rem; color: var(--text-muted); }
    .landing-stats b { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; color: #f8fafc; }
    .landing-credit { position: absolute; right: 18px; bottom: 12px; font-size: 0.7rem; color: rgba(148, 163, 184, 0.7); z-index: 2; }

    /* ================= DASHBOARD (map in the centre) ================= */
    .dash { display: grid; grid-template-columns: 340px minmax(0, 1fr) 300px; gap: 18px; margin-bottom: 22px; align-items: start; }
    @media(max-width: 1350px) { .dash { grid-template-columns: 320px minmax(0, 1fr); } .right-col { grid-column: 1 / -1; display: grid !important; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); } }
    @media(max-width: 950px) { .dash { grid-template-columns: 1fr; } }
    .right-col { display: flex; flex-direction: column; gap: 14px; }
    .mini-card { padding: 16px 18px; }
    .mini-card h4 { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em; color: var(--accent-cyan); text-transform: uppercase; margin-bottom: 10px; display: flex; justify-content: space-between; }
    .live-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .live-item { background: rgba(30, 41, 59, 0.55); border-radius: 12px; padding: 10px; }
    .live-item .lv { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; }
    .live-item .ll { font-size: 0.7rem; color: var(--text-muted); }
    .live-temp { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
    .live-temp .big { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }
    .wind-arrow { display: inline-block; transition: transform 0.4s; }
    .aqi-pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; color: #0f172a; margin-left: 4px; }
    .mini-note { font-size: 0.68rem; color: var(--text-muted); margin-top: 8px; }
    .mini-chart { position: relative; height: 150px; }

    /* map overlays */
    .map-card { padding: 0; overflow: hidden; }
    .map-wrap { position: relative; }
    #world-map { height: 640px; border: none; border-radius: 0; }
    .map-overlay { position: absolute; z-index: 700; pointer-events: none; }
    .map-overlay > * { pointer-events: auto; }
    .ov-place { top: 14px; left: 50%; transform: translateX(-50%); }
    .chip { background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 999px; padding: 8px 14px; font-size: 0.85rem; font-weight: 600; backdrop-filter: blur(8px); white-space: nowrap; }
    .chip small { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-weight: 400; font-size: 0.72rem; margin-left: 6px; }
    .ov-layers { top: 70px; right: 12px; display: flex; flex-direction: column; gap: 6px; }
    .layer-btn { width: 132px; display: flex; align-items: center; gap: 8px; background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.3); color: var(--text-main); border-radius: 12px; padding: 8px 10px; cursor: pointer; font-size: 0.78rem; font-weight: 600; backdrop-filter: blur(8px); text-align: left; }
    .layer-btn span.src { margin-left: auto; font-size: 0.58rem; font-family: 'JetBrains Mono', monospace; color: var(--text-muted); }
    .layer-btn.active { border-color: var(--accent-cyan); background: rgba(14, 116, 144, 0.55); box-shadow: 0 0 14px rgba(56, 189, 248, 0.35); }
    .ov-legend { top: 70px; left: 12px; width: 210px; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 14px; padding: 10px 12px; font-size: 0.74rem; backdrop-filter: blur(8px); }
    .ov-legend b { font-size: 0.82rem; }
    .ov-legend .lg-bar { height: 10px; border-radius: 5px; margin: 7px 0 3px; }
    .ov-legend .lg-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; color: var(--text-muted); }
    .ov-legend .lg-src { color: var(--text-muted); font-size: 0.66rem; margin-top: 6px; line-height: 1.35; }
    .ov-time { left: 12px; right: 12px; bottom: 12px; display: flex; align-items: center; gap: 10px; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 14px; padding: 8px 10px; backdrop-filter: blur(8px); }
    .ov-time .play { width: 38px; height: 38px; border-radius: 50%; border: none; background: linear-gradient(135deg, #0284c7, #9333ea); color: #fff; cursor: pointer; font-size: 0.95rem; flex-shrink: 0; }
    .ov-time .tlabel { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.92rem; min-width: 130px; }
    .ov-time .tlabel small { display: block; font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.68rem; color: var(--text-muted); }
    .tsteps { display: flex; gap: 4px; overflow-x: auto; flex: 1; }
    .tstep { background: transparent; border: 1px solid transparent; color: var(--text-muted); border-radius: 8px; padding: 6px 9px; cursor: pointer; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; white-space: nowrap; }
    .tstep.active { background: var(--accent-emerald); color: #04130d; font-weight: 700; }
    .ov-status { bottom: 74px; left: 50%; transform: translateX(-50%); }
    .ov-tools { bottom: 74px; right: 12px; display: flex; gap: 6px; }
    canvas.field-canvas { position: absolute; left: 0; top: 0; pointer-events: none; }
    #world-map .leaflet-bottom.leaflet-right { bottom: 72px; }
    #world-map .leaflet-control-attribution { font-size: 0.62rem; background: rgba(15, 23, 42, 0.7); color: #94a3b8; }
    #world-map .leaflet-control-attribution a { color: #cbd5e1; }
    .leaflet-container { background: #0b1220; }

    /* ================= Investigation cards with NASA satellite images ================= */
    .card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .inv-card { background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 16px; overflow: hidden; cursor: pointer; transition: transform 0.2s, border-color 0.2s; text-align: left; color: var(--text-main); padding: 0; }
    .inv-card:hover { transform: translateY(-3px); border-color: var(--accent-cyan); }
    .inv-img { height: 104px; background-size: cover; background-position: center; position: relative; background-color: #0f172a; }
    .inv-img .badge { position: absolute; left: 8px; bottom: 8px; font-size: 0.6rem; font-family: 'JetBrains Mono', monospace; background: rgba(0, 0, 0, 0.6); padding: 2px 6px; border-radius: 6px; color: #e2e8f0; }
    .inv-body { padding: 9px 11px 11px; }
    .inv-body h4 { font-size: 0.82rem; line-height: 1.25; margin-bottom: 7px; }
    .inv-actions { display: flex; gap: 6px; }
    .inv-actions button { flex: 1; padding: 6px 4px; font-size: 0.7rem; border-radius: 8px; }

    /* ---------- Pulse Guide chatbot (replaces the variable icon grid) ---------- */
    .guide { background: rgba(3, 7, 18, 0.55); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; overflow: hidden; }
    .guide-log { height: 340px; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
    .g-msg { max-width: 92%; padding: 9px 12px; border-radius: 12px; font-size: 0.82rem; line-height: 1.5; }
    .g-msg.bot { align-self: flex-start; background: rgba(30, 41, 59, 0.9); border: 1px solid rgba(56, 189, 248, 0.18); border-bottom-left-radius: 4px; }
    .g-msg.user { align-self: flex-end; background: linear-gradient(135deg, #0369a1, #7e22ce); border-bottom-right-radius: 4px; }
    .g-msg b { color: var(--accent-cyan); }
    .g-opts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .g-opt { background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.4); color: var(--text-main); padding: 6px 10px; border-radius: 999px; font-size: 0.76rem; cursor: pointer; }
    .g-opt:hover { background: rgba(56, 189, 248, 0.25); }
    .g-opt.act { border-color: rgba(52, 211, 153, 0.6); background: rgba(52, 211, 153, 0.14); }
    .g-opt.nav { border-color: rgba(148, 163, 184, 0.35); background: transparent; color: var(--text-muted); }
    .guide-form { display: flex; gap: 6px; padding: 8px; border-top: 1px solid rgba(56, 189, 248, 0.18); }
    .guide-form input { flex: 1; background: rgba(15, 23, 42, 0.95); border: 1px solid rgba(56, 189, 248, 0.35); color: var(--text-main); padding: 9px 11px; border-radius: 10px; font-size: 0.82rem; outline: none; }
    .guide-form button { border: none; border-radius: 10px; padding: 0 14px; color: #fff; cursor: pointer; background: linear-gradient(135deg, #0284c7, #9333ea); }
    .g-typing { color: var(--text-muted); font-size: 0.8rem; }
  </style>
</head>
<body class="locked">
  <div class="cosmic-bg"></div>

  <!-- ================= LANDING: 3D night Earth + mode choice ================= -->
  <section id="landing" aria-label="Welcome">
    <div id="globe"></div>
    <div id="globe-fallback" class="hidden"></div>
    <div class="landing-shade"></div>
    <div class="landing-content">
      <div class="kicker" style="font-size:0.85rem;color:#e2e8f0;">🌍 <b style="color:var(--accent-cyan);letter-spacing:0.18em;">PLANET PULSE</b> · </div>
      <h1>Detecting Earth's Environmental Trends Through <span class="glow">NASA Earth Observations</span></h1>
      <p>Find out <b>what</b> is changing on our planet, <b>where</b> it is changing, <b>how much</b> it has changed, and whether the change is <b>real</b>, using 45 years of NASA data and live Earth maps.</p>
      <div class="landing-buttons">
        <button type="button" class="landing-btn primary" onclick="chooseMode('simple')">🎓 Enter as Public<small>Clear visuals, plain language</small></button>
        <button type="button" class="landing-btn outline" onclick="chooseMode('science')">🔬 Enter as Scientist<small>Full statistics & data export</small></button>
      </div>
      <button type="button" class="landing-demo" onclick="chooseMode('simple', true)">▶ Not sure? Watch a 1-minute demo</button>
      <div class="landing-stats">
        <div><b>11</b>NASA POWER variables</div>
        <div><b id="landing-years">1981–2025</b>monthly record</div>
        <div><b>5</b>map layers</div>
      </div>
    </div>
    <div class="landing-credit">Earth at night: NASA Black Marble (VIIRS) via NASA GIBS</div>
  </section>

  <div id="app" class="hidden">
  <!-- ================= HEADER ================= -->
  <header class="navbar glass-panel">
    <div class="brand" style="cursor:pointer;" onclick="showLanding()" title="Back to the start page">
      <div class="pulse-orb"></div>
      <div>
        <h1>PLANET<span class="accent">PULSE</span></h1>
        <p class="subtitle">Detecting Earth's Environmental Trends Through NASA Earth Observations.</p>
      </div>
    </div>
    <nav class="tab-nav" aria-label="Sections">
      <button class="tab-btn active" data-tab="investigate" onclick="switchTab('investigate')">🔍 Investigate</button>
      <button class="tab-btn" data-tab="discover" onclick="switchTab('discover')">🧭 Discover</button>
      <button class="tab-btn" data-tab="examples" onclick="switchTab('examples')">🎬 Guided Examples</button>
      <button class="tab-btn" data-tab="learn" onclick="switchTab('learn')">📖 Learn</button>
    </nav>
    <div class="mode-toggle">
      <span>Mode</span>
      <button id="btn-simple" class="mode-btn active" onclick="setAudienceMode('simple')">Public</button>
      <button id="btn-science" class="mode-btn sci" onclick="setAudienceMode('science')">Scientist</button>
    </div>
  </header>

  <section id="tab-intro" class="tab-intro">
    <div class="kicker" id="tab-intro-kicker">Investigate</div>
    <h2 id="tab-intro-title">Three steps: <span class="accent">what, where, when.</span></h2>
    <p id="tab-intro-text">Choose a variable, click a place on the map and pick the years. Then press Investigate to see how it has changed and whether the change is real.</p>
  </section>

  <!-- ================= WORKSPACE: side panel + map ================= -->
  <main id="workspace" class="dash">
    <aside class="side">

      <!-- Investigate: the 3-step flow -->
      <div id="panel-investigate" class="card glass-panel tab-panel">
        <div class="step" id="step-what">
          <div class="step-head"><span class="step-num">1</span><div><h3>What? <span class="muted" style="font-size:0.75rem;font-weight:500;">(optional)</span></h3><p>Temperature is used unless you choose another</p></div></div>
          <select id="var-select" class="space-select" onchange="selectVariable(this.value)" aria-label="Variable to investigate" style="margin-bottom:10px;"></select>
          <div id="guide" class="guide">
            <div id="guide-log" class="guide-log" aria-live="polite"></div>
            <form class="guide-form" onsubmit="guideAsk(event)">
              <input id="guide-input" type="text" autocomplete="off" placeholder="Type a question, e.g. how do I read the stripes?" aria-label="Ask Pulse Guide" />
              <button type="submit" aria-label="Send">➤</button>
            </form>
          </div>
          <div id="var-def" class="var-def hidden"></div>
        </div>
        <div class="step" id="step-where">
          <div class="step-head"><span class="step-num">2</span><div><h3>Where?</h3><p>Click anywhere on the map</p></div></div>
          <div class="where-box"><span>📍</span><span id="where-text">No place selected yet.<small>Click the map on the right →</small></span></div>
        </div>
        <div class="step" id="step-when" style="border-bottom:none;">
          <div class="step-head"><span class="step-num">3</span><div><h3>When?</h3><p>Pick the years on the timeline</p></div></div>
          <div class="timeline-slot" id="slot-investigate"></div>
        </div>
        <div class="go-wrap">
          <button id="investigate-btn" class="primary-glow-btn" onclick="runInvestigation()" disabled>🔍 Investigate</button>
          <div class="go-hint" id="go-hint">Choose a variable and a place to start.</div>
        </div>
      </div>

      <!-- Discover -->
      <div id="panel-discover" class="card glass-panel tab-panel hidden">
        <div class="subtab-row">
          <button class="subtab-btn active" data-mode="auto" onclick="switchDiscoverMode('auto')">Which variables are changing here?</button>
          <button class="subtab-btn" data-mode="compare" onclick="switchDiscoverMode('compare')">Compare two places</button>
        </div>
        <div id="disc-mode-auto">
          <p class="helper">Tests every variable at the selected place and ranks them by how strongly they are changing.</p>
          <div class="where-box" style="margin-bottom:14px;"><span>📍</span><span class="js-place-name">Click the map to choose a place.</span></div>
        </div>
        <div id="disc-mode-compare" class="hidden">
          <p class="helper">Run the same variable at two places and compare their trends.</p>
          <label class="mini-label" for="compare-var">Variable</label>
          <select id="compare-var" class="space-select" style="margin-bottom:12px;"></select>
          <div class="loc-box loc-a"><h4>PLACE A · MAP MARKER</h4><div class="loc-name js-place-name">Click the map to choose place A.</div></div>
          <div class="loc-box loc-b">
            <h4>PLACE B</h4>
            <div class="loc-name" id="loc-b-name">Not set yet. Pick it on the map or type coordinates.</div>
            <div class="loc-inputs">
              <input type="number" id="loc-b-lat" class="space-input" placeholder="Lat" step="any" min="-90" max="90" />
              <input type="number" id="loc-b-lon" class="space-input" placeholder="Lon" step="any" min="-180" max="180" />
              <button type="button" id="pick-b-btn" class="pick-btn" onclick="togglePickB()">📍 Pick on map</button>
            </div>
          </div>
        </div>
        <div class="timeline-slot" id="slot-discover"></div>
        <div class="go-wrap"><button id="discover-btn" class="primary-glow-btn" onclick="runDiscover()">🧭 Scan all variables</button></div>
      </div>

      <!-- Guided examples -->
      <div id="panel-examples" class="card glass-panel tab-panel hidden">
        <p class="helper">Eight real investigations, each pictured from space by NASA satellites. <b>Watch</b> one being solved, or <b>Try it</b> yourself.</p>
        <div id="example-list" class="card-grid"></div>
      </div>
    </aside>

    <!-- Map in the centre (shared by all tabs except Learn) -->
    <section id="map-card" class="map-card glass-panel">
      <div class="map-wrap">
        <div id="world-map"></div>
        <div class="map-overlay ov-place"><div class="chip" id="map-place">🗺️ Click anywhere on Earth</div></div>
        <div class="map-overlay ov-legend" id="map-legend-box"></div>
        <div class="map-overlay ov-layers" id="layer-buttons"></div>
        <div class="map-overlay ov-status hidden" id="map-status"><div class="chip" id="map-status-text"></div></div>
        <div class="map-overlay ov-tools">
          <button type="button" id="trendmap-btn" class="ghost-btn chip hidden" onclick="showTrendMapAround()">🟥🟦 Trend map around here</button>
          <button type="button" id="trendmap-clear" class="ghost-btn chip hidden" onclick="clearTrendMap()">✕ Hide trend map</button>
        </div>
        <div class="map-overlay ov-time" id="time-bar">
          <button type="button" class="play" id="time-play" onclick="toggleTimePlay()" aria-label="Play">▶</button>
          <div class="tlabel" id="time-label">Now<small>Live</small></div>
          <div class="tsteps" id="time-steps"></div>
        </div>
        <div id="map-legend" class="map-legend hidden" style="bottom:130px;"></div>
      </div>
    </section>

    <!-- Right column: quick charts -->
    <aside class="right-col" id="right-col">
      <div class="mini-card glass-panel">
        <h4><span>Annual cycle</span><span id="cycle-years"></span></h4>
        <div id="cycle-empty" class="helper" style="margin:0;">Investigate a variable to see its average for each month.</div>
        <div class="mini-chart hidden" id="cycle-wrap"><canvas id="cycleChart"></canvas></div>
      </div>
      <div class="mini-card glass-panel">
        <h4><span>Trend at a glance</span></h4>
        <div id="glance-empty" class="helper" style="margin:0;">Your trend will appear here after you press Investigate.</div>
        <div id="glance-body" class="hidden"></div>
        <div class="mini-chart hidden" id="glance-wrap" style="height:110px;"><canvas id="glanceChart"></canvas></div>
      </div>
    </aside>

    <!-- Learn (replaces the workspace) -->
    <section id="learn-view" class="learn-view glass-panel hidden" style="grid-column: 1 / -1;">
      <div class="learn-cols">
        <div class="learn-list" id="learn-methods"></div>
        <div class="learn-list" id="learn-variables"></div>
      </div>
    </section>
  </main>

  <!-- ================= RESULTS ================= -->
  <section id="results" class="results">
    <div id="loader" class="loader hidden"><div class="radar-spinner"></div><p id="loader-text" style="font-weight:600;">Downloading monthly data from NASA POWER…</p></div>

    <div id="empty-state" class="empty-state glass-panel">
      <div class="big">🗂️</div>
      <h3>Your case file will appear here</h3>
      <p>Complete the three steps and press <b>Investigate</b>, or open <b>Guided Examples</b> to watch one being solved.</p>
    </div>

    <div id="casefile" class="hidden"></div>
    <div id="discover-results" class="hidden"></div>

    <!-- Scientist mode -->
    <div id="science-panel" class="sci-panel glass-panel hidden">
      <div class="sci-top">
        <div><h2>🔬 Scientist Analysis</h2><p class="helper" style="margin:2px 0 0;">Full statistics for the current case: robust estimators, autocorrelation-corrected tests, diagnostics, forecasts and export.</p></div>
        <span class="sci-badge">SCIENTIST MODE</span>
      </div>
      <div class="sci-controls">
        <div><span class="mini-label">Spatial scope</span>
          <div class="seg" id="sci-scope">
            <button type="button" class="active" data-scope="point" onclick="setSciScope('point')">Point</button>
            <button type="button" data-scope="region" onclick="setSciScope('region')">Region box</button>
          </div>
        </div>
        <div><label class="mini-label" for="sci-agg">Time series</label>
          <select id="sci-agg" class="space-select">
            <option value="monthly">Monthly, deseasonalized</option>
            <option value="annual">Annual means</option>
            <option value="DJF">Winter (DJF) means</option>
            <option value="MAM">Spring (MAM) means</option>
            <option value="JJA">Summer (JJA) means</option>
            <option value="SON">Autumn (SON) means</option>
          </select>
        </div>
        <div><label class="mini-label" for="sci-alpha">Significance level α</label>
          <select id="sci-alpha" class="space-select">
            <option value="0.01">0.01 (strict)</option>
            <option value="0.05" selected>0.05 (standard)</option>
            <option value="0.10">0.10 (lenient)</option>
          </select>
        </div>
        <div id="sci-grid-wrap" class="hidden"><label class="mini-label" for="sci-grid">Trend-map grid</label>
          <select id="sci-grid" class="space-select"><option value="3">3 × 3</option><option value="4" selected>4 × 4</option><option value="5">5 × 5</option></select>
        </div>
        <button id="sci-run-btn" class="primary-glow-btn" style="padding:11px;" onclick="runScience()">Re-run analysis</button>
      </div>
      <div id="sci-region-bar" class="region-bar hidden">
        <button type="button" id="sci-draw-btn" class="ghost-btn" onclick="toggleRegionDraw()">▭ Draw region on map</button>
        <span id="sci-box-text">No region yet. Click two opposite corners on the map (max 10° × 10°).</span>
      </div>
      <div class="sci-tabs" id="sci-tabs">
        <button class="sci-tab active" data-st="stats" onclick="setSciTab('stats')">Statistics</button>
        <button class="sci-tab" data-st="seasons" onclick="setSciTab('seasons')">Seasons</button>
        <button class="sci-tab" data-st="diag" onclick="setSciTab('diag')">Diagnostics</button>
        <button class="sci-tab" data-st="forecast" onclick="setSciTab('forecast')">Forecast</button>
        <button class="sci-tab" data-st="unusual" onclick="setSciTab('unusual')">Unusual months</button>
        <button class="sci-tab" data-st="indicators" onclick="setSciTab('indicators')">Indicators</button>
        <button class="sci-tab" data-st="data" onclick="setSciTab('data')">Data & export</button>
      </div>
      <div id="sci-status" class="helper">Run an investigation to fill in the scientific analysis.</div>
      <div id="st-stats" class="sci-pane"></div>
      <div id="st-seasons" class="sci-pane hidden"></div>
      <div id="st-diag" class="sci-pane hidden"></div>
      <div id="st-forecast" class="sci-pane hidden"></div>
      <div id="st-unusual" class="sci-pane hidden"></div>
      <div id="st-indicators" class="sci-pane hidden"></div>
      <div id="st-data" class="sci-pane hidden"></div>
    </div>
  </section>

  </div><!-- /#app -->

  <!-- Shared timeline (moved into the active panel's slot) -->
  <div id="timeline" class="timeline">
    <div class="tl-range"><strong id="tl-text">1981 – 2025</strong><small id="tl-sub">45 years</small></div>
    <div class="tl-slider">
      <div class="tl-track"></div>
      <div class="tl-fill" id="tl-fill"></div>
      <input type="range" id="tl-start" class="tl-start" min="1981" max="2025" step="1" value="1981" aria-label="Start year" />
      <input type="range" id="tl-end" class="tl-end" min="1981" max="2025" step="1" value="2025" aria-label="End year" />
    </div>
    <div class="tl-scale" id="tl-scale"></div>
    <div class="tl-chips">
      <button type="button" class="tl-chip" onclick="setLastYears(10)">Last 10 yrs</button>
      <button type="button" class="tl-chip" onclick="setLastYears(20)">Last 20 yrs</button>
      <button type="button" class="tl-chip" onclick="setLastYears(30)">Last 30 yrs</button>
      <button type="button" class="tl-chip" onclick="setLastYears(0)">Full record</button>
    </div>
    <div class="tl-note" id="tl-note">NASA POWER monthly data: 1981 to the last complete year.</div>
  </div>

  <!-- Demo caption bar -->
  <div id="demo-bar" class="demo-bar glass-panel hidden" role="status" aria-live="polite">
    <div class="demo-top"><span class="kicker" id="demo-step">Demo · step 1</span><span class="kicker" id="demo-title"></span></div>
    <div class="demo-text" id="demo-text"></div>
    <div class="demo-progress"><span id="demo-progress"></span></div>
    <div class="demo-controls">
      <button type="button" class="ghost-btn" id="demo-pause" onclick="demoTogglePause()">⏸ Pause</button>
      <button type="button" class="ghost-btn" onclick="demoNext()">Next ›</button>
      <button type="button" class="ghost-btn" onclick="demoStop()">✕ Exit demo</button>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/globe.gl@2.46.2/dist/globe.gl.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <script>
    /* =====================================================================
       Planet Pulse front end
       Flow: welcome (choose mode) → Investigate (what / where / when) →
       case file (visual answers) → Scientist analysis (Scientist mode only)
       All data comes live from the NASA POWER monthly API via this server.
       ===================================================================== */
    let map = null, marker = null, trendMarker = null;
    let variablesCatalog = [], audienceMode = 'simple', activeTab = 'investigate';
    let selectedVar = null, selectedLat = null, selectedLon = null, currentPlace = null, placeRequestId = 0;
    let yearStart = 1981, yearEnd = new Date().getFullYear() - 1, dataMaxYear = new Date().getFullYear() - 1, minSpanYears = 3;
    let lastAnalysis = null, trendMapLayer = null, trendMapBusy = false;
    const charts = {};

    const VAR_LOOK = {
      T2M: { icon: '🌡️', short: 'Temperature' },
      T2M_MAX: { icon: '🔥', short: 'Max temperature' },
      T2M_MIN: { icon: '🌙', short: 'Min temperature' },
      PRECTOTCORR: { icon: '🌧️', short: 'Rain & snow' },
      SNODP: { icon: '❄️', short: 'Snow depth' },
      RH2M: { icon: '💧', short: 'Relative humidity' },
      QV2M: { icon: '🌫️', short: 'Water vapour' },
      ALLSKY_SFC_SW_DWN: { icon: '☀️', short: 'Sunlight' },
      CLOUD_AMT: { icon: '☁️', short: 'Cloud cover' },
      GWETPROF: { icon: '🌱', short: 'Soil moisture' },
      WS10M: { icon: '💨', short: 'Wind speed' }
    };

    /* ---------------- helpers ---------------- */
    function escHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
    function formatCoords(lat, lon) { return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`; }
    function varById(id) { return variablesCatalog.find(v => v.id === id) || null; }
    function unitOf(id) { return (varById(id) || {}).unit || ''; }
    function shortUnit(u) { return /^0 to 1/.test(u) ? 'wetness index' : u; }
    function varShort(id) { return (VAR_LOOK[id] || {}).short || (varById(id) || {}).name || id; }
    function signed(n, d) { const x = Number(n).toFixed(d); return (n > 0 ? '+' : '') + x; }
    function fmtP(p) { return p === null || p === undefined ? '–' : (p < 0.0001 ? '< 0.0001' : Number(p).toFixed(4)); }
    function fmtN(v, d) { return v === null || v === undefined ? '–' : Number(v).toFixed(d === undefined ? 3 : d); }
    function niceDecimals(x) { const a = Math.abs(x); return a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.1 ? 2 : 3; }
    function wrapLon(lon) { return ((lon + 180) % 360 + 360) % 360 - 180; }
    function destroyChart(name) { if (charts[name]) { charts[name].destroy(); delete charts[name]; } }
    const AXIS = { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.08)' } };
    const LEGEND = { labels: { color: '#cbd5e1', font: { size: 11, family: 'Inter' }, boxWidth: 14 } };

    /* ---------------- welcome & mode ---------------- */
    function chooseMode(mode, withDemo) {
      document.getElementById('landing').classList.add('hidden');
      document.getElementById('app').classList.remove('hidden');
      document.body.classList.remove('locked');
      pauseGlobe(true);
      setAudienceMode(mode);
      if (map) { map.invalidateSize(); onMapViewChanged(); }
      window.scrollTo({ top: 0 });
      if (withDemo) { switchTab('examples'); setTimeout(() => startDemo(0), 500); }
    }

    function showLanding() {
      demoStop();
      document.getElementById('app').classList.add('hidden');
      document.getElementById('landing').classList.remove('hidden');
      document.body.classList.add('locked');
      pauseGlobe(false);
    }

    function setAudienceMode(mode) {
      audienceMode = mode;
      document.getElementById('btn-simple').classList.toggle('active', mode === 'simple');
      document.getElementById('btn-science').classList.toggle('active', mode === 'science');
      if (selectedVar) showVarDefinition();
      if (lastAnalysis) renderCaseFile(lastAnalysis);
      if (lastDiscover) renderDiscoverResults(lastDiscover);
      renderLearn();
      applyScientistLayout();
    }

    /* ---------------- tabs ---------------- */
    const TAB_INTRO = {
      investigate: { kicker: 'Investigate', title: 'Three steps: <span class="accent">what, where, when.</span>',
        text: 'Choose a variable, click a place on the map and pick the years. Then press Investigate to see how it has changed and whether the change is real.' },
      discover: { kicker: 'Discover', title: 'Let the data <span class="accent">find the trends.</span>',
        text: 'Not sure what to look for? Scan every variable at one place to see which are changing most, or compare the same variable at two places.' },
      examples: { kicker: 'Guided examples', title: 'Watch a case <span class="accent">being solved.</span>',
        text: 'Each example starts with a real question. Press Watch to see the detective work step by step, or Try it to explore it yourself.' },
      learn: { kicker: 'Learn', title: 'Understand <span class="accent">the science.</span>',
        text: 'Short explanations of every method and variable. Switch to Scientist mode for equations and references.' }
    };

    function switchTab(tab) {
      activeTab = tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
      ['investigate', 'discover', 'examples'].forEach(t => document.getElementById(`panel-${t}`).classList.toggle('hidden', t !== tab));
      const learning = tab === 'learn';
      document.querySelector('.side').classList.toggle('hidden', learning);
      document.getElementById('map-card').classList.toggle('hidden', learning);
      document.getElementById('right-col').classList.toggle('hidden', learning);
      document.getElementById('learn-view').classList.toggle('hidden', !learning);
      document.getElementById('results').classList.toggle('hidden', learning);
      if (tab === 'investigate') moveTimeline('slot-investigate');
      if (tab === 'discover') moveTimeline('slot-discover');
      if (learning) renderLearn();
      if (!learning && map) setTimeout(() => { map.invalidateSize(); onMapViewChanged(); }, 50);
      updateResultsVisibility();
      updateMarkerB();
      if (tab !== 'discover' && pickingB) togglePickB();
      const info = TAB_INTRO[tab], box = document.getElementById('tab-intro');
      document.getElementById('tab-intro-kicker').textContent = info.kicker;
      document.getElementById('tab-intro-title').innerHTML = info.title;
      document.getElementById('tab-intro-text').textContent = info.text;
      box.classList.remove('swap'); void box.offsetWidth; box.classList.add('swap');
    }

    function updateResultsVisibility() {
      const disc = activeTab === 'discover';
      const hasCase = !!lastAnalysis;
      document.getElementById('discover-results').classList.toggle('hidden', !disc || !lastDiscoverHtml());
      document.getElementById('casefile').classList.toggle('hidden', disc || !hasCase);
      document.getElementById('empty-state').classList.toggle('hidden', hasCase || (disc && !!lastDiscoverHtml()));
      document.getElementById('science-panel').classList.toggle('hidden', audienceMode !== 'science' || disc);
    }

    function moveTimeline(slotId) {
      const slot = document.getElementById(slotId), tl = document.getElementById('timeline');
      if (slot && tl.parentElement !== slot) slot.appendChild(tl);
    }

    /* ---------------- map ---------------- */
    function initMap() {
      map = L.map('world-map', { zoomControl: false, worldCopyJump: true, minZoom: 2 }).setView([20, 0], 2);
      L.control.zoom({ position: 'bottomright' }).addTo(map);
      // Standard OpenStreetMap basemap (no key needed)
      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors', maxZoom: 18 }).addTo(map);
      initLiveLayers();
      map.on('click', (e) => {
        if (drawingRegion) { handleRegionClick(e.latlng.lat, e.latlng.lng); return; }
        if (pickingB) { setLocationB(e.latlng.lat, e.latlng.lng); return; }
        setPoint(e.latlng.lat, e.latlng.lng);
      });
      setTimeout(() => map.invalidateSize(), 250);
    }

    function setPoint(lat, lon, fly) {
      selectedLat = lat; selectedLon = wrapLon(lon);
      if (!marker) marker = L.marker([selectedLat, selectedLon]).addTo(map);
      else marker.setLatLng([selectedLat, selectedLon]);
      if (fly) map.flyTo([selectedLat, selectedLon], 4, { duration: 1.2 });
      if (trendMarker) { map.removeLayer(trendMarker); trendMarker = null; }
      clearTrendMap();
      document.getElementById('trendmap-btn').classList.add('hidden');
      currentPlace = null;
      refreshPlaceMentions(true);
      updateSteps();
      return updatePlaceName(selectedLat, selectedLon);
    }

    async function updatePlaceName(lat, lon) {
      const myId = ++placeRequestId;
      try {
        const res = await fetch(`/api/place?lat=${lat}&lon=${lon}`);
        const place = await res.json();
        if (myId !== placeRequestId) return;
        if (!res.ok) throw new Error(place.error || 'lookup failed');
        currentPlace = place;
        if (marker) marker.bindPopup(`<strong>${escHtml(place.label)}</strong><br/>${escHtml(place.coords)}`);
      } catch (err) {
        if (myId !== placeRequestId) return;
        currentPlace = null;
      }
      refreshPlaceMentions(false);
    }

    function placeLabel() { return currentPlace ? currentPlace.label : (selectedLat === null ? '' : formatCoords(selectedLat, selectedLon)); }

    function refreshPlaceMentions(locating) {
      if (selectedLat === null) return;
      const name = locating ? 'Locating…' : placeLabel(), coords = formatCoords(selectedLat, selectedLon);
      document.getElementById('where-text').innerHTML = `${escHtml(name)}<small>${coords}</small>`;
      document.getElementById('map-place').innerHTML = `📍 ${escHtml(name)}<small>${coords}</small>`;
      document.querySelectorAll('.js-place-name').forEach(el => { el.textContent = locating ? 'Locating…' : `${name} (${coords})`; });
    }

    /* ---------------- step 1: variables ---------------- */
    async function loadVariables() {
      const res = await fetch('/api/variables');
      const data = await res.json();
      variablesCatalog = data.variables;
      if (data.max_year) dataMaxYear = data.max_year;
      if (data.min_span_years) minSpanYears = data.min_span_years;
      document.getElementById('var-select').innerHTML = variablesCatalog.map(v =>
        `<option value="${v.id}">${(VAR_LOOK[v.id] || {}).icon || ''} ${escHtml(varShort(v.id))}</option>`).join('');
      document.getElementById('compare-var').innerHTML = variablesCatalog.map(v => `<option value="${v.id}">${escHtml(v.name)} (${escHtml(v.unit)})</option>`).join('');
      setYears(1981, dataMaxYear);
      renderExamples();
      renderLearn();
      selectVariable('T2M');   // step 1 is optional: Temperature by default
      guideStart();
    }

    function selectVariable(id) {
      selectedVar = id;
      const vs = document.getElementById('var-select'); if (vs && vs.value !== id) vs.value = id;
      document.getElementById('compare-var').value = id;
      showVarDefinition();
      refreshTimeline();
      updateSteps();
      const layer = LAYER_FOR_VAR[id];
      if (layer && layer !== activeLayer) setActiveLayer(layer);
    }

    function showVarDefinition() {
      const v = varById(selectedVar), box = document.getElementById('var-def');
      if (!v) { box.classList.add('hidden'); return; }
      box.classList.remove('hidden');
      box.innerHTML = `<b>${escHtml(v.name)}</b>: ${escHtml(audienceMode === 'science' ? v.audience_science : v.audience_simple)}` +
        `<small>Unit: ${escHtml(v.unit)} · ${escHtml(v.agency)}</small>`;
    }

    function updateSteps() {
      if (!selectedVar) selectedVar = 'T2M';
      const where = selectedLat !== null;
      document.getElementById('step-what').classList.toggle('done', true);
      document.getElementById('step-where').classList.toggle('done', where);
      document.getElementById('step-when').classList.toggle('done', true);
      const btn = document.getElementById('investigate-btn');
      btn.disabled = !where;
      document.getElementById('go-hint').textContent = where
        ? `Ready! Investigating ${varShort(selectedVar).toLowerCase()} with live NASA POWER data.`
        : 'Click a place on the map to start.';
    }

    /* ---------------- step 3: timeline ---------------- */
    function firstYearFor(id) { const v = varById(id); return v && v.first_year ? v.first_year : 1981; }
    function timelineBounds() { return { min: firstYearFor(selectedVar), max: dataMaxYear }; }

    function setYears(s, e) {
      const b = timelineBounds();
      s = Math.round(Number(s)); e = Math.round(Number(e));
      if (!isFinite(s)) s = b.min;
      if (!isFinite(e)) e = b.max;
      s = Math.max(b.min, Math.min(s, b.max - minSpanYears + 1));
      e = Math.min(b.max, Math.max(e, s + minSpanYears - 1));
      yearStart = s; yearEnd = e;
      const ts = document.getElementById('tl-start'), te = document.getElementById('tl-end');
      [ts, te].forEach(el => { el.min = b.min; el.max = b.max; });
      ts.value = s; te.value = e;
      const span = b.max - b.min || 1, fill = document.getElementById('tl-fill');
      fill.style.left = `${100 * (s - b.min) / span}%`;
      fill.style.width = `${100 * (e - s) / span}%`;
      document.getElementById('tl-text').textContent = `${s} – ${e}`;
      document.getElementById('tl-sub').textContent = `${e - s + 1} years`;
      const v = varById(selectedVar);
      document.getElementById('tl-note').textContent = `NASA POWER monthly data${v ? ` for ${varShort(v.id).toLowerCase()}` : ''}: ${b.min}–${b.max}. Longer periods (20+ years) reveal trends more reliably.`;
      const marks = new Set([b.min, b.max]);
      for (let y = Math.ceil(b.min / 10) * 10; y <= b.max; y += 10) if (y - b.min >= 4 && b.max - y >= 4) marks.add(y);
      document.getElementById('tl-scale').innerHTML = [...marks].sort((a, c) => a - c).map(y => `<span style="left:${100 * (y - b.min) / span}%">${y}</span>`).join('');
    }
    function refreshTimeline() { setYears(yearStart, yearEnd); }
    function setLastYears(n) { const b = timelineBounds(); setYears(n ? b.max - n + 1 : b.min, b.max); }

    /* =====================================================================
       INVESTIGATE → CASE FILE
       ===================================================================== */
    function setLoading(on, text) {
      document.getElementById('loader').classList.toggle('hidden', !on);
      if (text) document.getElementById('loader-text').textContent = text;
      if (on) document.getElementById('empty-state').classList.add('hidden');
    }

    async function runInvestigation() {
      if (!selectedVar || selectedLat === null) return false;
      const btn = document.getElementById('investigate-btn');
      btn.disabled = true;
      setLoading(true, 'Downloading monthly data from NASA POWER and analysing it…');
      document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
      try {
        const res = await fetch('/api/analyze', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: selectedLat, lon: selectedLon, parameter: selectedVar, start_year: yearStart, end_year: yearEnd })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'The analysis failed.');
        lastAnalysis = data;
        lastScience = null;
        renderCaseFile(data);
        setTrendMarker(data);
        document.getElementById('trendmap-btn').classList.remove('hidden');
        updateResultsVisibility();
        if (audienceMode === 'science') { renderScientistExtras(); runScience(); }
        return true;
      } catch (err) {
        lastAnalysis = null;
        const cf = document.getElementById('casefile');
        cf.innerHTML = `<div class="err-box">⚠️ ${escHtml(err.message)}<br/>No numbers are shown because Planet Pulse only uses real NASA data. Please try again in a moment.</div>`;
        cf.classList.remove('hidden');
        document.getElementById('empty-state').classList.add('hidden');
        return false;
      } finally {
        setLoading(false);
        updateSteps();
      }
    }

    const EVIDENCE = {
      strong: { level: 3, word: 'Strong evidence', text: 'Very unlikely to be chance: less than a 1 in 100 probability that natural ups and downs alone would produce a trend this clear.' },
      moderate: { level: 2, word: 'Moderate evidence', text: 'Unlikely to be chance: less than a 1 in 20 probability that natural ups and downs alone would produce this trend.' },
      weak: { level: 1, word: 'Weak evidence', text: 'A hint of a trend, but it could still be chance (between 1 in 20 and 1 in 10). A longer period may settle it.' },
      none: { level: 0, word: 'No clear evidence', text: 'We cannot rule out chance: the change is small compared with normal year-to-year ups and downs. A longer period might reveal a trend.' }
    };

    function renderCaseFile(d) {
      const pub = d.public, v = varById(d.parameter) || { name: d.parameter, unit: '' };
      const u = shortUnit(v.unit), place = d.place ? d.place.label : formatCoords(d.location.lat, d.location.lon);
      const first = d.dates[0].slice(0, 4), last = d.dates[d.dates.length - 1].slice(0, 4);
      const ev = EVIDENCE[pub.evidence], dec = niceDecimals(pub.total_change);
      const rose = pub.direction === 'up' ? 'risen' : pub.direction === 'down' ? 'fallen' : 'stayed level';
      const arrow = pub.direction === 'up' ? '↑' : pub.direction === 'down' ? '↓' : '→';
      const science = audienceMode === 'science';
      const anomalies = (d.ml && d.ml.anomalies) || [];

      let html = `<div class="cf-head"><div><div class="kicker">Case file</div><h2>${escHtml(v.name)} · ${escHtml(place)}</h2></div>` +
        `<div class="kicker">${first}–${last} · NASA POWER monthly data</div></div><div class="cf-grid">`;

      // 1. How much?
      html += `<div id="cf-headline" class="cf-card glass-panel span-7"><div class="cf-q">What changed & how much?</div>` +
        `<div class="headline-num"><div class="headline-arrow ${pub.direction}">${arrow}</div>` +
        `<div class="headline-value">${signed(pub.total_change, dec)} ${escHtml(u)}<small>since ${first} · ${signed(pub.sen_per_decade, niceDecimals(pub.sen_per_decade))} ${escHtml(u)} per decade</small></div></div>` +
        `<p class="headline-text">${escHtml(varShort(d.parameter))} at <b>${escHtml(place)}</b> has <b>${rose}</b> by about <b>${Math.abs(pub.total_change).toFixed(dec)} ${escHtml(u)}</b> over ${pub.span_years} years, once the normal seasonal cycle is set aside.` +
        `${pub.evidence === 'none' ? ' This could still be natural variability.' : ''}</p>` +
        `<p class="headline-sub">Typical value over the period: ${fmtN(pub.period_mean, niceDecimals(pub.period_mean))} ${escHtml(u)}.${science ? ` Sen's slope 95% CI: ${fmtN(pub.sen_ci_low, 4)} to ${fmtN(pub.sen_ci_high, 4)} ${escHtml(u)}/decade.` : ''}</p></div>`;

      // 2. Is it real?
      html += `<div id="cf-confidence" class="cf-card glass-panel span-5"><div class="cf-q">Is the change real?</div>` +
        `<div class="evidence-word" style="color:${['#94a3b8', '#fbbf24', '#34d399', '#10b981'][ev.level]}">${ev.word}</div>` +
        `<div class="meter"><div class="meter-track">${[0, 1, 2, 3].map(i => `<div class="meter-seg l${i} ${i <= ev.level ? 'on' : ''}"></div>`).join('')}</div>` +
        `<div class="meter-labels">${['None', 'Weak', 'Moderate', 'Strong'].map((w, i) => `<span class="${i === ev.level ? 'cur' : ''}">${w}</span>`).join('')}</div></div>` +
        `<p class="headline-text" style="font-size:0.88rem;">${ev.text}</p>` +
        (science ? `<p class="headline-sub">${escHtml(pub.test)}: p = ${fmtP(pub.p_value)}.</p>` : '') + `</div>`;

      // 3. Trend over time
      html += `<div id="cf-trend" class="cf-card glass-panel span-12"><div class="cf-q">How has it changed over time?</div>` +
        `<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap;">` +
        `<p class="helper" style="max-width:720px;">Each dot is one year's average. The dashed line is the trend and the shaded band shows how certain we are about it. Switch to monthly to see every month and the most unusual ones (red).</p>` +
        `<div class="seg seg-mini"><button type="button" class="active" id="tv-year" onclick="setTrendView('year')">Yearly</button><button type="button" id="tv-month" onclick="setTrendView('month')">Monthly</button></div></div>` +
        `<div class="chart-box"><canvas id="trendChart"></canvas></div></div>`;

      // 4. Climate stripes
      const st = pub.yearly.stripes;
      if (st && st.length) {
        const maxAbs = Math.max(1e-9, ...st.map(x => Math.abs(x)));
        html += `<div id="cf-stripes" class="cf-card glass-panel span-12"><div class="cf-q">The whole story at a glance</div>` +
          `<p class="helper">Climate stripes: one stripe per year, coloured by how far that year was from the ${first}–${last} average. Hover a stripe for its value.</p>` +
          `<div class="stripes">${st.map((x, i) => `<div style="background:${stripeColor(x / maxAbs)}" data-tip="${pub.yearly.years[i]}: ${signed(x, niceDecimals(maxAbs))} ${escHtml(u)} vs average" onmouseenter="stripeTip(this)"></div>`).join('')}</div>` +
          `<div class="stripe-axis"><span>${pub.yearly.years[0]}</span><span>${pub.yearly.years[pub.yearly.years.length - 1]}</span></div>` +
          `<div class="stripe-legend"><span>Below average</span><span class="sw"></span><span>Above average</span><span id="stripe-tip" class="stripe-tip" style="margin-left:auto;"></span></div></div>`;
      }

      // 5. Which months changed?
      if (pub.month_compare) {
        html += `<div id="cf-months" class="cf-card glass-panel span-7"><div class="cf-q">Which months changed?</div>` +
          `<p class="helper">Average for each month in the first years (${pub.month_compare.early_label}) and the most recent years (${pub.month_compare.late_label}).</p>` +
          `<div class="chart-box short"><canvas id="monthChart"></canvas></div></div>`;
      }

      // 6. Unusual months + where
      html += `<div id="cf-where" class="cf-card glass-panel span-5"><div class="cf-q">Where & what stood out?</div>` +
        `<p class="helper">The marker on the map is now coloured by this trend (red rising, blue falling). Does the area around it change the same way?</p>` +
        `<button type="button" class="ghost-btn" style="width:100%;margin-bottom:14px;" onclick="showTrendMapAround()">🟥🟦 Show trend map around here</button>` +
        `<div class="mini-label">Most unusual months</div>` +
        (anomalies.length ? `<div class="unusual-list">${anomalies.slice(0, 3).map(a =>
          `<div class="unusual-item"><b>${escHtml(a.date)}</b><span>${a.value} ${escHtml(u)} vs a normal ${a.expected} ${escHtml(u)} (${a.z_score > 0 ? 'much higher' : 'much lower'} than usual)</span></div>`).join('')}</div>`
          : '<p class="helper">No month stood out as extremely unusual for its time of year.</p>') + `</div>`;

      html += '</div>';
      const cf = document.getElementById('casefile');
      cf.innerHTML = html;
      updateResultsVisibility();
      drawTrendChart('year');
      drawMonthChart();
      renderRightColumn(d);
    }

    function stripeColor(f) {
      f = Math.max(-1, Math.min(1, f));
      const stops = [[-1, [8, 48, 107]], [-0.5, [107, 174, 214]], [0, [247, 247, 247]], [0.5, [251, 106, 74]], [1, [103, 0, 13]]];
      for (let i = 0; i < stops.length - 1; i++) {
        const [a, ca] = stops[i], [b, cb] = stops[i + 1];
        if (f >= a && f <= b) { const k = (f - a) / (b - a); return `rgb(${ca.map((c, j) => Math.round(c + (cb[j] - c) * k)).join(',')})`; }
      }
      return 'rgb(247,247,247)';
    }
    function stripeTip(el) { document.getElementById('stripe-tip').textContent = el.dataset.tip; }

    let trendView = 'year';
    function setTrendView(view) {
      trendView = view;
      document.getElementById('tv-year').classList.toggle('active', view === 'year');
      document.getElementById('tv-month').classList.toggle('active', view === 'month');
      drawTrendChart(view);
    }

    function drawTrendChart(view) {
      const d = lastAnalysis; if (!d || typeof Chart === 'undefined') return;
      destroyChart('trend');
      const u = shortUnit(unitOf(d.parameter)), y = d.public.yearly;
      let cfg;
      if (view === 'year') {
        cfg = { type: 'line', data: { labels: y.years, datasets: [
          { label: 'Uncertainty (95%)', data: y.band_low || [], borderWidth: 0, pointRadius: 0, fill: false },
          { label: 'Uncertainty band', data: y.band_high || [], borderWidth: 0, pointRadius: 0, backgroundColor: 'rgba(192, 132, 252, 0.18)', fill: '-1' },
          { label: `Yearly average (${u})`, data: y.values, borderColor: '#38bdf8', backgroundColor: '#38bdf8', borderWidth: 1.5, pointRadius: 4, pointHoverRadius: 6, tension: 0.2 },
          { label: 'Trend', data: y.fit || [], borderColor: '#fb7185', borderDash: [7, 5], borderWidth: 2.5, pointRadius: 0 }
        ] } };
        cfg.options = { plugins: { legend: { ...LEGEND, labels: { ...LEGEND.labels, filter: it => it.text !== 'Uncertainty (95%)' } } } };
      } else {
        const idx = new Map(d.dates.map((dt, i) => [dt, i]));
        const unusual = d.dates.map(() => null);
        (d.ml.anomalies || []).forEach(a => { const i = idx.get(a.date); if (i !== undefined) unusual[i] = a.value; });
        cfg = { type: 'line', data: { labels: d.dates, datasets: [
          { label: `Monthly value (${u})`, data: d.values, borderColor: 'rgba(56, 189, 248, 0.8)', backgroundColor: 'rgba(56, 189, 248, 0.08)', borderWidth: 1.2, pointRadius: 0, fill: true, tension: 0.2 },
          { label: 'Unusual month', data: unusual, borderColor: '#fb7185', backgroundColor: '#fb7185', pointRadius: 6, showLine: false }
        ] } };
        cfg.options = { plugins: { legend: LEGEND } };
      }
      cfg.options = Object.assign({ responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        scales: { x: { ...AXIS, ticks: { ...AXIS.ticks, maxTicksLimit: 14 } }, y: AXIS } }, cfg.options);
      charts.trend = new Chart(document.getElementById('trendChart').getContext('2d'), cfg);
    }

    function drawMonthChart() {
      const d = lastAnalysis; if (!d || !d.public.month_compare || typeof Chart === 'undefined') return;
      destroyChart('months');
      const m = d.public.month_compare, u = shortUnit(unitOf(d.parameter));
      charts.months = new Chart(document.getElementById('monthChart').getContext('2d'), {
        type: 'line',
        data: { labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], datasets: [
          { label: `${m.early_label}`, data: m.early, borderColor: '#60a5fa', backgroundColor: '#60a5fa', borderWidth: 2, pointRadius: 3, tension: 0.3 },
          { label: `${m.late_label}`, data: m.late, borderColor: '#f87171', backgroundColor: '#f87171', borderWidth: 2.5, pointRadius: 3, tension: 0.3 }
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
          plugins: { legend: LEGEND, tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y} ${u}` } } }, scales: { x: AXIS, y: AXIS } }
      });
    }

    function setTrendMarker(d) {
      if (!map) return;
      if (trendMarker) map.removeLayer(trendMarker);
      const pub = d.public, sig = pub.evidence === 'strong' || pub.evidence === 'moderate';
      const color = pub.direction === 'up' ? '#ef4444' : pub.direction === 'down' ? '#3b82f6' : '#94a3b8';
      trendMarker = L.circleMarker([d.location.lat, d.location.lon], { radius: 16, color: sig ? '#f8fafc' : color, weight: sig ? 3 : 1.5, fillColor: color, fillOpacity: 0.55 });
      trendMarker.bindPopup(`<strong>${escHtml(varShort(d.parameter))}: ${signed(pub.sen_per_decade, niceDecimals(pub.sen_per_decade))} ${escHtml(shortUnit(unitOf(d.parameter)))}/decade</strong><br/>${EVIDENCE[pub.evidence].word}`);
      trendMarker.addTo(map);
    }

    /* ---------------- trend map around the point ---------------- */
    async function showTrendMapAround() {
      if (!lastAnalysis || trendMapBusy) return;
      const lat = lastAnalysis.location.lat, lon = lastAnalysis.location.lon;
      const south = Math.max(-89.5, lat - 2), north = Math.min(89.5, lat + 2);
      const west = Math.max(-180, lon - 2.5), east = Math.min(180, lon + 2.5);
      trendMapBusy = true;
      const legend = document.getElementById('map-legend');
      legend.classList.remove('hidden');
      legend.innerHTML = '⏳ Downloading 16 nearby points from NASA POWER… (about 10–40 seconds)';
      document.getElementById('map-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
      try {
        const res = await fetch('/api/science/region', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ parameter: lastAnalysis.parameter, start_year: yearStart, end_year: yearEnd, alpha: 0.05, aggregation: 'monthly', grid: 4, box: { south, north, west, east } }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'failed');
        drawTrendCells(data, 'public');
        map.flyToBounds ? map.flyToBounds([[south, west], [north, east]], { duration: 1 }) : null;
      } catch (err) {
        legend.innerHTML = `⚠️ Trend map unavailable: ${escHtml(err.message)}`;
      } finally {
        trendMapBusy = false;
      }
    }

    function trendColor(v, maxAbs) {
      const f = Math.max(-1, Math.min(1, maxAbs > 0 ? v / maxAbs : 0));
      const mix = (a, b, k) => Math.round(a + (b - a) * k);
      const neutral = [226, 232, 240], red = [220, 38, 38], blue = [37, 99, 235];
      const end = f >= 0 ? red : blue, k = Math.abs(f);
      return `rgb(${mix(neutral[0], end[0], k)}, ${mix(neutral[1], end[1], k)}, ${mix(neutral[2], end[2], k)})`;
    }

    function drawTrendCells(data, who) {
      clearTrendMap(true);
      trendMapLayer = L.layerGroup();
      const ok = data.cells.filter(c => c.status === 'ok');
      const maxAbs = Math.max(1e-9, ...ok.map(c => Math.abs(c.per_decade)));
      const u = shortUnit(unitOf(data.provenance.parameter));
      data.cells.forEach(c => {
        const good = c.status === 'ok';
        const rect = L.rectangle([[c.south, c.west], [c.north, c.east]], {
          color: good && c.significant ? '#f8fafc' : '#64748b', weight: good && c.significant ? 2.5 : 1,
          dashArray: good && c.significant ? null : '4 3', fillColor: good ? trendColor(c.per_decade, maxAbs) : '#334155', fillOpacity: good ? 0.5 : 0.15
        });
        rect.bindPopup(good ? `<strong>${signed(c.per_decade, 3)} ${escHtml(u)}/decade</strong><br/>${c.significant ? 'Significant' : 'Not significant'} trend<br/>${formatCoords(c.lat, c.lon)}`
          + (audienceMode === 'science' ? `<br/>OLS p = ${c.p_ols} · MK (Hamed-Rao) p = ${c.p_mk_hr}` : '') : `No data<br/>${formatCoords(c.lat, c.lon)}`);
        trendMapLayer.addLayer(rect);
      });
      trendMapLayer.addTo(map);
      const up = ok.filter(c => c.significant && c.per_decade > 0).length, down = ok.filter(c => c.significant && c.per_decade < 0).length;
      const legend = document.getElementById('map-legend');
      legend.classList.remove('hidden');
      legend.innerHTML = `<b>Trend of ${escHtml(varShort(data.provenance.parameter).toLowerCase())} per decade</b><div class="legend-bar"></div>` +
        `<div class="legend-labels"><span>${signed(-maxAbs, 2)}</span><span>0</span><span>${signed(maxAbs, 2)} ${escHtml(u)}</span></div>` +
        `<div style="margin-top:6px;color:var(--text-muted);">White outline = significant. ${up} rising, ${down} falling significantly.` +
        (up && down ? ' <b style="color:#fde68a;">Opposite trends in one area!</b>' : '') + `</div>`;
      document.getElementById('trendmap-clear').classList.remove('hidden');
    }

    function clearTrendMap(keepLegend) {
      if (trendMapLayer && map) map.removeLayer(trendMapLayer);
      trendMapLayer = null;
      if (!keepLegend) {
        document.getElementById('map-legend').classList.add('hidden');
        document.getElementById('trendmap-clear').classList.add('hidden');
      }
    }

    /* =====================================================================
       LANDING: rotating 3D Earth at night (NASA Black Marble)
       ===================================================================== */
    let globe = null;
    const NASA_NIGHT_TEXTURE = 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1' +
      '&LAYERS=VIIRS_Black_Marble&STYLES=&SRS=EPSG:4326&BBOX=-180,-90,180,90&WIDTH=4096&HEIGHT=2048&FORMAT=image/jpeg';
    const FALLBACK_NIGHT_TEXTURE = 'https://unpkg.com/three-globe/example/img/earth-night.jpg';
    const NIGHT_SKY = 'https://unpkg.com/three-globe/example/img/night-sky.png';

    function probeImage(url, timeoutMs) {
      return new Promise(resolve => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        const t = setTimeout(() => resolve(false), timeoutMs || 12000);
        img.onload = () => { clearTimeout(t); resolve(img.naturalWidth > 0); };
        img.onerror = () => { clearTimeout(t); resolve(false); };
        img.src = url;
      });
    }

    function showGlobeFallback() {
      document.getElementById('globe-fallback').classList.remove('hidden');
    }

    function initGlobe() {
      const el = document.getElementById('globe');
      if (typeof Globe === 'undefined') { showGlobeFallback(); return; }
      try {
        const offset = window.innerWidth > 900 ? window.innerWidth * 0.22 : 0;
        el.style.left = `${offset}px`;
        globe = Globe()(el)
          .width(window.innerWidth).height(window.innerHeight)
          .backgroundColor('rgba(0,0,0,0)')
          .backgroundImageUrl(NIGHT_SKY)
          .showAtmosphere(true).atmosphereColor('#4cc9ff').atmosphereAltitude(0.24);
        probeImage(NASA_NIGHT_TEXTURE).then(ok => globe.globeImageUrl(ok ? NASA_NIGHT_TEXTURE : FALLBACK_NIGHT_TEXTURE));
        const c = globe.controls();
        c.autoRotate = true; c.autoRotateSpeed = 0.35; c.enableZoom = false;
        globe.pointOfView({ lat: 28, lng: -20, altitude: 1.9 });
        window.addEventListener('resize', () => { if (globe) globe.width(window.innerWidth).height(window.innerHeight); });
      } catch (e) {
        console.warn('3D globe unavailable', e);
        showGlobeFallback();
      }
    }

    function pauseGlobe(paused) {
      if (!globe) return;
      try { paused ? globe.pauseAnimation() : globe.resumeAnimation(); } catch (e) {}
    }

    /* =====================================================================
       LIVE MAP LAYERS
       - Temperature & wind: live hourly model data (Open-Meteo), drawn as a
         smooth colour field and animated wind particles
       - Precipitation, air quality, night lights: NASA GIBS satellite imagery
       ===================================================================== */
    const TEMP_STOPS = [[-40, [76, 29, 149]], [-25, [37, 99, 235]], [-10, [6, 182, 212]], [0, [167, 243, 208]], [10, [34, 197, 94]], [20, [250, 204, 21]], [30, [249, 115, 22]], [40, [220, 38, 38]], [50, [127, 29, 29]]];
    const WIND_STOPS = [[0, [30, 58, 138]], [3, [14, 165, 233]], [6, [34, 197, 94]], [10, [250, 204, 21]], [14, [249, 115, 22]], [18, [220, 38, 38]], [25, [126, 34, 206]]];
    const stopsCss = stops => `linear-gradient(90deg, ${stops.map(([v, c], i) => `rgb(${c.join(',')}) ${(100 * i / (stops.length - 1)).toFixed(0)}%`).join(', ')})`;

    const LAYERS = {
      temperature: { name: 'Temperature', icon: '🌡️', kind: 'field', src: 'LIVE',
        legend: { title: 'Air temperature (2 m)', bar: stopsCss(TEMP_STOPS), labels: ['-40', '-10', '10', '30', '50 °C'], source: 'Live hourly model data · Open-Meteo.com' } },
      wind: { name: 'Wind', icon: '💨', kind: 'wind', src: 'LIVE',
        legend: { title: 'Wind speed (10 m)', bar: stopsCss(WIND_STOPS), labels: ['0', '6', '14', '25 m/s'], source: 'Live hourly model data · Open-Meteo.com. Moving lines show wind direction.' } },
      precipitation: { name: 'Precipitation', icon: '🌧️', kind: 'gibs', src: 'NASA', layer: 'IMERG_Precipitation_Rate', lag: 2, opacity: 0.85,
        legend: { title: 'Precipitation rate', bar: 'linear-gradient(90deg, #a5f3fc, #22c55e, #facc15, #f97316, #dc2626, #a21caf)', labels: ['Light', 'Heavy'], source: 'NASA GPM IMERG via NASA GIBS (daily)' } },
      air: { name: 'Air quality', icon: '🌫️', kind: 'gibs', src: 'NASA', layer: 'MODIS_Combined_Value_Added_AOD', lag: 2, opacity: 0.8,
        legend: { title: 'Haze, smoke & dust (aerosol optical depth)', bar: 'linear-gradient(90deg, #fef9c3, #fdba74, #f97316, #b91c1c, #450a0a)', labels: ['Clean', 'Very hazy'], source: 'NASA MODIS (Terra + Aqua) via NASA GIBS. Gaps = cloud or no satellite pass. Click the map for live AQI.' } },
      night: { name: 'Night lights', icon: '🌃', kind: 'gibs', src: 'NASA', layer: 'VIIRS_Black_Marble', static: true, opacity: 1, format: 'image/png', transparent: false,
        legend: { title: 'Earth at night', bar: 'linear-gradient(90deg, #000, #1e293b, #fde68a, #fff)', labels: ['Dark', 'Bright'], source: 'NASA Black Marble (VIIRS, 2016) via NASA GIBS' } }
    };
    const LAYER_FOR_VAR = { T2M: 'temperature', T2M_MAX: 'temperature', T2M_MIN: 'temperature', WS10M: 'wind', PRECTOTCORR: 'precipitation',
    };

    let activeLayer = 'temperature', gibsLayer = null, gibsDays = [], gibsDayIdx = 0, gibsErrors = 0, gibsLoads = 0;
    let liveGrid = null, liveGridBox = null, liveGridTime = 0, liveIdx = 0, liveLoading = false;
    let fieldCanvas = null, windCanvas = null, particles = [], windRAF = null, fieldRAF = null, timePlayTimer = null;

    function initLiveLayers() {
      // Canvases live in their own Leaflet pane: above the basemap & NASA tiles, below labels and markers
      const pane = map.createPane('livefield');
      pane.style.zIndex = 420;
      pane.style.pointerEvents = 'none';
      fieldCanvas = document.createElement('canvas'); fieldCanvas.className = 'field-canvas';
      windCanvas = document.createElement('canvas'); windCanvas.className = 'field-canvas';
      pane.appendChild(fieldCanvas); pane.appendChild(windCanvas);
      document.getElementById('layer-buttons').innerHTML = Object.entries(LAYERS).map(([k, l]) =>
        `<button type="button" class="layer-btn" data-layer="${k}" onclick="setActiveLayer('${k}')">${l.icon} ${l.name}<span class="src">${l.src}</span></button>`).join('');
      map.on('move', scheduleFieldDraw);
      map.on('zoomstart', () => { clearCanvas(fieldCanvas); clearCanvas(windCanvas); });
      map.on('zoomend resize', () => { resetParticles(); scheduleFieldDraw(); });
      map.on('moveend', debounce(onMapViewChanged, 600));
      setActiveLayer('temperature');
    }

    function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
    function clearCanvas(c) { if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height); }
    function sizeCanvases() {
      const size = map.getSize(), topLeft = map.containerPointToLayerPoint([0, 0]);
      [fieldCanvas, windCanvas].forEach(c => {
        if (c.width !== size.x || c.height !== size.y) { c.width = size.x; c.height = size.y; }
        L.DomUtil.setPosition(c, topLeft);   // keep the canvas pinned to the visible map while the pane pans
      });
    }
    function mapVisible() { const el = document.getElementById('map-card'); return el && el.offsetParent !== null; }

    function setMapStatus(text) {
      const box = document.getElementById('map-status');
      if (!text) { box.classList.add('hidden'); return; }
      document.getElementById('map-status-text').innerHTML = text;
      box.classList.remove('hidden');
    }

    function setActiveLayer(key) {
      activeLayer = key;
      const cfg = LAYERS[key];
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.toggle('active', b.dataset.layer === key));
      stopTimePlay();
      if (gibsLayer) { map.removeLayer(gibsLayer); gibsLayer = null; }
      cancelAnimationFrame(windRAF); windRAF = null;
      clearCanvas(fieldCanvas); clearCanvas(windCanvas);
      setMapStatus('');
      const lg = cfg.legend;
      document.getElementById('map-legend-box').innerHTML = `<b>${cfg.icon} ${escHtml(lg.title)}</b>` +
        (lg.bar ? `<div class="lg-bar" style="background:${lg.bar}"></div><div class="lg-labels">${lg.labels.map(x => `<span>${x}</span>`).join('')}</div>` : '') +
        `<div class="lg-src">${escHtml(lg.source)}</div>`;
      if (cfg.kind === 'gibs') {
        buildGibsDays(cfg);
        addGibsLayer(cfg);
      } else {
        buildLiveSteps();
        onMapViewChanged();
      }
    }

    /* ---------- NASA GIBS satellite layers ---------- */
    function isoDay(d) { return d.toISOString().slice(0, 10); }
    function buildGibsDays(cfg) {
      const steps = document.getElementById('time-steps'), bar = document.getElementById('time-bar');
      if (cfg.static) {
        gibsDays = [];
        document.getElementById('time-label').innerHTML = '2016<small>Annual composite</small>';
        steps.innerHTML = '<span class="muted" style="font-size:0.78rem;">This layer is a single NASA composite image, so there is no timeline.</span>';
        document.getElementById('time-play').disabled = true;
        return;
      }
      document.getElementById('time-play').disabled = false;
      const today = new Date();
      gibsDays = [];
      for (let k = cfg.lag + 7; k >= cfg.lag; k--) { const d = new Date(today); d.setUTCDate(d.getUTCDate() - k); gibsDays.push(isoDay(d)); }
      gibsDayIdx = gibsDays.length - 1;
      steps.innerHTML = gibsDays.map((d, i) => `<button type="button" class="tstep" data-i="${i}" onclick="setGibsDay(${i})">${new Date(d + 'T12:00:00Z').toLocaleDateString([], { month: 'short', day: 'numeric' })}</button>`).join('');
      markTimeStep(gibsDayIdx);
      updateGibsLabel();
      bar.classList.remove('hidden');
    }
    function updateGibsLabel() {
      const d = gibsDays[gibsDayIdx];
      document.getElementById('time-label').innerHTML = `${new Date(d + 'T12:00:00Z').toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}<small>NASA satellite · daily</small>`;
    }
    function addGibsLayer(cfg) {
      gibsErrors = 0; gibsLoads = 0;
      const opts = { layers: cfg.layer, format: cfg.format || 'image/png', transparent: cfg.transparent !== false, opacity: cfg.opacity,
        uppercase: true, attribution: 'Imagery: NASA GIBS', maxNativeZoom: 8, maxZoom: 18 };
      if (!cfg.static) opts.time = gibsDays[gibsDayIdx];
      gibsLayer = L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', opts);
      gibsLayer.on('loading', () => { gibsErrors = 0; gibsLoads = 0; setMapStatus('⏳ Loading NASA satellite imagery…'); });
      gibsLayer.on('tileload', () => { gibsLoads++; });
      gibsLayer.on('tileerror', () => { gibsErrors++; });
      gibsLayer.on('load', () => {
        setMapStatus(gibsLoads === 0 && gibsErrors > 0 ? '⚠️ NASA has no image for this day yet. Try an earlier day on the timeline.' : '');
      });
      gibsLayer.addTo(map);
    }
    function setGibsDay(i) {
      gibsDayIdx = i;
      markTimeStep(i);
      updateGibsLabel();
      if (gibsLayer) gibsLayer.setParams({ time: gibsDays[i] });
    }

    /* ---------- live hourly grid (Open-Meteo) ---------- */
    function viewBox() {
      const b = map.getBounds();
      let west = b.getWest(), east = b.getEast();
      if (east - west >= 300) { west = -180; east = 180; }
      else { const c = (west + east) / 2, shift = Math.round(c / 360) * 360; west -= shift; east -= shift; }
      const padLat = (b.getNorth() - b.getSouth()) * 0.15, padLon = (east - west) * 0.15;
      return { south: Math.max(-85, b.getSouth() - padLat), north: Math.min(85, b.getNorth() + padLat),
        west: Math.max(-180, west - padLon), east: Math.min(180, east + padLon) };
    }
    function boxCovers(outer, inner) {
      return outer && inner.south >= outer.south - 1e-6 && inner.north <= outer.north + 1e-6 && inner.west >= outer.west - 1e-6 && inner.east <= outer.east + 1e-6;
    }

    async function onMapViewChanged() {
      if (!map || !mapVisible()) return;
      const cfg = LAYERS[activeLayer];
      if (cfg.kind === 'gibs') return;
      const need = viewBox(), b = map.getBounds();
      const visible = { south: Math.max(-85, b.getSouth()), north: Math.min(85, b.getNorth()), west: Math.max(-180, need.west + (need.east - need.west) * 0.13), east: Math.min(180, need.east - (need.east - need.west) * 0.13) };
      const fresh = Date.now() - liveGridTime < 10 * 60 * 1000;
      const tooCoarse = liveGridBox && (liveGridBox.east - liveGridBox.west) > 2.5 * (need.east - need.west);
      if (!(fresh && boxCovers(liveGridBox, visible) && !tooCoarse)) await loadLiveGrid(need);
      scheduleFieldDraw();
      if (activeLayer === 'wind') startWind();
    }

    async function loadLiveGrid(box) {
      if (liveLoading) return;
      liveLoading = true;
      setMapStatus('⏳ Loading live weather for this view…');
      try {
        const q = new URLSearchParams({ south: box.south.toFixed(2), north: box.north.toFixed(2), west: box.west.toFixed(2), east: box.east.toFixed(2), nx: 13, ny: 9 });
        const res = await fetch(`/api/live/grid?${q}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'failed');
        liveGrid = data; liveGridBox = box; liveGridTime = Date.now();
        liveIdx = Math.min(liveIdx, data.times.length - 1);
        buildLiveSteps();
        setMapStatus('');
      } catch (err) {
        setMapStatus(`⚠️ ${escHtml(err.message)}`);
      } finally {
        liveLoading = false;
      }
    }

    function buildLiveSteps() {
      document.getElementById('time-play').disabled = false;
      const steps = document.getElementById('time-steps');
      if (!liveGrid) { steps.innerHTML = ''; document.getElementById('time-label').innerHTML = 'Now<small>Live · loading</small>'; return; }
      const idxs = liveGrid.times.map((t, i) => i).filter(i => i % 3 === 0);
      steps.innerHTML = idxs.map(i => `<button type="button" class="tstep" data-i="${i}" onclick="setLiveHour(${i})">${i === 0 ? 'Now' : hourLabel(liveGrid.times[i])}</button>`).join('');
      markTimeStep(liveIdx);
      updateLiveLabel();
    }
    function hourLabel(t) { return new Date(t + 'Z').toLocaleTimeString([], { hour: 'numeric' }); }
    function updateLiveLabel() {
      if (!liveGrid) return;
      const t = new Date(liveGrid.times[liveIdx] + 'Z');
      document.getElementById('time-label').innerHTML = `${liveIdx === 0 ? 'Now' : t.toLocaleDateString([], { weekday: 'short' }) + ' ' + t.toLocaleTimeString([], { hour: 'numeric' })}<small>Live · ${liveIdx === 0 ? 'current hour' : 'forecast'}</small>`;
    }
    function setLiveHour(i) { liveIdx = i; markTimeStep(i); updateLiveLabel(); scheduleFieldDraw(); }
    function markTimeStep(i) { document.querySelectorAll('.tstep').forEach(b => b.classList.toggle('active', Number(b.dataset.i) === i)); }

    function toggleTimePlay() { timePlayTimer ? stopTimePlay() : startTimePlay(); }
    function stopTimePlay() { clearInterval(timePlayTimer); timePlayTimer = null; document.getElementById('time-play').textContent = '▶'; }
    function startTimePlay() {
      const cfg = LAYERS[activeLayer];
      if (cfg.static) return;
      document.getElementById('time-play').textContent = '❚❚';
      timePlayTimer = setInterval(() => {
        if (cfg.kind === 'gibs') setGibsDay((gibsDayIdx + 1) % gibsDays.length);
        else if (liveGrid) setLiveHour((liveIdx + 3) % liveGrid.times.length);
      }, cfg.kind === 'gibs' ? 2500 : 1200);
    }

    /* ---------- field drawing ---------- */
    function interp(series, lat, lon) {
      const g = liveGrid; if (!g) return null;
      const ny = g.lats.length, nx = g.lons.length;
      const fy = (lat - g.lats[0]) / (g.lats[ny - 1] - g.lats[0]) * (ny - 1);
      let lonN = lon; while (lonN < g.lons[0]) lonN += 360; while (lonN >= g.lons[0] + 360) lonN -= 360;
      const fx = (lonN - g.lons[0]) / (g.lons[nx - 1] - g.lons[0]) * (nx - 1);
      if (fy < 0 || fy > ny - 1 || fx < 0 || fx > nx - 1) return null;
      const y0 = Math.min(ny - 2, Math.floor(fy)), x0 = Math.min(nx - 2, Math.floor(fx)), ty = fy - y0, tx = fx - x0;
      const at = (iy, ix) => { const v = series[iy * nx + ix]; return v ? v[liveIdx] : null; };
      const a = at(y0, x0), b = at(y0, x0 + 1), c = at(y0 + 1, x0), d = at(y0 + 1, x0 + 1);
      if ([a, b, c, d].some(v => v === null || v === undefined)) return null;
      return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
    function colorFor(stops, v) {
      if (v <= stops[0][0]) return stops[0][1];
      for (let i = 0; i < stops.length - 1; i++) {
        const [a, ca] = stops[i], [b, cb] = stops[i + 1];
        if (v <= b) { const k = (v - a) / (b - a); return ca.map((c, j) => Math.round(c + (cb[j] - c) * k)); }
      }
      return stops[stops.length - 1][1];
    }
    function scheduleFieldDraw() {
      if (fieldRAF) return;
      fieldRAF = requestAnimationFrame(() => { fieldRAF = null; drawField(); });
    }
    function drawField() {
      if (!map || !fieldCanvas || !mapVisible()) return;
      sizeCanvases();
      const ctx = fieldCanvas.getContext('2d');
      ctx.clearRect(0, 0, fieldCanvas.width, fieldCanvas.height);
      const kind = LAYERS[activeLayer].kind;
      if (!liveGrid || (kind !== 'field' && kind !== 'wind')) return;
      const step = 7, W = fieldCanvas.width, H = fieldCanvas.height;
      const wind = kind === 'wind';
      for (let y = 0; y < H; y += step) {
        for (let x = 0; x < W; x += step) {
          const ll = map.containerPointToLatLng([x + step / 2, y + step / 2]);
          let v;
          if (wind) { const u = interp(liveGrid.u, ll.lat, ll.lng), w = interp(liveGrid.v, ll.lat, ll.lng); v = u === null || w === null ? null : Math.hypot(u, w); }
          else v = interp(liveGrid.temperature, ll.lat, ll.lng);
          if (v === null) continue;
          const c = colorFor(wind ? WIND_STOPS : TEMP_STOPS, v);
          ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${wind ? 0.5 : 0.58})`;
          ctx.fillRect(x, y, step, step);
        }
      }
    }

    /* ---------- animated wind particles ---------- */
    function resetParticles() {
      particles = [];
      clearCanvas(windCanvas);
    }
    function spawn(p) {
      p.x = Math.random() * windCanvas.width; p.y = Math.random() * windCanvas.height;
      p.age = Math.floor(Math.random() * 90); p.max = 60 + Math.floor(Math.random() * 60);
      return p;
    }
    function startWind() {
      if (activeLayer !== 'wind' || windRAF) return;
      sizeCanvases();
      const ctx = windCanvas.getContext('2d');
      const count = Math.round(Math.min(2200, windCanvas.width * windCanvas.height / 260));
      particles = Array.from({ length: count }, () => spawn({}));
      let lastCenter = map.getCenter();
      const frame = () => {
        if (activeLayer !== 'wind') { windRAF = null; clearCanvas(windCanvas); return; }
        windRAF = requestAnimationFrame(frame);
        if (!mapVisible() || !liveGrid) return;
        const c = map.getCenter();
        if (c.lat !== lastCenter.lat || c.lng !== lastCenter.lng) { lastCenter = c; sizeCanvases(); clearCanvas(windCanvas); particles.forEach(spawn); return; }
        ctx.globalCompositeOperation = 'destination-in';
        ctx.fillStyle = 'rgba(0, 0, 0, 0.92)';
        ctx.fillRect(0, 0, windCanvas.width, windCanvas.height);
        ctx.globalCompositeOperation = 'source-over';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        const scale = 0.18 * Math.pow(1.35, map.getZoom() - 2);
        for (const p of particles) {
          const ll = map.containerPointToLatLng([p.x, p.y]);
          const u = interp(liveGrid.u, ll.lat, ll.lng), v = interp(liveGrid.v, ll.lat, ll.lng);
          if (u === null || v === null || p.age++ > p.max) { spawn(p); continue; }
          const nx = p.x + u * scale, ny = p.y - v * scale;
          ctx.moveTo(p.x, p.y); ctx.lineTo(nx, ny);
          p.x = nx; p.y = ny;
          if (p.x < 0 || p.y < 0 || p.x > windCanvas.width || p.y > windCanvas.height) spawn(p);
        }
        ctx.stroke();
      };
      windRAF = requestAnimationFrame(frame);
    }

    /* =====================================================================
       RIGHT COLUMN: live readout, annual cycle, trend at a glance
       ===================================================================== */
    function renderRightColumn(d) {
      const u = shortUnit(unitOf(d.parameter)), pub = d.public;
      // annual cycle: average of each calendar month over the whole period
      const sums = Array(12).fill(0), counts = Array(12).fill(0);
      d.dates.forEach((dt, i) => { const m = Number(dt.slice(5, 7)) - 1; sums[m] += d.values[i]; counts[m]++; });
      const cycle = sums.map((s, m) => counts[m] ? s / counts[m] : null);
      document.getElementById('cycle-empty').classList.add('hidden');
      document.getElementById('cycle-wrap').classList.remove('hidden');
      document.getElementById('cycle-years').textContent = `${d.dates[0].slice(0, 4)}–${d.dates[d.dates.length - 1].slice(0, 4)}`;
      destroyChart('cycle');
      if (typeof Chart !== 'undefined') {
        charts.cycle = new Chart(document.getElementById('cycleChart').getContext('2d'), {
          type: 'bar',
          data: { labels: ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], datasets: [{ data: cycle, backgroundColor: 'rgba(56, 189, 248, 0.75)', borderRadius: 4 }] },
          options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${Number(c.parsed.y).toFixed(2)} ${u}` } } },
            scales: { x: { ...AXIS, grid: { display: false } }, y: { ...AXIS, ticks: { ...AXIS.ticks, maxTicksLimit: 4 } } } }
        });
      }
      // trend at a glance
      const ev = EVIDENCE[pub.evidence], dec = niceDecimals(pub.total_change);
      const arrow = pub.direction === 'up' ? '↑' : pub.direction === 'down' ? '↓' : '→';
      const gb = document.getElementById('glance-body');
      gb.innerHTML = `<div style="display:flex;align-items:center;gap:10px;"><div class="headline-arrow ${pub.direction}" style="width:42px;height:42px;font-size:1.3rem;border-radius:12px;">${arrow}</div>` +
        `<div><div style="font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:700;">${signed(pub.total_change, dec)} ${escHtml(u)}</div>` +
        `<div class="muted" style="font-size:0.74rem;">since ${d.dates[0].slice(0, 4)} · ${ev.word}</div></div></div>`;
      document.getElementById('glance-empty').classList.add('hidden');
      gb.classList.remove('hidden');
      document.getElementById('glance-wrap').classList.remove('hidden');
      destroyChart('glance');
      if (typeof Chart !== 'undefined' && pub.yearly.values.length) {
        charts.glance = new Chart(document.getElementById('glanceChart').getContext('2d'), {
          type: 'line',
          data: { labels: pub.yearly.years, datasets: [
            { data: pub.yearly.values, borderColor: '#38bdf8', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
            { data: pub.yearly.fit || [], borderColor: '#fb7185', borderDash: [5, 4], borderWidth: 2, pointRadius: 0 }
          ] },
          options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: { x: { display: false }, y: { display: false } } }
        });
      }
    }

    document.addEventListener('DOMContentLoaded', () => {
      initGlobe();
      document.getElementById('landing-years').textContent = `1981–${new Date().getFullYear() - 1}`;
    });

    /* =====================================================================
       DISCOVER: scan all variables / compare two places
       ===================================================================== */
    let discoverMode = 'auto', lastDiscover = null, lastCompare = null;
    let pickingB = false, markerB = null, locB = null;

    function lastDiscoverHtml() { return document.getElementById('discover-results').innerHTML.trim().length > 0; }

    function switchDiscoverMode(mode) {
      discoverMode = mode;
      document.querySelectorAll('.subtab-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
      document.getElementById('disc-mode-auto').classList.toggle('hidden', mode !== 'auto');
      document.getElementById('disc-mode-compare').classList.toggle('hidden', mode !== 'compare');
      document.getElementById('discover-btn').textContent = mode === 'auto' ? '🧭 Scan all variables' : '⚖️ Compare the two places';
      if (mode !== 'compare' && pickingB) togglePickB();
      updateMarkerB();
    }

    function runDiscover() { return discoverMode === 'auto' ? runAutoDiscover() : runCompare(); }

    function effectWord(e) { const a = Math.abs(e); return a < 0.1 ? 'small' : a < 0.3 ? 'moderate' : 'large'; }
    function evidenceFromP(p) { return p < 0.01 ? 'strong' : p < 0.05 ? 'moderate' : p < 0.10 ? 'weak' : 'none'; }

    async function runAutoDiscover() {
      if (selectedLat === null) { alert('Click a place on the map first.'); return; }
      const btn = document.getElementById('discover-btn');
      btn.disabled = true;
      setLoading(true, 'Downloading every variable from NASA POWER and testing each one…');
      try {
        const res = await fetch('/api/discover', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lat: selectedLat, lon: selectedLon, start_year: yearStart, end_year: yearEnd }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        lastDiscover = data;
        renderDiscoverResults(data);
      } catch (err) {
        document.getElementById('discover-results').innerHTML = `<div class="err-box">⚠️ ${escHtml(err.message)}</div>`;
      } finally {
        setLoading(false); btn.disabled = false; updateResultsVisibility();
      }
    }

    function renderDiscoverResults(data) {
      const box = document.getElementById('discover-results'), science = audienceMode === 'science';
      const ok = data.results.filter(r => r.status === 'ok'), sig = ok.filter(r => r.significant_after_adjustment);
      let html = `<div class="cf-head"><div><div class="kicker">Discover</div><h2>What is changing at ${escHtml(data.place ? data.place.label : '')}?</h2></div>` +
        `<div class="kicker">${data.start_year}–${data.end_year} · NASA POWER monthly data</div></div>`;
      if (data.power_error) html += `<div class="err-box" style="margin-bottom:14px;">⚠️ ${escHtml(data.power_error)}. No variables could be tested.</div>`;
      html += `<div class="cf-grid"><div class="cf-card glass-panel span-7"><div class="cf-q">Which variables are changing most?</div>` +
        `<p class="helper">${sig.length} of ${ok.length} variables show a real (significant) trend. Bars show how big the change is compared with normal ups and downs: red rising, blue falling, faded = could be chance.</p>` +
        `<div class="chart-box" style="height:${Math.max(240, ok.length * 34)}px;"><canvas id="discChart"></canvas></div></div>` +
        `<div class="cf-card glass-panel span-5"><div class="cf-q">Investigate one</div><div class="disc-list">`;
      data.results.forEach((r, i) => {
        if (r.status !== 'ok') { html += `<div class="disc-row na"><div class="disc-rank">–</div><div>${escHtml(varShort(r.id))}<div class="disc-meta">${escHtml(r.reason || 'Unavailable')}</div></div><div></div></div>`; return; }
        const u = shortUnit(r.unit), arrow = r.slope_per_decade > 0 ? '↑' : '↓';
        const meta = science
          ? `${signed(r.slope_per_decade, 3)} ${escHtml(u)}/dec · effect ${signed(r.effect_size_sd_per_decade, 2)} SD/dec · MK p ${r.mann_kendall_p_value}, BH-adj. ${r.adjusted_p_value}` +
            (r.mk_hamed_rao_p_deseasonalized !== null && r.mk_hamed_rao_p_deseasonalized !== undefined ? `, Hamed-Rao ${r.mk_hamed_rao_p_deseasonalized}` : '')
          : `${arrow} ${Math.abs(r.slope_per_decade).toFixed(niceDecimals(r.slope_per_decade))} ${escHtml(u)} per decade · ${effectWord(r.effect_size_sd_per_decade)} change`;
        html += `<div class="disc-row ${r.significant_after_adjustment ? 'sig' : ''}"><div class="disc-rank">${i + 1}</div>` +
          `<div>${(VAR_LOOK[r.id] || {}).icon || ''} ${escHtml(varShort(r.id))} ${r.significant_after_adjustment ? '<span class="tag ok">REAL TREND</span>' : '<span class="tag no">COULD BE CHANCE</span>'}<div class="disc-meta">${meta}</div></div>` +
          `<button class="ghost-btn" onclick="investigateFromDiscover('${r.id}')">Open →</button></div>`;
      });
      html += `</div>${science ? '<p class="helper" style="margin-top:10px;">Significance after Benjamini–Hochberg FDR correction of Mann-Kendall p-values; effect size = trend per decade ÷ SD of deseasonalized anomalies.</p>' : ''}</div></div>`;
      box.innerHTML = html;
      updateResultsVisibility();
      drawDiscoverChart(ok);
    }

    function drawDiscoverChart(ok) {
      if (typeof Chart === 'undefined') return;
      destroyChart('disc');
      const rows = ok.slice().sort((a, b) => Math.abs(b.effect_size_sd_per_decade) - Math.abs(a.effect_size_sd_per_decade));
      const colors = rows.map(r => {
        const base = r.effect_size_sd_per_decade >= 0 ? [239, 68, 68] : [59, 130, 246];
        return `rgba(${base.join(',')}, ${r.significant_after_adjustment ? 0.9 : 0.3})`;
      });
      charts.disc = new Chart(document.getElementById('discChart').getContext('2d'), {
        type: 'bar',
        data: { labels: rows.map(r => `${(VAR_LOOK[r.id] || {}).icon || ''} ${varShort(r.id)}`), datasets: [{ data: rows.map(r => r.effect_size_sd_per_decade), backgroundColor: colors, borderRadius: 6 }] },
        options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => { const r = rows[c.dataIndex]; return `${signed(r.slope_per_decade, 3)} ${shortUnit(r.unit)}/decade · ${r.significant_after_adjustment ? 'real trend' : 'could be chance'}`; } } } },
          scales: { x: { ...AXIS, title: { display: true, text: 'Change per decade (in units of normal variability)', color: '#94a3b8' } }, y: { ...AXIS, ticks: { color: '#e2e8f0', font: { size: 12 } } } } }
      });
    }

    function investigateFromDiscover(id) {
      selectVariable(id);
      switchTab('investigate');
      runInvestigation();
    }

    function togglePickB() {
      pickingB = !pickingB;
      const btn = document.getElementById('pick-b-btn');
      btn.classList.toggle('picking', pickingB);
      btn.textContent = pickingB ? 'Click the map…' : '📍 Pick on map';
    }

    async function setLocationB(lat, lon) {
      lon = wrapLon(lon);
      locB = { lat, lon };
      document.getElementById('loc-b-lat').value = lat.toFixed(4);
      document.getElementById('loc-b-lon').value = lon.toFixed(4);
      if (pickingB) togglePickB();
      updateMarkerB();
      const nameEl = document.getElementById('loc-b-name');
      nameEl.textContent = 'Locating…';
      try {
        const res = await fetch(`/api/place?lat=${lat}&lon=${lon}`);
        const p = await res.json();
        nameEl.textContent = res.ok ? `${p.label} (${p.coords})` : formatCoords(lat, lon);
      } catch (e) { nameEl.textContent = formatCoords(lat, lon); }
    }

    function updateMarkerB() {
      if (!map) return;
      const show = activeTab === 'discover' && discoverMode === 'compare' && locB;
      if (show) {
        if (!markerB) markerB = L.circleMarker([locB.lat, locB.lon], { radius: 9, color: '#fb7185', weight: 3, fillColor: '#fb7185', fillOpacity: 0.45 });
        markerB.setLatLng([locB.lat, locB.lon]);
        if (!map.hasLayer(markerB)) markerB.addTo(map);
      } else if (markerB && map.hasLayer(markerB)) map.removeLayer(markerB);
    }

    async function runCompare() {
      if (selectedLat === null) { alert('Click place A on the map first.'); return; }
      const lat = parseFloat(document.getElementById('loc-b-lat').value), lon = parseFloat(document.getElementById('loc-b-lon').value);
      if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90) { alert('Set place B: pick it on the map or type a latitude (-90 to 90) and longitude.'); return; }
      if (!locB || Math.abs(lat - locB.lat) > 1e-6 || Math.abs(lon - locB.lon) > 1e-6) await setLocationB(lat, lon);
      const btn = document.getElementById('discover-btn');
      btn.disabled = true;
      setLoading(true, 'Downloading both places from NASA POWER…');
      try {
        const res = await fetch('/api/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ parameter: document.getElementById('compare-var').value, start_year: yearStart, end_year: yearEnd,
            locations: [{ lat: selectedLat, lon: selectedLon }, { lat: locB.lat, lon: locB.lon }] }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        lastCompare = data;
        renderCompare(data);
      } catch (err) {
        document.getElementById('discover-results').innerHTML = `<div class="err-box">⚠️ ${escHtml(err.message)}</div>`;
      } finally {
        setLoading(false); btn.disabled = false; updateResultsVisibility();
      }
    }

    function renderCompare(data) {
      const [A, B] = data.locations, u = escHtml(shortUnit(data.unit)), science = audienceMode === 'science';
      const diff = A.stats.slope_per_decade - B.stats.slope_per_decade;
      const sigTxt = s => s.is_significant ? '✅ Real trend' : '⚠️ Could be chance';
      const row = (label, fa, fb) => `<tr><td>${label}</td><td class="cmp-a">${fa}</td><td class="cmp-b">${fb}</td></tr>`;
      let html = `<div class="cf-head"><div><div class="kicker">Compare</div><h2>${escHtml(data.name)}: ${escHtml(A.place.label)} vs ${escHtml(B.place.label)}</h2></div>` +
        `<div class="kicker">${escHtml(A.first)} – ${escHtml(A.last)} · NASA POWER</div></div><div class="cf-grid">` +
        `<div class="cf-card glass-panel span-7"><div class="cf-q">Yearly averages</div><div class="chart-box"><canvas id="compareChart"></canvas></div></div>` +
        `<div class="cf-card glass-panel span-5"><div class="cf-q">Side by side</div><p class="headline-text" style="margin-bottom:12px;">` +
        (Math.abs(diff) < 1e-9 ? 'Both places change at the same rate.' : `<b>${escHtml(diff > 0 ? A.place.label : B.place.label)}</b> has the higher trend, by ${Math.abs(diff).toFixed(3)} ${u} per decade.`) + `</p>` +
        `<table class="cmp-table"><tr><th></th><th class="cmp-a">A</th><th class="cmp-b">B</th></tr>` +
        row('Average', `${A.stats.mean} ${u}`, `${B.stats.mean} ${u}`) +
        row('Change / decade', signed(A.stats.slope_per_decade, 3), signed(B.stats.slope_per_decade, 3)) +
        row('Is it real?', sigTxt(A.stats), sigTxt(B.stats)) +
        (science ? row('95% CI (±)', A.stats.ci_95_decade, B.stats.ci_95_decade) + row('Mann-Kendall p', A.stats.mann_kendall_p_value, B.stats.mann_kendall_p_value) + row('t-test p', A.stats.ttest_p_value, B.stats.ttest_p_value) : '') +
        `</table></div></div>`;
      document.getElementById('discover-results').innerHTML = html;
      updateResultsVisibility();
      if (typeof Chart === 'undefined') return;
      destroyChart('compare');
      const years = [...new Set([...A.annual.years, ...B.annual.years])].sort((a, b) => a - b);
      const align = L_ => years.map(y => { const i = L_.annual.years.indexOf(y); return i >= 0 ? L_.annual.values[i] : null; });
      charts.compare = new Chart(document.getElementById('compareChart').getContext('2d'), {
        type: 'line',
        data: { labels: years, datasets: [
          { label: `A · ${A.place.label}`, data: align(A), borderColor: '#38bdf8', backgroundColor: '#38bdf8', borderWidth: 2, pointRadius: 3, tension: 0.25, spanGaps: true },
          { label: `B · ${B.place.label}`, data: align(B), borderColor: '#fb7185', backgroundColor: '#fb7185', borderWidth: 2, pointRadius: 3, tension: 0.25, spanGaps: true }
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: LEGEND }, scales: { x: AXIS, y: AXIS } }
      });
    }

    /* =====================================================================
       GUIDED EXAMPLES + WATCHABLE DEMO
       ===================================================================== */
    // Eight guided investigations. Card pictures are real NASA satellite views (MODIS true colour via NASA GIBS).
    const EXAMPLES = [
      { icon: '🧊', title: 'Is the Arctic warming faster?', question: 'How fast is Svalbard, in the high Arctic, warming?', param: 'T2M', lat: 78.2232, lon: 15.6267, place: 'Svalbard (Norway)',
        bbox: [8, 76.5, 24, 81], date: '2023-07-20', look: 'Compare the warming per decade with the global average of roughly +0.2 °C per decade.' },
      { icon: '🌧️', title: 'Rain in the desert', question: 'Is rainfall changing in a desert city like Dubai?', param: 'PRECTOTCORR', lat: 25.2048, lon: 55.2708, place: 'Dubai, UAE',
        bbox: [53.9, 24.0, 56.9, 25.7], date: '2024-04-16', look: 'In a desert most months are dry, so single wet months stand out. Switch the trend chart to Monthly.' },
      { icon: '🏜️', title: 'Snow in the Sahara', question: 'Does snow ever settle at Aïn Séfra, on the edge of the Sahara?', param: 'SNODP', lat: 32.75, lon: -0.58, place: 'Aïn Séfra, Algeria',
        bbox: [-2.6, 31.7, 1.4, 33.9], date: '2018-01-08', look: 'Most months have no snow at all, so a real detective may find no trend. That is a result too!' },
      { icon: '🇨🇦', title: "Canada's winter snowpack", question: 'Is the snow on the ground around Ottawa getting shallower?', param: 'SNODP', lat: 45.4215, lon: -75.6972, place: 'Ottawa, Canada',
        bbox: [-77.6, 44.5, -73.8, 46.6], date: '2019-02-20', look: 'Look at "Which months changed?": the winter months tell the story.' },
      { icon: '🔥', title: 'Fire-weather conditions', question: 'Are the hottest temperatures rising in British Columbia’s wildfire country?', param: 'T2M_MAX', lat: 49.888, lon: -119.496, place: 'Kelowna, Canada',
        bbox: [-121.5, 48.8, -117.5, 51.0], date: '2023-08-18', look: 'In Scientist mode, the Indicators tab counts hot and dry "fire-weather" months.' },
      { icon: '🌱', title: 'Is the soil drying out?', question: 'Is root-zone soil moisture changing near Manaus in the Amazon?', param: 'GWETPROF', lat: -3.119, lon: -60.0217, place: 'Manaus, Brazil',
        bbox: [-62.0, -4.2, -58.0, -2.0], date: '2023-10-10', look: 'Soil wetness runs from 0 (dry) to 1 (saturated). Then show the trend map: does the whole area dry out?' },
      { icon: '🌾', title: 'Are growing conditions shifting?', question: 'Is the Canadian prairie around Saskatoon getting warmer?', param: 'T2M', lat: 52.1332, lon: -106.67, place: 'Saskatoon, Canada',
        bbox: [-108.7, 51.0, -104.7, 53.2], date: '2023-07-25', look: 'Warmer springs and autumns mean a longer growing season. Check which months changed most.' },
      { icon: '☀️', title: 'Is the Earth getting sunnier or dimmer?', question: 'Is the sunlight reaching the ground over Beijing changing?', param: 'ALLSKY_SFC_SW_DWN', lat: 39.9042, lon: 116.4074, place: 'Beijing, China',
        bbox: [114.4, 38.8, 118.4, 41.0], date: '2023-10-20', look: 'Air pollution and clouds can dim sunlight; cleaner air can brighten it.' }
    ];

    function gibsSnapshot(ex) {
      const [w, s_, e, n] = ex.bbox;
      return 'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1' +
        `&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&STYLES=&SRS=EPSG:4326&BBOX=${w},${s_},${e},${n}&WIDTH=480&HEIGHT=270&FORMAT=image/jpeg&TIME=${ex.date}`;
    }

    function renderExamples() {
      document.getElementById('example-list').innerHTML = EXAMPLES.map((ex, i) =>
        `<div class="inv-card" role="group" aria-label="${escHtml(ex.title)}">` +
        `<div class="inv-img" style="background-image:url('${gibsSnapshot(ex)}'), linear-gradient(135deg,#0c4a6e,#3b0764);"><span class="badge">NASA · ${ex.date}</span></div>` +
        `<div class="inv-body"><h4>${ex.icon} ${escHtml(ex.title)}</h4>` +
        `<div class="inv-actions"><button type="button" class="play-btn" onclick="startDemo(${i})">▶ Watch</button><button type="button" class="ghost-btn" onclick="tryExample(${i})">Try it</button></div></div></div>`).join('');
    }

    async function loadExample(ex) {
      selectVariable(ex.param);
      setLastYears(0);
      await setPoint(ex.lat, ex.lon, true);
    }

    async function tryExample(i) {
      const ex = EXAMPLES[i];
      switchTab('investigate');
      await loadExample(ex);
      runInvestigation();
    }

    let demo = null;
    function demoSteps(ex) {
      const v = varById(ex.param) || { name: ex.param, audience_simple: '' };
      return [
        { target: '#tab-intro', text: () => `🎬 <b>${escHtml(ex.title)}</b> ${escHtml(ex.question)} Let's investigate it like a detective, in three steps.` },
        { target: '#step-what', action: () => { selectVariable(ex.param); }, text: () => `<b>Step 1 · What?</b> We choose <b>${escHtml(varShort(ex.param))}</b>. ${escHtml(v.audience_simple)}` },
        { target: '#map-card', action: () => setPoint(ex.lat, ex.lon, true), text: () => `<b>Step 2 · Where?</b> We click on <b>${escHtml(ex.place)}</b>. The map shows the matching <b>${escHtml(LAYERS[activeLayer].name.toLowerCase())}</b> layer.` },
        { target: '#step-when', action: () => setLastYears(0), text: () => `<b>Step 3 · When?</b> We use the whole NASA record: <b>${yearStart}–${yearEnd}</b>. Longer records make trends easier to detect.` },
        { target: '#investigate-btn', wait: true, action: () => runInvestigation(), text: () => 'Now we press <b>Investigate</b>. The app downloads every month of data live from NASA POWER and analyses it…' },
        { target: '#cf-headline', text: () => { const p = lastAnalysis.public, u = shortUnit(unitOf(lastAnalysis.parameter)); return `<b>How much?</b> ${escHtml(varShort(ex.param))} has ${p.direction === 'up' ? 'risen' : 'fallen'} by about <b>${Math.abs(p.total_change).toFixed(niceDecimals(p.total_change))} ${escHtml(u)}</b> since ${lastAnalysis.dates[0].slice(0, 4)}, which is ${signed(p.sen_per_decade, niceDecimals(p.sen_per_decade))} ${escHtml(u)} per decade.`; } },
        { target: '#cf-confidence', text: () => `<b>Is it real?</b> ${EVIDENCE[lastAnalysis.public.evidence].word}. ${EVIDENCE[lastAnalysis.public.evidence].text}` },
        { target: '#cf-trend', text: () => 'Each dot is one year. The dashed line is the trend, and the shaded band shows how certain we are about it.' },
        { target: '#cf-stripes', text: () => 'Climate stripes tell the whole story at a glance: blue years were below the long-term average, red years above it.' },
        { target: '#cf-months', text: () => `<b>Which months changed?</b> Blue is the early years, red the most recent ones. ${escHtml(ex.look)}` },
        { target: '#right-col', text: () => 'On the right: the <b>annual cycle</b> (average for each month) and the <b>trend at a glance</b> for this place.' },
        { target: '#cf-where', text: () => '<b>Where?</b> The marker on the map is now coloured by the trend. Press "Show trend map around here" to check whether nearby areas change the same way.' },
        { target: '#panel-investigate', text: () => '🕵️ Case solved! Now it’s your turn: choose a variable, a place and the years, and press Investigate.' }
      ];
    }

    async function startDemo(i) {
      demoStop();
      const ex = EXAMPLES[i];
      switchTab('investigate');
      demo = { ex, steps: demoSteps(ex), i: -1, paused: false, timer: null, raf: null };
      document.getElementById('demo-title').textContent = ex.title;
      document.getElementById('demo-bar').classList.remove('hidden');
      demoNext();
    }

    async function demoNext() {
      if (!demo) return;
      clearTimeout(demo.timer); cancelAnimationFrame(demo.raf);
      demo.i += 1;
      if (demo.i >= demo.steps.length) { demoStop(); return; }
      const step = demo.steps[demo.i], token = demo.i;
      document.querySelectorAll('.demo-focus').forEach(e => e.classList.remove('demo-focus'));
      document.getElementById('demo-step').textContent = `Demo · step ${demo.i + 1} of ${demo.steps.length}`;
      const el = document.querySelector(step.target);
      if (el) { el.classList.add('demo-focus'); el.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
      if (step.wait) document.getElementById('demo-text').innerHTML = step.text();
      let ok = true;
      if (step.action) { try { const r = step.action(); if (step.wait) ok = await r; } catch (e) { ok = false; } }
      if (!demo || demo.i !== token) return;
      if (step.wait && ok === false) {
        document.getElementById('demo-text').innerHTML = '⚠️ NASA POWER could not be reached, so the demo stopped. Please check your internet connection and try again.';
        return;
      }
      document.getElementById('demo-text').innerHTML = step.text();
      if (step.wait && el) { el.classList.remove('demo-focus'); }
      demoSchedule(7000);
    }

    function demoSchedule(ms) {
      const bar = document.getElementById('demo-progress'), start = performance.now();
      demo.remaining = ms;
      const tick = (now) => {
        if (!demo) return;
        if (demo.paused) { demo.raf = requestAnimationFrame(tick); return; }
        const done = Math.min(1, (now - start - (demo.pausedFor || 0)) / ms);
        bar.style.width = `${done * 100}%`;
        if (done >= 1) { demo.pausedFor = 0; demoNext(); } else demo.raf = requestAnimationFrame(tick);
      };
      demo.pausedFor = 0;
      demo.raf = requestAnimationFrame(tick);
    }

    function demoTogglePause() {
      if (!demo) return;
      demo.paused = !demo.paused;
      if (demo.paused) demo.pauseStart = performance.now();
      else demo.pausedFor = (demo.pausedFor || 0) + performance.now() - demo.pauseStart;
      document.getElementById('demo-pause').textContent = demo.paused ? '▶ Play' : '⏸ Pause';
    }

    function demoStop() {
      if (demo) { clearTimeout(demo.timer); cancelAnimationFrame(demo.raf); }
      demo = null;
      document.querySelectorAll('.demo-focus').forEach(e => e.classList.remove('demo-focus'));
      document.getElementById('demo-bar').classList.add('hidden');
      document.getElementById('demo-pause').textContent = '⏸ Pause';
      document.getElementById('demo-progress').style.width = '0';
    }

    /* =====================================================================
       LEARN
       ===================================================================== */
    const LEARN_TOPICS = [
      { title: 'What is a trend?',
        simple: 'A trend is the long-term direction of a value, ignoring the ups and downs of the seasons. <b>"+0.3 °C per decade"</b> means the value goes up about 0.3 °C every 10 years.',
        science: 'The case file removes the seasonal cycle (each month minus its long-term calendar-month mean) and estimates the trend with Sen’s slope, the median of all pairwise slopes. The yearly chart shows annual means with an OLS line and its 95% confidence band. The Scientist panel adds OLS, autocorrelation-adjusted OLS and seasonal Sen estimates.' },
      { title: 'Is the change real? (the evidence meter)',
        simple: 'A trend could appear just by luck, because weather naturally goes up and down. The meter shows how unlikely it is that luck alone made the trend: <b>Strong</b> means less than a 1 in 100 chance, <b>Moderate</b> less than 1 in 20, <b>Weak</b> less than 1 in 10.',
        science: 'The meter uses the Mann-Kendall test with the Hamed & Rao (1998) variance correction for autocorrelation, applied to deseasonalized monthly anomalies: p < 0.01 strong, < 0.05 moderate, < 0.10 weak. Correcting for autocorrelation avoids the overconfident p-values that consecutive months otherwise produce.' },
      { title: 'Climate stripes',
        simple: 'Each stripe is one year. Blue years were below the average of the chosen period and red years above it. Darker colours mean bigger differences. A shift from blue to red shows a rising trend at a glance.',
        science: 'Stripe colour = annual mean minus the mean of all complete years in the period, scaled by the largest absolute departure, on a diverging blue–white–red scale (after Hawkins’ warming stripes). Only calendar years with all 12 months are shown.' },
      { title: 'The Mann-Kendall test',
        simple: 'Mann-Kendall checks, for every pair of months, whether the later one is higher or lower than the earlier one. If "higher" wins much more often, there is an upward trend.',
        science: 'S = Σ sign(xⱼ − xₖ) over all pairs, Var(S) = [n(n−1)(2n+5) − Σ tied-group terms]/18, Z with continuity correction, and Kendall’s τ = S / [n(n−1)/2]. It is rank-based, so it is robust to outliers and doesn’t assume a straight line or normal data.' },
      { title: 'Unusual months',
        simple: 'A month is unusual when it is far from normal <b>for that time of year</b>. A very hot January is unusual even if it is colder than a normal July.',
        science: 'Values are deseasonalized and divided by the calendar-month standard deviation (Z-score). A month is flagged when an Isolation Forest (contamination 3%) marks it as an outlier and |Z| ≥ 2.2; |Z| ≥ 3.2 is labelled Extreme.' },
      { title: 'Discover: testing many variables at once',
        simple: 'If you test 11 things, one might look "real" just by chance. Discover makes the test stricter to account for this, so the trends it highlights are more trustworthy.',
        science: 'Mann-Kendall p-values across all tested variables are adjusted with the Benjamini–Hochberg procedure (FDR 5%). Effect size = trend per decade ÷ SD of deseasonalized monthly anomalies, which makes variables with different units comparable.' },
      { title: 'Where does the data come from?',
        simple: 'All data comes from <b>NASA POWER</b>, a free NASA service built from satellite measurements and computer models of the atmosphere. The app downloads it live every time you investigate, and never makes up numbers: if NASA can’t be reached, it shows an error. Place names come from OpenStreetMap.',
        science: 'NASA POWER monthly point API (community AG). Meteorology from the MERRA-2 reanalysis (1981 onward), including bias-corrected precipitation; surface irradiance and cloud amount from SRB (1984–2000) and CERES SYN1deg / FLASHFlux (2001 onward). Month-13 annual values and −999 fill values are removed; at least 24 valid months are required. No synthetic or fallback data is used anywhere.' },
      { title: 'The maps',
        simple: '<b>Temperature</b> and <b>wind</b> are live maps from a weather model (Open-Meteo): colours show the values, moving lines show the wind, and ▶ plays the next 24 hours. <b>Precipitation</b>, <b>air quality</b> and <b>night lights</b> are real NASA satellite images; step through the last week with the timeline. The maps are for exploring; the trend answers always use NASA POWER data.',
        science: 'Temperature and wind: hourly 2 m temperature and 10 m wind from the Open-Meteo forecast API on a 13 × 9 grid covering the view, bilinearly interpolated (particles advected with u/v). NASA GIBS WMS (EPSG:3857): IMERG_Precipitation_Rate, MODIS_Combined_Value_Added_AOD and VIIRS_Black_Marble (2016). None of these feed the trend statistics.' },
      { title: 'Point data vs. regional averages',
        simple: 'Results are for the grid square around your marker (tens of kilometres wide), not a whole city or country. The trend map shows how the trend varies across the surrounding area.',
        science: 'Each request returns the POWER grid cell containing the point (≈ 0.5° × 0.625° for MERRA-2 meteorology, 1° for solar). The trend map samples a grid of points around it; the Scientist region tool adds a cos(latitude)-weighted regional mean.' }
    ];

    const LEARN_SCIENCE_ONLY = [
      { title: "Sen's slope and its confidence interval",
        body: "Median of all pairwise slopes (xⱼ − xᵢ)/(tⱼ − tᵢ), i &lt; j. The 100(1−α)% interval runs from the M₁-th to the (M₂+1)-th ordered slope, with M₁ = (N′ − C)/2, M₂ = (N′ + C)/2 and C = z₁₋α/₂ √Var(S), where N′ is the number of pairwise slopes (Gilbert 1987)." },
      { title: 'Seasonal Mann-Kendall test',
        body: 'S is computed separately for each calendar month and summed: S = Σₘ Sₘ, Var(S) = Σₘ Var(Sₘ) (Hirsch, Slack & Smith 1982).' },
      { title: 'Autocorrelation: Hamed-Rao and effective sample size',
        body: 'Hamed & Rao (1998) multiply Var(S) by 1 + 2/[n(n−1)(n−2)] Σ (n−k)(n−k−1)(n−k−2) ρₖ over significant rank autocorrelations ρₖ of the Sen-detrended series. For regression, n_eff = n(1 − r₁)/(1 + r₁) replaces n in the standard error and degrees of freedom (Santer et al. 2000).' },
      { title: 'How long a record do you need?',
        body: 'Weatherhead et al. (1998): years of monthly data needed to detect a trend ω (per year) with 90% probability at the 5% level: n* ≈ [(3.3 σ_N / |ω|) √((1 + φ)/(1 − φ))]^(2/3), with noise SD σ_N and lag-1 autocorrelation φ.' },
      { title: 'Residual diagnostics',
        body: 'Durbin-Watson d = Σ(eₜ − eₜ₋₁)² / Σeₜ² (≈ 2 for independent residuals). Shapiro-Wilk tests residual normality; if it fails, prefer the rank-based Mann-Kendall and Sen results.' },
      { title: 'Machine-learning forecast',
        body: 'Seasonal Ridge regression (time + 1st/2nd harmonics), Random Forest and Gradient Boosting (harmonics, trained on the linearly detrended series; trend added back). Ensemble = mean of the three; band = ±1.96 × residual SD of the Ridge fit. Extrapolation only, not a physical climate projection.' },
      { title: 'References',
        body: 'Benjamini & Hochberg (1995) J. R. Stat. Soc. B 57 · Durbin & Watson (1950) Biometrika 37 · Gilbert (1987) Statistical Methods for Environmental Pollution Monitoring · Hamed & Rao (1998) J. Hydrology 204 · Hirsch, Slack & Smith (1982) Water Resour. Res. 18 · Kendall (1975) Rank Correlation Methods · Mann (1945) Econometrica 13 · Santer et al. (2000) J. Geophys. Res. 105 · Sen (1968) J. Am. Stat. Assoc. 63 · Shapiro & Wilk (1965) Biometrika 52 · Weatherhead et al. (1998) J. Geophys. Res. 103.' }
    ];

    function renderLearn() {
      const science = audienceMode === 'science';
      const item = (title, body) => `<details><summary>${escHtml(title)}</summary><div class="learn-body">${body}</div></details>`;
      let m = '<div class="learn-section">Methods</div>' + LEARN_TOPICS.map(t => item(t.title, science ? t.science : t.simple)).join('');
      if (science) m += '<div class="learn-section" style="margin-top:18px;">Advanced (Scientist)</div>' + LEARN_SCIENCE_ONLY.map(t => item(t.title, t.body)).join('');
      document.getElementById('learn-methods').innerHTML = m;
      document.getElementById('learn-variables').innerHTML = '<div class="learn-section">Variables</div>' + variablesCatalog.map(v => item(`${(VAR_LOOK[v.id] || {}).icon || ''} ${v.name}`,
        `${escHtml(science ? v.audience_science : v.audience_simple)}<br/><br/><b>Unit:</b> ${escHtml(v.unit)} · <b>Source:</b> ${escHtml(v.agency)} · <b>Available:</b> ${v.first_year}–${dataMaxYear}<br/><b>Example of an unusual event:</b> ${escHtml(v.anomaly_example)}`)).join('');
    }

    /* =====================================================================
       PULSE GUIDE: topic-based chatbot (no AI service, no API)
       Topics → sub-topics → answer and/or an action that navigates the site.
       Typed questions are matched against every answer's keywords.
       ===================================================================== */
    const VAR_GROUPS = [
      { id: 'cat_temp', label: '🌡️ Temperature', vars: ['T2M', 'T2M_MAX', 'T2M_MIN'] },
      { id: 'cat_water', label: '💧 Water, rain & snow', vars: ['PRECTOTCORR', 'SNODP', 'RH2M', 'QV2M', 'GWETPROF'] },
      { id: 'cat_air', label: '💨 Wind', vars: ['WS10M'] },
      { id: 'cat_sun', label: '☀️ Sun & clouds', vars: ['ALLSKY_SFC_SW_DWN', 'CLOUD_AMT'] }
    ];
    const VAR_WORDS = {
      T2M: ['temperature', 'temp', 'warming', 'hot', 'cold', 'heat'], T2M_MAX: ['max', 'maximum', 'hottest', 'heatwave'], T2M_MIN: ['min temp', 'minimum', 'coldest', 'warm nights'],
      PRECTOTCORR: ['rain', 'rainfall', 'precipitation', 'precip'], SNODP: ['snow', 'snowpack'], RH2M: ['humidity', 'humid'], QV2M: ['vapour', 'vapor', 'moisture air'],
      GWETPROF: ['soil', 'drought', 'dry'], WS10M: ['wind', 'windy', 'breeze'], ALLSKY_SFC_SW_DWN: ['sun', 'sunlight', 'solar', 'sunny', 'dimmer'], CLOUD_AMT: ['cloud', 'cloudy', 'clouds']
    };

    function scrollToEl(sel) { const el = document.querySelector(sel); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }); return !!el; }
    function needCase(sel) {
      if (!lastAnalysis) { guideNote('You need a result first. Choose a variable and a place, then press <b>Investigate</b>.', ['pick']); return; }
      switchTab('investigate'); setTimeout(() => scrollToEl(sel), 60);
    }
    function showLayer(k) { switchTab('investigate'); setActiveLayer(k); scrollToEl('#map-card'); }

    const G = {
      root: { label: '🏠 Main menu', text: "Hi, I'm <b>Pulse Guide</b> 🛰️. I can help you choose what to investigate, find your way around, or explain any result. What would you like to do?",
        children: ['pick', 'howto', 'maps', 'results', 'examples', 'discover', 'scientist', 'data', 'help'] },

      pick: { label: '🔍 Choose what to investigate', kw: 'choose pick select variable what investigate topic data',
        text: 'What kind of change are you curious about?', children: VAR_GROUPS.map(g => g.id) },

      howto: { label: '🧭 How to use this site', kw: 'how use start steps begin guide tutorial navigate',
        text: 'Which part would you like help with?', children: ['h_steps', 'h_place', 'h_years', 'h_modes', 'h_tabs'] },
      h_steps: { label: 'The 3 steps', kw: 'steps how work investigate button start',
        text: '<b>1 · What?</b> (optional) pick a variable from the list or ask me; Temperature is the default. <b>2 · Where?</b> click a place on the map. <b>3 · When?</b> pick the years on the timeline. Then press <b>🔍 Investigate</b> and your case file appears below the map.',
        actions: [{ label: '🔍 Choose a variable', go: 'pick' }, { label: '📍 Show me the map', run: () => scrollToEl('#map-card') }] },
      h_place: { label: 'Choosing a place', kw: 'place location map click where city country marker',
        text: 'Click anywhere on the map. A marker appears and the place name shows at the top of the map. The data comes from the NASA grid square (about 50 km) around that point.',
        actions: [{ label: '📍 Show me the map', run: () => scrollToEl('#map-card') }] },
      h_years: { label: 'Choosing the years', kw: 'years timeline period when date range slider',
        text: 'Drag the two handles of the timeline in step 3, or use the quick buttons. NASA POWER data starts in 1981 (1984 for sunlight and cloud) and runs to the last complete year. Longer periods (20+ years) reveal trends more reliably.',
        actions: [{ label: 'Use the full record', run: () => { setLastYears(0); guideNote(`Timeline set to <b>${yearStart}–${yearEnd}</b>.`); } }, { label: 'Last 20 years', run: () => { setLastYears(20); guideNote(`Timeline set to <b>${yearStart}–${yearEnd}</b>.`); } }, { label: '📅 Show the timeline', run: () => scrollToEl('#step-when') }] },
      h_modes: { label: 'Public vs Scientist', kw: 'mode public scientist switch difference statistics',
        text: '<b>Public</b> shows clear visuals and plain answers. <b>Scientist</b> adds the full statistics: extra trend tests, seasonal breakdown, diagnostics, forecast, unusual months, indicators and CSV export. Switch any time at the top right.',
        actions: [{ label: '🎓 Switch to Public', run: () => { setAudienceMode('simple'); guideNote('Public mode is on.'); } }, { label: '🔬 Switch to Scientist', run: () => { setAudienceMode('science'); guideNote('Scientist mode is on.'); } }] },
      h_tabs: { label: 'What the tabs do', kw: 'tabs menu discover examples learn investigate sections pages',
        text: '<b>Investigate</b>: the 3-step flow. <b>Discover</b>: scan every variable at a place, or compare two places. <b>Guided Examples</b>: watch real cases being solved. <b>Learn</b>: plain explanations of every method and variable.',
        actions: [{ label: 'Open Discover', run: () => switchTab('discover') }, { label: 'Open Guided Examples', run: () => switchTab('examples') }, { label: 'Open Learn', run: () => switchTab('learn') }] },

      maps: { label: '🗺️ The maps', kw: 'map maps layer layers satellite image',
        text: 'The map has 5 layers: live temperature and wind, plus NASA precipitation, air quality and night lights. Which one?', children: ['m_temp', 'm_wind', 'm_precip', 'm_air', 'm_night', 'm_time', 'm_trendmap'] },
      m_temp: { label: '🌡️ Temperature map', kw: 'temperature map layer heat live', text: 'A live colour map of air temperature right now (purple and blue cold, red hot). Press ▶ under the map to play the next 24 hours. It comes from a live weather model (Open-Meteo), not NASA.', actions: [{ label: 'Show it', run: () => showLayer('temperature') }] },
      m_precip: { label: '🌧️ Precipitation map', kw: 'rain precipitation map layer imerg gpm', text: 'Daily rain and snow from NASA’s GPM IMERG, which combines many satellites. Light colours are light rain, red and purple are heavy.', actions: [{ label: 'Show it', run: () => showLayer('precipitation') }] },
      m_air: { label: '🌫️ Air quality map', kw: 'air quality pollution smoke haze dust aerosol map', text: 'Haze, smoke and dust (aerosol optical depth) from NASA MODIS. Yellow is clean air, dark red is very hazy. Gaps mean clouds or no satellite pass that day.', actions: [{ label: 'Show it', run: () => showLayer('air') }] },
      m_wind: { label: '💨 Wind speed map', kw: 'wind speed map layer live particles', text: 'Live wind: colours show the speed (blue calm, red and purple strong) and moving white lines show where the wind is blowing. Press ▶ to play the next 24 hours. It comes from a live weather model (Open-Meteo), not NASA.', actions: [{ label: 'Show it', run: () => showLayer('wind') }] },
      m_night: { label: '🌃 Night lights map', kw: 'night lights city black marble', text: 'NASA Black Marble: Earth at night in 2016. Bright areas are cities and roads.', actions: [{ label: 'Show it', run: () => showLayer('night') }] },
      m_time: { label: '▶ Times & play button', kw: 'date dates time play animate timeline map day hour', text: 'The strip under the map lists times: the next 24 hours for live temperature and wind, or the last week for NASA layers. Tap one to see it, or press ▶ to play through them. If a NASA date looks empty, it has not been published yet; tap an earlier one.', actions: [{ label: 'Show the map', run: () => scrollToEl('#map-card') }] },
      m_trendmap: { label: '🟥🟦 Trend map', kw: 'trend map around area nearby region grid', text: 'After an investigation, "Trend map around here" colours a 4 × 4 grid of nearby points: red rising, blue falling, white outline = a real (significant) trend. It shows whether the area changes the same way.',
        actions: [{ label: 'Make the trend map', run: () => lastAnalysis ? showTrendMapAround() : guideNote('Run an investigation first, then I can draw the trend map.', ['pick']) }] },

      results: { label: '📊 Understanding my results', kw: 'results result case file understand explain meaning',
        text: 'Which part of the case file?', children: ['r_much', 'r_real', 'r_trend', 'r_stripes', 'r_months', 'r_where'] },
      r_much: { label: 'How much did it change?', kw: 'how much change number total per decade headline', text: 'The big number is the total change over your period, after setting aside the normal seasons. "Per decade" is the same change spread over every 10 years.', actions: [{ label: 'Take me there', run: () => needCase('#cf-headline') }] },
      r_real: { label: 'Is the change real?', kw: 'real significant significance evidence chance meter p value sure', text: 'The meter shows how unlikely it is that natural ups and downs alone made the trend: <b>Strong</b> = less than 1 in 100, <b>Moderate</b> = less than 1 in 20, <b>Weak</b> = less than 1 in 10.', actions: [{ label: 'Take me there', run: () => needCase('#cf-confidence') }] },
      r_trend: { label: 'The trend chart', kw: 'trend chart graph line dots band yearly monthly', text: 'Each dot is one year’s average. The dashed line is the trend and the shaded band shows its uncertainty. Switch to <b>Monthly</b> to see every month, with unusual months in red.', actions: [{ label: 'Take me there', run: () => needCase('#cf-trend') }] },
      r_stripes: { label: 'Climate stripes', kw: 'stripes stripe colours colors blue red', text: 'One stripe per year: blue years were below the period’s average, red years above. A shift from blue to red means a rising trend.', actions: [{ label: 'Take me there', run: () => needCase('#cf-stripes') }] },
      r_months: { label: 'Which months changed?', kw: 'months month season seasonal which changed early recent', text: 'Two lines compare the average for each month in the first years and the most recent years, so you can see which part of the year changed most.', actions: [{ label: 'Take me there', run: () => needCase('#cf-months') }] },
      r_where: { label: 'Unusual months & where', kw: 'unusual extreme anomaly anomalies outlier marker where', text: 'This card lists the months that were far from normal for their time of year, and lets you draw the trend map around your place.', actions: [{ label: 'Take me there', run: () => needCase('#cf-where') }] },

      examples: { label: '🎬 Examples & demo', kw: 'example examples demo watch case study preset tutorial video',
        text: 'Pick a real case to watch or try:', children: [] },

      discover: { label: '🧭 Discover & compare', kw: 'discover scan compare two places all variables which', text: 'What would you like to do?', children: ['d_scan', 'd_compare'] },
      d_scan: { label: 'Which variables are changing here?', kw: 'scan all variables changing most rank', text: 'Discover tests all 11 variables at your place and ranks them by how strongly they change. Click a place on the map first, then press <b>Scan all variables</b>.',
        actions: [{ label: 'Open it', run: () => { switchTab('discover'); switchDiscoverMode('auto'); } }] },
      d_compare: { label: 'Compare two places', kw: 'compare two places versus difference', text: 'Your map marker is place A. Pick place B on the map (or type its coordinates), choose a variable, then press <b>Compare</b> to see both on one chart.',
        actions: [{ label: 'Open it', run: () => { switchTab('discover'); switchDiscoverMode('compare'); } }] },

      scientist: { label: '🔬 Scientist mode', kw: 'scientist statistics advanced export csv diagnostics forecast', text: 'What do you want to know?', children: ['s_what', 's_export', 's_region'] },
      s_what: { label: 'What it adds', kw: 'scientist adds tests sen mann kendall hamed rao seasonal diagnostics forecast', text: 'A panel under the case file with tabs: Statistics (5 trend estimates, 5 significance tests), Seasons, Diagnostics (record length, assumption checks), Forecast, Unusual months, Indicators, and Data & export.',
        actions: [{ label: '🔬 Turn on Scientist mode', run: () => { setAudienceMode('science'); guideNote('Scientist mode is on. Run an investigation to fill the panel.'); } }] },
      s_export: { label: 'Download the data', kw: 'download export csv data methods citation', text: 'In Scientist mode, open the <b>Data & export</b> tab: download a CSV with every month and all statistics, copy a ready-made methods paragraph, and open the exact NASA request.',
        actions: [{ label: 'Take me there', run: () => { if (audienceMode !== 'science') setAudienceMode('science'); if (!lastAnalysis) { guideNote('Run an investigation first.', ['pick']); return; } setSciTab('data'); scrollToEl('#science-panel'); } }] },
      s_region: { label: 'Analyse a region', kw: 'region box area average draw', text: 'In Scientist mode choose <b>Region box</b>, press "Draw region on map", click two corners, then <b>Re-run analysis</b>. You get the area-average trend and a trend map.',
        actions: [{ label: 'Turn on Scientist mode', run: () => setAudienceMode('science') }] },

      data: { label: '📚 Data & sources', kw: 'data source sources nasa power where from real', text: 'Which data question?', children: ['dt_power', 'dt_maps', 'dt_real', 'dt_names'] },
      dt_power: { label: 'Where trend data comes from', kw: 'nasa power merra ceres source trend data monthly', text: 'Every trend uses <b>NASA POWER</b> monthly data, downloaded live each time you investigate. Weather variables come from NASA’s MERRA-2 (from 1981), and sunlight and cloud from NASA’s SRB and CERES (from 1984).' },
      dt_maps: { label: 'Where the maps come from', kw: 'maps gibs satellite images source open meteo live', text: 'Precipitation (GPM IMERG), air quality (MODIS aerosols) and night lights (Black Marble) come from <b>NASA GIBS</b>, NASA’s satellite imagery service. The live temperature and wind maps come from <b>Open-Meteo</b>, a free weather-model service. The trend answers always use NASA POWER.' },
      dt_real: { label: 'Is the data real?', kw: 'real fake synthetic true actual', text: 'Yes. The site never makes up numbers. If NASA can’t be reached, it shows an error instead of results.' },
      dt_names: { label: 'Where place names come from', kw: 'place names openstreetmap nominatim city name', text: 'Place names come from OpenStreetMap. Only the names; all the science data is NASA’s.' },

      help: { label: '🛠️ Something isn’t working', kw: 'help problem error broken not working wrong issue', text: 'What’s happening?', children: ['hp_button', 'hp_nodata', 'hp_map', 'hp_slow'] },
      hp_button: { label: "The Investigate button won't work", kw: 'button disabled grey investigate not clickable', text: 'It stays grey until you click a place on the map (step 2). Step 1 is optional: Temperature is used unless you pick another variable.', actions: [{ label: '🔍 Choose a variable', go: 'pick' }] },
      hp_nodata: { label: '"Couldn’t get NASA data"', kw: 'no data couldnt error nasa reach fail', text: 'NASA POWER didn’t answer, usually a slow connection or a busy NASA server. Wait a moment and press Investigate again.' },
      hp_map: { label: 'The map layer looks empty', kw: 'map empty blank layer missing image', text: 'NASA publishes daily images 1–3 days late, so tap an earlier date in the strip under the map. If live temperature or wind do not appear, the live weather service may be busy; move the map a little or try again in a minute.' },
      hp_slow: { label: 'It’s slow', kw: 'slow loading wait long time', text: 'Each investigation downloads up to 45 years of monthly data from NASA and runs the statistics, which can take 5–20 seconds. The trend map downloads 16 points, so it takes longer.' }
    };

    // variable groups and variables become nodes automatically
    VAR_GROUPS.forEach(g => { G[g.id] = { label: g.label, kw: g.vars.map(v => (VAR_WORDS[v] || []).join(' ')).join(' '), text: 'Which one?', children: g.vars.map(v => `v_${v}`) }; });
    function buildVarNodes() {
      variablesCatalog.forEach(v => {
        G[`v_${v.id}`] = { label: `${(VAR_LOOK[v.id] || {}).icon || ''} ${varShort(v.id)}`, kw: `${v.name} ${(VAR_WORDS[v.id] || []).join(' ')}`,
          text: () => `<b>${escHtml(v.name)}</b>: ${escHtml(audienceMode === 'science' ? v.audience_science : v.audience_simple)} (unit: ${escHtml(v.unit)})<br/><br/>` +
            `✅ Selected for step 1.${selectedLat === null ? ' Next, <b>click a place on the map</b>.' : ' Your place is set, so you can press <b>Investigate</b>.'}`,
          onShow: () => { switchTab('investigate'); selectVariable(v.id); },
          actions: [{ label: '📍 Show me the map', run: () => scrollToEl('#map-card') },
            { label: '🔍 Investigate now', run: () => { if (selectedLat === null) guideNote('First click a place on the map.'); else runInvestigation(); } },
            { label: '🔁 Pick another', go: 'pick' }] };
      });
      G.examples.children = EXAMPLES.map((ex, i) => {
        G[`ex_${i}`] = { label: `${ex.icon} ${ex.title}`, kw: `${ex.title} ${ex.question} ${ex.place}`, text: `${escHtml(ex.question)} <i>(${escHtml(ex.place)})</i>`,
          actions: [{ label: '▶ Watch the demo', run: () => startDemo(i) }, { label: 'Try it myself', run: () => tryExample(i) }] };
        return `ex_${i}`;
      });
    }

    let guideHistory = [];
    function guideLog() { return document.getElementById('guide-log'); }
    function guideBubble(cls, html) {
      const m = document.createElement('div');
      m.className = `g-msg ${cls}`;
      m.innerHTML = html;
      guideLog().appendChild(m);
      guideLog().scrollTop = guideLog().scrollHeight;
      return m;
    }
    function guideOptions(m, ids, actions, nav) {
      const wrap = document.createElement('div');
      wrap.className = 'g-opts';
      (ids || []).forEach(id => { if (G[id]) wrap.appendChild(optBtn(G[id].label, '', () => { guideBubble('user', escHtml(G[id].label)); guideShow(id); })); });
      (actions || []).forEach(a => wrap.appendChild(optBtn(a.label, 'act', () => {
        guideBubble('user', escHtml(a.label));
        if (a.go) guideShow(a.go); else a.run();
      })));
      if (nav) {
        if (guideHistory.length > 1) wrap.appendChild(optBtn('⬅ Back', 'nav', () => { guideHistory.pop(); const prev = guideHistory.pop(); guideShow(prev); }));
        wrap.appendChild(optBtn('🏠 Main menu', 'nav', () => guideShow('root')));
      }
      m.appendChild(wrap);
      guideLog().scrollTop = guideLog().scrollHeight;
    }
    function optBtn(label, cls, fn) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = `g-opt ${cls}`; b.textContent = label;
      b.addEventListener('click', fn);
      return b;
    }

    function guideShow(id) {
      const n = G[id]; if (!n) return;
      if (id === 'root') guideHistory = [];
      guideHistory.push(id);
      if (n.onShow) n.onShow();
      const m = guideBubble('bot', typeof n.text === 'function' ? n.text() : n.text);
      guideOptions(m, n.children, n.actions, id !== 'root');
    }

    // short bot note (used by actions); optional follow-up topic chips
    function guideNote(html, ids) {
      if (!document.getElementById('guide-log')) return;
      const m = guideBubble('bot', html);
      if (ids && ids.length) guideOptions(m, ids, null, false);
    }

    // The guide is optional: hidden until the user asks for it
    function toggleGuide(forceOpen) {
      const box = document.getElementById('guide'), btn = document.getElementById('guide-toggle');
      if (!btn) return;
      const open = forceOpen === true ? true : box.classList.contains('hidden');
      box.classList.toggle('hidden', !open);
      btn.textContent = open ? '✕ Close Pulse Guide' : '💬 Need help? Ask Pulse Guide';
      if (open) setTimeout(() => document.getElementById('guide-input').focus(), 50);
    }

    function guideStart() {
      buildVarNodes();
      guideLog().innerHTML = '';
      guideShow('root');
    }

    /* ---------- typed questions ---------- */
    const G_STOP = new Set('a an the is are was be to of in on at for and or how what why where when which do does did can could i my me you it this that with about show tell please want need'.split(' '));
    function gWords(s) { return String(s).toLowerCase().replace(/<[^>]+>/g, ' ').replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter(w => w && !G_STOP.has(w)).map(w => w.length > 4 ? w.replace(/(ing|ed|es|s)$/, '') : w); }

    function guideAsk(e) {
      e.preventDefault();
      const input = document.getElementById('guide-input'), q = input.value.trim();
      if (!q) return;
      input.value = '';
      guideBubble('user', escHtml(q));
      const low = q.toLowerCase();
      if (/^(hi|hello|hey|salam|yo)\b/.test(low)) { guideShow('root'); return; }
      if (/^(thanks|thank you|thx|great|cool|ok|okay)\b/.test(low)) { guideNote('You’re welcome! Anything else?', ['root']); return; }

      // a variable mentioned together with a "choose" intent → select it
      const qw = new Set(gWords(q));
      const wantsPick = /\b(investigate|choose|pick|select|look at|study|analy[sz]e|check|trend)\b/.test(low);
      let varHit = null;
      for (const [vid, words] of Object.entries(VAR_WORDS)) if (words.some(w => low.includes(w))) { varHit = vid; if (!['T2M'].includes(vid)) break; }

      let best = null, bestScore = 0;
      Object.entries(G).forEach(([id, n]) => {
        if (id === 'root') return;
        const bag = new Set(gWords(`${n.label} ${n.kw || ''} ${typeof n.text === 'string' ? n.text : ''}`));
        const kwBag = new Set(gWords(`${n.label} ${n.kw || ''}`));
        let s = 0;
        qw.forEach(w => { if (kwBag.has(w)) s += 2; else if (bag.has(w)) s += 0.5; });
        s = s / Math.sqrt(qw.size || 1);
        if (!n.children || !n.children.length) s += 0.05;   // prefer a direct answer over a menu
        if (s > bestScore) { bestScore = s; best = id; }
      });

      const aboutMap = /\b(map|layer|lights)\b/.test(low);
      if (varHit && !aboutMap && (wantsPick || qw.size <= 2) && G[`v_${varHit}`]) { guideShow(`v_${varHit}`); return; }
      if (best && bestScore >= 1) { guideShow(best); return; }
      const m = guideBubble('bot', 'I’m not sure I understood. Here are the topics I can help with:');
      guideOptions(m, G.root.children, null, false);
    }

    /* =====================================================================
       SCIENTIST MODE (forecast, unusual months and indicators live here)
       ===================================================================== */
    let sciScope = 'point', lastScience = null, sciBreakdown = 'season', sciTab = 'stats';
    let drawingRegion = false, regionCorner = null, regionBox = null, regionBoxLayer = null;

    function applyScientistLayout() {
      updateResultsVisibility();
      const science = audienceMode === 'science';
      if (!science && drawingRegion) toggleRegionDraw();
      if (regionBoxLayer && map) { if (science && sciScope === 'region') { if (!map.hasLayer(regionBoxLayer)) regionBoxLayer.addTo(map); } else if (map.hasLayer(regionBoxLayer)) map.removeLayer(regionBoxLayer); }
      if (science && lastAnalysis) { renderScientistExtras(); if (!lastScience) runScience(); }
    }

    function setSciTab(tab) {
      sciTab = tab;
      document.querySelectorAll('.sci-tab').forEach(b => b.classList.toggle('active', b.dataset.st === tab));
      document.querySelectorAll('.sci-pane').forEach(p => p.classList.toggle('hidden', p.id !== `st-${tab}`));
      Object.values(charts).forEach(c => { try { c.resize(); } catch (e) {} });
    }

    function setSciScope(scope) {
      sciScope = scope;
      document.querySelectorAll('#sci-scope button').forEach(b => b.classList.toggle('active', b.dataset.scope === scope));
      document.getElementById('sci-region-bar').classList.toggle('hidden', scope !== 'region');
      document.getElementById('sci-grid-wrap').classList.toggle('hidden', scope !== 'region');
      if (scope !== 'region' && drawingRegion) toggleRegionDraw();
      if (regionBoxLayer && map) { if (scope === 'region') regionBoxLayer.addTo(map); else if (map.hasLayer(regionBoxLayer)) map.removeLayer(regionBoxLayer); }
    }

    function toggleRegionDraw() {
      drawingRegion = !drawingRegion;
      regionCorner = null;
      const btn = document.getElementById('sci-draw-btn');
      btn.classList.toggle('active', drawingRegion);
      btn.textContent = drawingRegion ? 'Click the first corner on the map…' : '▭ Draw region on map';
      if (drawingRegion) document.getElementById('map-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function handleRegionClick(lat, lon) {
      lon = wrapLon(lon);
      if (!regionCorner) { regionCorner = { lat, lon }; document.getElementById('sci-draw-btn').textContent = 'Click the opposite corner…'; return; }
      const fit = (lo, hi, floor, ceil) => {
        const c = (lo + hi) / 2, span = Math.min(Math.max(hi - lo, 0.5), 10);
        lo = Math.max(floor, c - span / 2); hi = Math.min(ceil, lo + span); lo = hi - span;
        return [lo, hi];
      };
      const [south, north] = fit(Math.min(lat, regionCorner.lat), Math.max(lat, regionCorner.lat), -90, 90);
      const [west, east] = fit(Math.min(lon, regionCorner.lon), Math.max(lon, regionCorner.lon), -180, 180);
      regionBox = { south, north, west, east };
      toggleRegionDraw();
      if (regionBoxLayer) map.removeLayer(regionBoxLayer);
      regionBoxLayer = L.rectangle([[south, west], [north, east]], { color: '#c084fc', weight: 2, dashArray: '6 4', fill: false }).addTo(map);
      document.getElementById('sci-box-text').textContent = `Box: ${formatCoords(south, west)} to ${formatCoords(north, east)} (${(north - south).toFixed(1)}° × ${(east - west).toFixed(1)}°). Press Re-run analysis.`;
    }

    async function runScience() {
      if (!lastAnalysis && sciScope === 'point') return;
      const status = document.getElementById('sci-status'), btn = document.getElementById('sci-run-btn');
      const body = { parameter: selectedVar, start_year: yearStart, end_year: yearEnd,
        alpha: parseFloat(document.getElementById('sci-alpha').value), aggregation: document.getElementById('sci-agg').value };
      let url = '/api/science';
      if (sciScope === 'region') {
        if (!regionBox) { status.innerHTML = '<div class="err-box">Draw a region on the map first (▭ Draw region on map, then click two corners).</div>'; return; }
        url = '/api/science/region'; body.box = regionBox; body.grid = parseInt(document.getElementById('sci-grid').value, 10);
        status.textContent = `Requesting ${body.grid * body.grid} grid points from NASA POWER… this can take up to a minute.`;
      } else {
        body.lat = selectedLat; body.lon = selectedLon;
        status.textContent = 'Computing the full statistics…';
      }
      btn.disabled = true;
      try {
        const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Request failed');
        lastScience = data;
        status.textContent = '';
        renderScience(data);
        if (data.mode === 'region') drawTrendCells(data, 'science');
      } catch (err) {
        status.innerHTML = `<div class="err-box">Scientific analysis failed: ${escHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
      }
    }

    function renderScience(d) {
      const u = shortUnit(unitOf(d.provenance.parameter)), p = d.provenance;
      const sen = d.estimates.find(e => e.method === "Sen's slope");
      const sigCount = d.tests.filter(t => t.significant).length;
      const where = d.mode === 'region' ? `the region around <strong>${escHtml(d.place.label)}</strong> (area-weighted mean of ${d.cells_loaded} points)` : `<strong>${escHtml(d.place.label)}</strong>`;
      const dir = sen.per_decade > 0 ? 'increasing' : sen.per_decade < 0 ? 'decreasing' : 'flat';
      const ci = Math.round((1 - d.alpha) * 100);

      // ---- Statistics ----
      let s = `<div class="sci-summary">${escHtml(p.variable)} at ${where}, ${escHtml(d.period.start)} to ${escHtml(d.period.end)}: <strong>${dir}</strong> by ` +
        `<strong>${signed(sen.per_decade, 4)} ${escHtml(u)} per decade</strong> (Sen's slope, ${ci}% CI ${fmtN(sen.ci_low, 4)} to ${fmtN(sen.ci_high, 4)}). ` +
        `<strong>${sigCount} of ${d.tests.length}</strong> tests reject "no trend" at α = ${d.alpha}.<br/><small class="muted">Tested series: ${escHtml(d.series_description)}.</small></div>`;
      s += `<div class="sci-grid"><div class="sci-block wide"><h3>Tested series with trend lines</h3><p class="helper">Dashed: OLS fit. Dotted: Sen's slope.</p><div class="chart-box short"><canvas id="sciChart"></canvas></div></div>`;
      s += `<div class="sci-block"><h3>Trend estimates</h3><p class="helper">Per decade, ${ci}% confidence intervals.</p><table class="sci-table"><tr><th>Method</th><th>${escHtml(u)}/dec</th><th>CI</th><th>p</th></tr>` +
        d.estimates.map(e => `<tr><td>${escHtml(e.method)}<small>${escHtml(e.note || '')}</small></td><td class="num">${signed(e.per_decade, 4)}</td>` +
          `<td class="num">${e.ci_low === null ? '–' : `${fmtN(e.ci_low)} to ${fmtN(e.ci_high)}`}</td><td class="num">${fmtP(e.p)}</td></tr>`).join('') + '</table></div>';
      s += `<div class="sci-block"><h3>Significance tests</h3><p class="helper">H₀: no monotonic trend; decision at α = ${d.alpha}.</p><table class="sci-table"><tr><th>Test</th><th>Statistic</th><th>p</th><th>Reject H₀?</th></tr>` +
        d.tests.map(t => `<tr><td>${escHtml(t.test)}${t.note ? `<small>${escHtml(t.note)}</small>` : ''}${t.tau !== undefined ? `<small>Kendall τ = ${t.tau}</small>` : ''}</td>` +
          `<td class="num">${t.statistic} = ${fmtN(t.value, 3)}</td><td class="num">${fmtP(t.p)}</td><td>${t.significant ? '<span class="pill-yes">Yes</span>' : '<span class="pill-no">No</span>'}</td></tr>`).join('') + '</table></div>';
      if (d.mode === 'region') {
        const ok = d.cells.filter(c => c.status === 'ok');
        const up = ok.filter(c => c.significant && c.per_decade > 0).length, down = ok.filter(c => c.significant && c.per_decade < 0).length;
        s += `<div class="sci-block wide"><h3>Regional trend map</h3><p class="helper">${ok.length} of ${d.cells.length} points loaded (deseasonalized OLS trend per point; significance from Hamed-Rao Mann-Kendall). Significant increases: <b>${up}</b>, decreases: <b>${down}</b>.` +
          (up && down ? ' <b>The same variable trends in opposite directions inside this region.</b>' : '') + ' See the map above.</p></div>';
      }
      s += '</div>';
      document.getElementById('st-stats').innerHTML = s;

      // ---- Seasons ----
      document.getElementById('st-seasons').innerHTML = `<div class="sci-block"><h3>Which part of the year is changing?</h3>` +
        `<p class="helper">Trend of each season's (or month's) yearly mean. Bars show the OLS trend; bold rows are significant at α = ${d.alpha}.</p>` +
        `<div class="seg seg-mini"><button type="button" class="${sciBreakdown === 'season' ? 'active' : ''}" onclick="setSciBreakdown('season')">Seasons</button><button type="button" class="${sciBreakdown === 'month' ? 'active' : ''}" onclick="setSciBreakdown('month')">Months</button></div>` +
        `<div id="sci-breakdown">${breakdownTable(d, u)}</div></div>`;

      // ---- Diagnostics ----
      const det = d.detectability, g = d.diagnostics;
      const pct = det.years_needed ? Math.min(100, 100 * det.years_have / det.years_needed) : 0;
      let dg = `<div class="sci-grid"><div class="sci-block"><h3>Is the record long enough?</h3><p class="helper">Years of monthly data needed to detect this trend with 90% probability at the 5% level (Weatherhead et al. 1998).</p>`;
      dg += det.years_needed
        ? `<div class="detect-big">${det.years_needed} years needed</div><div class="det-meter"><span style="width:${pct}%"></span></div>` +
          `<p class="headline-sub">You have <b>${det.years_have} years</b>. ${det.detectable_now ? 'The record is long enough to reliably detect a trend of this size.' : `About <b>${(det.years_needed - det.years_have).toFixed(1)} more years</b> would be needed; a non-significant result may simply mean the record is too short.`}</p>` +
          `<p class="headline-sub">Inputs: trend ${signed(det.slope_per_decade, 4)} ${escHtml(u)}/decade, σ<sub>N</sub> = ${det.noise_sd}, φ = ${det.phi}.</p>`
        : '<p class="helper">The estimated trend is essentially zero, so no detection time can be computed.</p>';
      dg += '</div>';
      const diag = (label, value, ok, okText, warnText) => `<div class="diag-row"><div>${label}<small>${ok ? okText : warnText}</small></div><div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;">${value}</div><div class="${ok ? 'diag-ok' : 'diag-warn'}">${ok ? 'OK' : 'CHECK'}</div></div>`;
      dg += '<div class="sci-block"><h3>Assumption checks</h3><p class="helper">Residuals of the OLS fit on the tested series.</p>' +
        diag('Lag-1 autocorrelation', fmtN(g.lag1_autocorrelation), Math.abs(g.lag1_autocorrelation) < 0.2, 'Residuals are close to independent.', 'Serial correlation present: rely on the adjusted and Hamed-Rao results.') +
        diag('Durbin-Watson', fmtN(g.durbin_watson), g.durbin_watson !== null && g.durbin_watson > 1.5 && g.durbin_watson < 2.5, 'Within 1.5–2.5.', 'Outside 1.5–2.5: autocorrelated residuals.') +
        diag('Shapiro-Wilk normality p', fmtP(g.shapiro_p), g.shapiro_p === null || g.shapiro_p > 0.05, 'No evidence against normal residuals.', 'Not normal: prefer Mann-Kendall and Sen.') +
        diag('Sample size', `n = ${g.n}, n<sub>eff</sub> = ${g.n_eff}`, g.n >= 20, 'Enough points for the tests used.', 'Short series: tests have little power.') +
        `<p class="headline-sub">R² of the linear fit: ${g.r_squared}</p></div></div>`;
      document.getElementById('st-diag').innerHTML = dg;

      // ---- Data & export ----
      document.getElementById('st-data').innerHTML = `<div class="sci-block"><h3>Data provenance</h3><table class="sci-table prov-table">` +
        `<tr><td>Variable</td><td>${escHtml(p.variable)} <code>${escHtml(p.parameter)}</code></td></tr>` +
        `<tr><td>Source</td><td>${escHtml(p.source || '')}</td></tr><tr><td>Spatial support</td><td>${escHtml(p.grid || '')}</td></tr>` +
        `<tr><td>Units</td><td>${escHtml(p.catalog_unit)}${p.api_units ? ` (NASA POWER reports: <code>${escHtml(p.api_units)}</code>)` : ''}</td></tr>` +
        (p.api_longname ? `<tr><td>NASA long name</td><td>${escHtml(p.api_longname)}</td></tr>` : '') +
        `<tr><td>Period</td><td>${escHtml(d.period.start)} to ${escHtml(d.period.end)}</td></tr><tr><td>Retrieved</td><td>${escHtml(p.retrieved_utc)}</td></tr>` +
        (p.url ? `<tr><td>Reproduce request</td><td><a href="${escHtml(p.url)}" target="_blank" rel="noopener">${escHtml(p.url)}</a></td></tr>` : '') +
        (p.note ? `<tr><td>Note</td><td>${escHtml(p.note)}</td></tr>` : '') + `</table>` +
        `<div class="sci-actions"><button type="button" class="ghost-btn" onclick="downloadScienceCsv()">⬇️ Download CSV (data + statistics)</button><button type="button" class="ghost-btn" onclick="copyMethods()">📋 Copy methods paragraph</button></div>` +
        `<div class="methods-text">${escHtml(methodsParagraph(d))}</div></div>`;

      drawSciChart(d, u);
    }

    function breakdownTable(d, u) {
      const rows = sciBreakdown === 'season' ? d.by_season : d.by_month;
      const maxAbs = Math.max(1e-9, ...rows.map(r => Math.abs(r.ols_per_decade)));
      return `<table class="sci-table"><tr><th>Period</th><th>OLS ${escHtml(u)}/dec</th><th>Sen ${escHtml(u)}/dec</th><th>OLS p</th><th>MK p</th><th>n (yrs)</th><th style="min-width:110px;"></th></tr>` +
        rows.map(r => {
          const w = 50 * Math.abs(r.ols_per_decade) / maxAbs;
          const bar = r.ols_per_decade >= 0 ? `<span style="left:50%;width:${w}%;background:#dc2626;"></span>` : `<span style="left:${50 - w}%;width:${w}%;background:#2563eb;"></span>`;
          return `<tr style="${r.significant ? 'font-weight:700;' : ''}"><td>${r.period}</td><td class="num">${signed(r.ols_per_decade, 4)}</td><td class="num">${signed(r.sen_per_decade, 4)}</td>` +
            `<td class="num">${fmtP(r.ols_p)}</td><td class="num">${fmtP(r.mk_p)}</td><td class="num">${r.n}</td><td><div class="bar-cell">${bar}<div class="mid"></div></div></td></tr>`;
        }).join('') + '</table>';
    }
    function setSciBreakdown(kind) { sciBreakdown = kind; if (lastScience) renderScience(lastScience); setSciTab('seasons'); }

    function drawSciChart(d, u) {
      if (typeof Chart === 'undefined') return;
      destroyChart('sci');
      charts.sci = new Chart(document.getElementById('sciChart').getContext('2d'), {
        type: 'line',
        data: { labels: d.series.labels, datasets: [
          { label: d.aggregation === 'monthly' ? `Anomaly (${u})` : `Value (${u})`, data: d.series.values, borderColor: '#38bdf8', backgroundColor: 'rgba(56, 189, 248, 0.1)', borderWidth: 1.4, pointRadius: d.aggregation === 'monthly' ? 0 : 2, tension: 0.15, fill: true },
          { label: 'OLS trend', data: d.series.ols_line, borderColor: '#fb7185', borderDash: [6, 4], borderWidth: 2, pointRadius: 0 },
          { label: "Sen's slope", data: d.series.sen_line, borderColor: '#c084fc', borderDash: [2, 3], borderWidth: 2, pointRadius: 0 }
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: LEGEND },
          scales: { x: { ...AXIS, ticks: { ...AXIS.ticks, maxTicksLimit: 12 } }, y: AXIS } }
      });
    }

    /* ---------- Forecast, unusual months, indicators (from /api/analyze) ---------- */
    function renderScientistExtras() {
      const d = lastAnalysis; if (!d) return;
      const u = shortUnit(unitOf(d.parameter));
      document.getElementById('st-forecast').innerHTML = `<div class="sci-block"><h3>3-year machine-learning forecast</h3>` +
        `<p class="helper">Seasonal Ridge, Random Forest and Gradient Boosting extend the past seasonal pattern and linear trend. An extrapolation, not a physical climate projection.</p>` +
        `<div class="chart-box"><canvas id="forecastChart"></canvas></div></div>`;
      const anomalies = d.ml.anomalies || [];
      document.getElementById('st-unusual').innerHTML = anomalies.length
        ? `<p class="helper">Months far from normal for their calendar month (Isolation Forest + |Z| ≥ 2.2). Likely causes and precautions are general guidance, not official warnings.</p>` +
          `<div class="sci-grid">${anomalies.map(a => `<div class="anom-card"><div class="ad">${escHtml(a.date)} · ${escHtml(a.severity.toUpperCase())} · ${escHtml(a.type)}</div>` +
            `<div class="av">${a.value} ${escHtml(u)} <span class="muted" style="font-size:0.8rem;font-weight:400;">normal ${a.expected} · Z = ${a.z_score}σ</span></div>` +
            `<div style="font-size:0.82rem;line-height:1.45;">${escHtml(a.explanation)}</div>` +
            `<details><summary>Precautions</summary><div class="prec">${escHtml(a.precautions)}</div></details></div>`).join('')}</div>`
        : '<p class="helper">No unusual months were flagged.</p>';
      const ch = d.challenges_integration;
      let ind;
      if (!ch || ch.error) ind = `<div class="err-box">${escHtml((ch && ch.error) || 'Indicators unavailable.')}</div>`;
      else {
        const sg = (v, k) => (v > 0 ? '+' : '') + Number(v).toFixed(k), h = ch.heat_drought, g = ch.growing, f = ch.fire_weather;
        const foot = `<div style="font-size:0.72rem;margin-top:6px;">${ch.periods.early} vs ${ch.periods.late} · ${escHtml(ch.source)}</div>`;
        ind = `<p class="helper">Computed from NASA POWER monthly temperature, max temperature, precipitation, humidity, wind and root-zone soil wetness at this point, comparing the start and end of the period.</p><div class="ind-grid">` +
          `<div class="ind-box"><h4>🌡️ Heat & drought stress</h4><strong>${escHtml(h.headline)}</strong><br/>Hottest month each year: <strong>${sg(h.hottest_month_trend_per_decade, 2)} °C/decade</strong> (p = ${h.hottest_month_trend_p})<br/>Dry months (driest 20% for their month): <strong>${h.dry_month_share_early}% → ${h.dry_month_share_late}%</strong><br/>Root-zone soil wetness: <strong>${h.soil_wetness_early} → ${h.soil_wetness_late}</strong>${foot}</div>` +
          `<div class="ind-box"><h4>🌾 Growing conditions</h4><strong>${escHtml(g.headline)}</strong> (rough guide)<br/>Months averaging ≥ 5 °C: <strong>${g.season_months_early} → ${g.season_months_late}</strong><br/>Growing degree-days (base 10 °C, approx.): <strong>${g.gdd_early} → ${g.gdd_late}</strong><br/>Growing-season rain: <strong>${g.season_rain_early_mm} → ${g.season_rain_late_mm} mm</strong>${g.season_rain_change_pct !== null ? ` (${sg(g.season_rain_change_pct, 1)}%)` : ''}${foot}</div>` +
          `<div class="ind-box"><h4>🔥 Fire-weather conditions</h4><strong>${escHtml(f.headline)}</strong><br/>Hot and dry months per year: <strong>${f.months_per_year_early} → ${f.months_per_year_late}</strong><br/>Wind in those months: <strong>${f.wind_in_fire_months === null ? '–' : f.wind_in_fire_months + ' m/s'}</strong> vs ${f.wind_all_months} m/s overall<br/>Weather that favours fires, not detected fires.${foot}</div></div>`;
      }
      document.getElementById('st-indicators').innerHTML = ind;
      drawForecastChart(d, u);
    }

    function drawForecastChart(d, u) {
      if (typeof Chart === 'undefined') return;
      destroyChart('forecast');
      const histLen = d.dates.length, futLen = d.ml.future_dates.length, pad = Array(histLen - 1).fill(null), last = d.values[histLen - 1];
      const ext = arr => [...pad, last, ...arr];
      charts.forecast = new Chart(document.getElementById('forecastChart').getContext('2d'), {
        type: 'line',
        data: { labels: [...d.dates, ...d.ml.future_dates], datasets: [
          { label: `Observed (${u})`, data: [...d.values, ...Array(futLen).fill(null)], borderColor: '#38bdf8', borderWidth: 1.4, pointRadius: 0, tension: 0.2 },
          { label: 'Random Forest', data: ext(d.ml.predictions.random_forest), borderColor: '#34d399', borderWidth: 2, pointRadius: 0, tension: 0.3 },
          { label: 'Gradient Boosting', data: ext(d.ml.predictions.gradient_boosting), borderColor: '#c084fc', borderWidth: 2, pointRadius: 0, tension: 0.3 },
          { label: 'Seasonal Ridge', data: ext(d.ml.predictions.seasonal_ridge), borderColor: '#fbbf24', borderDash: [3, 3], borderWidth: 2, pointRadius: 0, tension: 0.3 },
          { label: 'Ensemble 95% band', data: ext(d.ml.predictions.lower_95), borderWidth: 0, pointRadius: 0, fill: false },
          { label: 'Ensemble 95% band ', data: ext(d.ml.predictions.upper_95), borderWidth: 0, pointRadius: 0, backgroundColor: 'rgba(251, 191, 36, 0.12)', fill: '-1' }
        ] },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
          plugins: { legend: { ...LEGEND, labels: { ...LEGEND.labels, filter: it => it.text !== 'Ensemble 95% band' } } },
          scales: { x: { ...AXIS, ticks: { ...AXIS.ticks, maxTicksLimit: 14 } }, y: AXIS } }
      });
    }

    /* ---------- export ---------- */
    function methodsParagraph(d) {
      const p = d.provenance, spatial = d.mode === 'region'
        ? `For a ${(d.box.north - d.box.south).toFixed(1)}° × ${(d.box.east - d.box.west).toFixed(1)}° region, monthly series were retrieved at the centres of a ${d.grid} × ${d.grid} grid (${d.cells_loaded} points loaded) and combined into a cos(latitude)-weighted regional mean.`
        : `A monthly time series was retrieved for the grid cell containing ${d.location.lat}°, ${d.location.lon}° (${d.place.label}).`;
      return `Data: ${p.variable} (${p.parameter}) from ${p.source}, ${d.period.start} to ${d.period.end}. ${spatial} ` +
        `Analysis: the tested series was ${d.series_description.toLowerCase()}. Linear trends were estimated with ordinary least squares and Sen's slope (Sen 1968; confidence interval after Gilbert 1987). ` +
        `Significance was assessed with the t-test on the OLS slope, the same test with an effective sample size accounting for lag-1 autocorrelation (Santer et al. 2000), ` +
        `the Mann-Kendall test (Mann 1945; Kendall 1975) and its Hamed-Rao (1998) autocorrelation-corrected variant${d.aggregation === 'monthly' ? ', and the Seasonal Mann-Kendall test (Hirsch et al. 1982)' : ''}, at α = ${d.alpha}. ` +
        `The record length needed to detect the trend was estimated following Weatherhead et al. (1998). Residuals were checked with the Durbin-Watson and Shapiro-Wilk tests. Analysis performed with Planet Pulse (NASA Space Apps 2026).`;
    }

    function copyMethods() {
      const text = lastScience ? methodsParagraph(lastScience) : '';
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(() => alert('Methods paragraph copied.'), () => alert('Copy failed: select the text and copy it manually.'));
    }

    function csvCell(v) {
      if (v === null || v === undefined) return '';
      const s = String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }

    function downloadScienceCsv() {
      const d = lastScience; if (!d) return;
      const p = d.provenance, lines = [], row = arr => lines.push(arr.map(csvCell).join(','));
      lines.push('# Planet Pulse scientific analysis export');
      row(['# variable', p.variable, p.parameter]); row(['# units', p.catalog_unit, p.api_units || '']);
      row(['# source', p.source]); row(['# request', p.url || '']);
      row(['# location', d.mode === 'region' ? JSON.stringify(d.box) : `${d.location.lat},${d.location.lon}`, d.place.label]);
      row(['# period', d.period.start, d.period.end]); row(['# alpha', d.alpha]); row(['# tested series', d.series_description]);
      lines.push(''); lines.push('## Monthly data'); row(['date', 'value', 'deseasonalized_anomaly']);
      d.raw.dates.forEach((dt, i) => row([dt, d.raw.values[i], d.raw.anomalies[i]]));
      if (d.aggregation !== 'monthly') { lines.push(''); lines.push('## Tested series'); row(['period', 'value', 'ols_fit', 'sen_fit']); d.series.labels.forEach((l, i) => row([l, d.series.values[i], d.series.ols_line[i], d.series.sen_line[i]])); }
      lines.push(''); lines.push('## Trend estimates (per decade)'); row(['method', 'per_decade', 'ci_low', 'ci_high', 'p', 'note']);
      d.estimates.forEach(e => row([e.method, e.per_decade, e.ci_low, e.ci_high, e.p, e.note]));
      lines.push(''); lines.push('## Significance tests'); row(['test', 'statistic', 'value', 'p', 'significant']);
      d.tests.forEach(t => row([t.test, t.statistic, t.value, t.p, t.significant]));
      lines.push(''); lines.push('## Detectability (Weatherhead 1998)'); row(['years_have', 'years_needed', 'noise_sd', 'phi']);
      row([d.detectability.years_have, d.detectability.years_needed, d.detectability.noise_sd, d.detectability.phi]);
      lines.push(''); lines.push('## Diagnostics'); row(Object.keys(d.diagnostics)); row(Object.values(d.diagnostics));
      lines.push(''); lines.push('## Seasonal breakdown'); row(['period', 'ols_per_decade', 'ols_p', 'sen_per_decade', 'mk_p', 'n_years', 'significant']);
      [...d.by_season, ...d.by_month].forEach(r => row([r.period, r.ols_per_decade, r.ols_p, r.sen_per_decade, r.mk_p, r.n, r.significant]));
      if (d.mode === 'region') { lines.push(''); lines.push('## Trend map points'); row(['lat', 'lon', 'status', 'per_decade', 'p_ols', 'p_mk_hamed_rao', 'significant']); d.cells.forEach(c => row([c.lat, c.lon, c.status, c.per_decade, c.p_ols, c.p_mk_hr, c.significant])); }
      lines.push(''); lines.push('## Methods'); row([methodsParagraph(d)]);
      const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `planet-pulse_${p.parameter}_${d.mode}_${d.period.start}_${d.period.end}.csv`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }

    /* =====================================================================
       START-UP
       ===================================================================== */
    document.addEventListener('DOMContentLoaded', async () => {
      initMap();
      moveTimeline('slot-investigate');
      const ts = document.getElementById('tl-start'), te = document.getElementById('tl-end');
      ts.addEventListener('input', () => { if (Number(ts.value) > Number(te.value) - minSpanYears + 1) ts.value = Number(te.value) - minSpanYears + 1; setYears(ts.value, te.value); });
      te.addEventListener('input', () => { if (Number(te.value) < Number(ts.value) + minSpanYears - 1) te.value = Number(ts.value) + minSpanYears - 1; setYears(ts.value, te.value); });
      ['sci-agg', 'sci-alpha'].forEach(id => document.getElementById(id).addEventListener('change', () => { if (sciScope === 'point' && lastAnalysis) runScience(); }));
      document.addEventListener('keydown', e => { if (e.key === 'Escape' && demo) demoStop(); });
      try { await loadVariables(); }
      catch (err) { document.getElementById('empty-state').innerHTML = `<div class="err-box">Could not load the variable list from the server: ${escHtml(err.message)}</div>`; }
      updateSteps();
    });
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)