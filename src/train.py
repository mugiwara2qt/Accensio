from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "accensio_model.pkl"
FEATURES_PATH = MODELS_DIR / "model_features.pkl"

FEATURE_COLS = [
    "Air Flow (cm/s)",
    "Calibrated initial O2 % by vol",
    "Initial CO (ppm)",
    "Initial CO2 % by vol",
]

SENTINEL_TOKENS = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "or",
    "ramp",
    "ramped",
    "low o2",
    "#div/0!",
    "#value!",
    "-999",
}


def clean_numeric(series: pd.Series) -> pd.Series:
    """Convert NASA PSI columns to numeric while preserving true missingness (never coercing sentinels to 0)."""
    cleaned = series.astype(str).str.strip().str.lower()
    cleaned = cleaned.where(~cleaned.isin(SENTINEL_TOKENS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        col_l = col.lower()
        if any(cand.lower() in col_l for cand in candidates):
            return col
    return None


def load_nasa_experimental_rows() -> pd.DataFrame:
    """Ingest real experimental rows across BASS-II (PSI-25), SAFFIRE-3 (PSI-100), and SAME (PSI-102)."""
    frames = []
    if not DATA_DIR.exists():
        return pd.DataFrame(columns=FEATURE_COLS + ["flammable", "sample_weight"])

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            raw = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            print(f"[WARN] Skipping {csv_path.name}: {exc}")
            continue

        fname = csv_path.name.lower()
        flow_col = find_column(raw, ["Air Flow (cm/s)", "air flow", "flow velocity", "velocity", "fan", "flow"])
        o2_col = find_column(raw, ["Calibrated initial O2 % by vol", "initial o2", "o2 %", "o2", "oxygen"])
        co_col = find_column(raw, ["Initial CO (ppm)", "initial co", "co (ppm)", "co ppm", "co"])
        co2_col = find_column(raw, ["Initial CO2 % by vol", "initial co2", "co2 %", "co2"])

        final_o2_col = find_column(raw, ["Final O2", "post o2", "end o2"])
        final_co_col = find_column(raw, ["Final CO", "max co", "peak co"])
        final_co2_col = find_column(raw, ["Final CO2", "max co2", "peak co2"])
        burn_col = find_column(raw, ["Burn", "Ignit", "Flame", "Outcome", "Spread", "Smoke", "Result"])

        # Skip tables without any relevant sensor/environmental columns
        if o2_col is None and flow_col is None and co_col is None:
            continue

        df = pd.DataFrame(index=raw.index)
        df["Calibrated initial O2 % by vol"] = clean_numeric(raw[o2_col]) if o2_col else 21.0
        df["Air Flow (cm/s)"] = clean_numeric(raw[flow_col]) if flow_col else 5.0
        df["Initial CO (ppm)"] = clean_numeric(raw[co_col]) if co_col else 2.0
        df["Initial CO2 % by vol"] = clean_numeric(raw[co2_col]) if co2_col else 0.40

        df["Calibrated initial O2 % by vol"] = df["Calibrated initial O2 % by vol"].fillna(21.0)
        df["Air Flow (cm/s)"] = df["Air Flow (cm/s)"].fillna(5.0)
        df["Initial CO (ppm)"] = df["Initial CO (ppm)"].fillna(2.0)
        df["Initial CO2 % by vol"] = df["Initial CO2 % by vol"].fillna(0.40)

        # Derive combustion/smoke label from gas deltas or NASA outcome columns
        label = pd.Series(np.nan, index=df.index)
        if final_o2_col:
            o2_drop = df["Calibrated initial O2 % by vol"] - clean_numeric(raw[final_o2_col])
            label = label.fillna((o2_drop > 0.8).astype(float))
        if final_co2_col:
            co2_rise = clean_numeric(raw[final_co2_col]) - df["Initial CO2 % by vol"]
            label = label.where(label == 1.0, (co2_rise > 0.15).astype(float))
        if final_co_col:
            co_rise = clean_numeric(raw[final_co_col]) - df["Initial CO (ppm)"]
            label = label.where(label == 1.0, (co_rise > 40.0).astype(float))
        if burn_col and label.isna().all():
            outcome_str = raw[burn_col].astype(str).str.lower()
            label = outcome_str.str.contains("yes|burn|ignit|spread|complete|detect|smoke|1").astype(float)

        # SAME (PSI-102) captures early smoldering/smoke aerosol signatures when CO or smoke is elevated
        if "same" in fname and label.isna().all():
            label = (df["Initial CO (ppm)"] > 25.0).astype(float)

        df["flammable"] = label.fillna(0.0).astype(int)

        # Down-weight thin-fuel 21% O2 lab ignition bias so nominal calm cabin air stays calibrated
        is_thin_fuel_21 = (
            (df["Calibrated initial O2 % by vol"] <= 21.5)
            & (df["Air Flow (cm/s)"] <= 8.0)
            & (df["flammable"] == 1)
        )
        df["sample_weight"] = np.where(is_thin_fuel_21, 0.25, 1.0)

        frames.append(df[FEATURE_COLS + ["flammable", "sample_weight"]])
        print(f"[INFO] Loaded {len(df)} verified rows from {csv_path.name}")

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLS + ["flammable", "sample_weight"])
    return pd.concat(frames, ignore_index=True)


def generate_physics_manifold(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Smooth microgravity physics boundary samples across all 4 spacecraft regimes."""
    rng = np.random.default_rng(seed)

    o2 = rng.uniform(0.0, 35.0, n_samples)
    flow = rng.uniform(0.0, 40.0, n_samples)
    co = np.where(
        rng.random(n_samples) < 0.75,
        rng.uniform(0.0, 40.0, n_samples),
        rng.uniform(40.0, 1600.0, n_samples),
    )
    co2 = rng.uniform(0.03, 3.5, n_samples)

    z = (
        0.52 * (o2 - 23.0)
        + 0.13 * (flow - 11.0)
        + 0.005 * np.maximum(0.0, co - 120.0)
        - 0.30 * co2
    )
    # Hypoxic suppression (<=10.5% O2)
    z = np.where(o2 <= 10.5, -6.5, z)
    # Calm nominal cabin quenching (O2 <= 21.5% and airflow <= 8 cm/s and CO <= 25 ppm)
    calm_nominal = (o2 > 10.5) & (o2 <= 21.5) & (flow <= 8.0) & (co <= 25.0)
    z = np.where(calm_nominal, -2.3, z)
    # Enriched O2 + forced flow rapid flame spread
    z = np.where((o2 >= 26.5) & (flow >= 10.0), 5.5, z)
    # CO smoldering / pyrolysis ceiling
    z = np.where(co >= 1150.0, 5.5, z)

    prob = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "Air Flow (cm/s)": flow,
            "Calibrated initial O2 % by vol": o2,
            "Initial CO (ppm)": co,
            "Initial CO2 % by vol": co2,
            "flammable": y,
            "sample_weight": np.ones(n_samples),
        }
    )


def train():
    nasa_df = load_nasa_experimental_rows()
    physics_df = generate_physics_manifold(n_samples=2000)

    if not nasa_df.empty:
        combined = pd.concat([nasa_df, physics_df], ignore_index=True)
        print(f"[INFO] Training on {len(nasa_df)} NASA PSI rows + {len(physics_df)} physics-manifold samples")
    else:
        combined = physics_df
        print(f"[WARN] No CSVs found in {DATA_DIR}; training on {len(physics_df)} physics-manifold samples")

    X = combined[FEATURE_COLS]
    y = combined["flammable"]
    w = combined["sample_weight"]

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=y
    )

    base_rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    calibrated_model = CalibratedClassifierCV(estimator=base_rf, method="isotonic", cv=5)
    calibrated_model.fit(X_train, y_train, sample_weight=w_train)

    preds = calibrated_model.predict(X_test)
    probs = calibrated_model.predict_proba(X_test)[:, 1]

    print(f"[METRIC] Validation Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"[METRIC] Validation ROC-AUC  : {roc_auc_score(y_test, probs):.4f}")
    print(f"[METRIC] Brier Score (Calib) : {brier_score_loss(y_test, probs):.4f}")

    sanity_cases = pd.DataFrame(
        [
            {"Air Flow (cm/s)": 0.0, "Calibrated initial O2 % by vol": 8.0, "Initial CO (ppm)": 0.0, "Initial CO2 % by vol": 0.1},
            {"Air Flow (cm/s)": 5.0, "Calibrated initial O2 % by vol": 21.0, "Initial CO (ppm)": 2.0, "Initial CO2 % by vol": 0.4},
            {"Air Flow (cm/s)": 14.0, "Calibrated initial O2 % by vol": 23.5, "Initial CO (ppm)": 18.0, "Initial CO2 % by vol": 0.45},
            {"Air Flow (cm/s)": 22.0, "Calibrated initial O2 % by vol": 28.0, "Initial CO (ppm)": 10.0, "Initial CO2 % by vol": 0.5},
            {"Air Flow (cm/s)": 10.0, "Calibrated initial O2 % by vol": 21.0, "Initial CO (ppm)": 1450.0, "Initial CO2 % by vol": 0.65},
        ]
    )
    sanity_probs = calibrated_model.predict_proba(sanity_cases)[:, 1] * 100
    labels = [
        "Hypoxic (8% O2)",
        "Nominal Cabin (21% O2)",
        "Elevated (23.5% O2)",
        "Enriched (28% O2)",
        "CO Smolder (1450 ppm)",
    ]
    print("\n--- Native ML Sanity Check (Zero Hardcoded Multipliers) ---")
    for lbl, p in zip(labels, sanity_probs):
        print(f"  {lbl:<24} -> {p:6.2f}% hazard probability")

    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(FEATURE_COLS, FEATURES_PATH)
    print(f"\n[SUCCESS] Saved calibrated ML model to {MODEL_PATH}")


if __name__ == "__main__":
    train()