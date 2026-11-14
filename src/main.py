import json
import os
from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)

try:
    from google import genai
except ImportError:
    genai = None

MODEL_PATH = ROOT_DIR / "models" / "accensio_model.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "model_features.pkl"
DATA_DIR = ROOT_DIR / "data"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if (genai and GEMINI_API_KEY) else None

app = FastAPI(
    title="ACCENSIO // Spacecraft Microgravity Fire Super Intelligence",
    description="Calibrated ML + Counterfactual SI Optimizer + Gemini Grounded Synthesis on NASA PSI datasets.",
    version="3.1.2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_features: list[str] = []
model_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model = joblib.load(MODEL_PATH)
        model_features = joblib.load(FEATURES_PATH)
    else:
        model_load_error = "Model artifacts missing in models/. Run src/train.py first."
except Exception as exc:
    model_load_error = str(exc)

DATASET_PROVENANCE = {
    "PSI-25": {
        "name": "BASS-II (Burning and Suppression of Solids - II)",
        "file": "PSI-25_Experimental table_BASS-II.csv",
        "role": "Primary ignition, oxygen-drop, and forced-convection flammability baseline",
        "limitation": "Tests thin solid fuels in small flow ducts; high positive ignition bias in 21% O2 lab setups.",
    },
    "PSI-100": {
        "name": "SAFFIRE-3 (Spacecraft Fire Experiment - III)",
        "file": "PSI-100_Experimental table_SAFFIRE-3.csv",
        "role": "Large-scale spacecraft duct airflow, oxygen concentration, and flame-spread context",
        "limitation": "Engineering sensor telemetry; duct geometry differs from open habitat modules.",
    },
    "PSI-102": {
        "name": "SAME (Smoke Aerosol Measurement Experiment)",
        "file": "PSI-102_Experimental table_SAME.csv",
        "role": "Early smoldering, pyrolysis products (CO/CO2), and smoke-detector response",
        "limitation": "Measures aerosol/smoke signatures rather than direct flame spread rates.",
    },
}

MATERIAL_PROFILES = {
    "general_cabin": {
        "label": "Standard Spacecraft Cabin Composite",
        "risk_multiplier": 1.00,
        "moc_o2": 20.5,
        "source": "NASA PSI-25 / PSI-100 Composite Baseline",
    },
    "cotton_cellulose": {
        "label": "Cotton / Cellulosic Fabric (BASS-II)",
        "risk_multiplier": 1.15,
        "moc_o2": 17.5,
        "source": "NASA PSI-25 (BASS-II Thin Cellulosic Fuels)",
    },
    "pmma_acrylic": {
        "label": "PMMA Acrylic / Cast Polymer (SAFFIRE-3)",
        "risk_multiplier": 1.10,
        "moc_o2": 18.0,
        "source": "NASA PSI-100 (SAFFIRE-3 PMMA Slab Tests)",
    },
    "kapton_wire": {
        "label": "Kapton / PTFE Wire Insulation",
        "risk_multiplier": 0.85,
        "moc_o2": 24.0,
        "source": "NASA Avionics Wire Insulation Flammability Standard",
    },
    "nomex_kevlar": {
        "label": "Nomex / Flame-Retardant Crew Gear",
        "risk_multiplier": 0.70,
        "moc_o2": 28.0,
        "source": "NASA Crew Garment Flame-Retardant Benchmark",
    },
}

class EnvironmentInput(BaseModel):
    zone_name: str = Field("Destiny Lab", min_length=1, max_length=80)
    material_type: str = Field("general_cabin")
    air_flow: float = Field(..., ge=0.0, le=200.0)
    initial_o2: float = Field(..., ge=0.0, le=100.0)
    initial_co: float = Field(..., ge=0.0, le=50000.0)
    initial_co2: float = Field(..., ge=0.0, le=100.0)

class MultiZoneRequest(BaseModel):
    zones: List[EnvironmentInput] = Field(..., min_length=1, max_length=25)

class AssistantQuery(BaseModel):
    question: str = Field(..., min_length=2, max_length=600)
    current_telemetry: Optional[EnvironmentInput] = None

def call_gemini_grounded(prompt: str) -> tuple[Optional[str], str]:
    if gemini_client is None:
        return None, "calibrated-ml-fallback"
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return response.text.strip(), "gemini-2.5-flash"
    except Exception as exc:
        print(f"[GEMINI WARN] {exc}")
    return None, "calibrated-ml-fallback"

def raw_ml_probability(air_flow: float, o2: float, co_ppm: float, co2: float, multiplier: float) -> float:
    if o2 <= 10.0:
        return 0.0
    if (o2 >= 26.5 and air_flow >= 10.0) or o2 >= 30.0 or co_ppm >= 1200.0:
        return 99.0

    if model is None or not model_features:
        raise HTTPException(status_code=503, detail=model_load_error or "Model unavailable.")

    input_df = pd.DataFrame([[0.0] * len(model_features)], columns=model_features)
    feature_map = {
        "Air Flow (cm/s)": air_flow,
        "Calibrated initial O2 % by vol": o2,
        "Initial CO (ppm)": co_ppm,
        "Initial CO2 % by vol": co2,
    }
    for col, val in feature_map.items():
        if col in input_df.columns:
            input_df[col] = val

    probs = model.predict_proba(input_df)[0]
    base_prob = (
        float(probs[1]) * 100.0
        if len(probs) > 1
        else (100.0 if model.predict(input_df)[0] == 1 else 0.0)
    )
    return round(min(99.0, max(0.0, base_prob * multiplier)), 2)

def compute_si_counterfactuals(data: EnvironmentInput, current_prob: float, multiplier: float) -> list[str]:
    if current_prob < 20.0:
        return [f"Nominal envelope confirmed ({current_prob:.1f}% risk). Maintain continuous ECLSS sensor polling."]

    candidates = []

    for target_flow in [8.0, 5.0, 2.0, 0.0]:
        if target_flow < data.air_flow:
            p_new = raw_ml_probability(target_flow, data.initial_o2, data.initial_co, data.initial_co2, multiplier)
            drop = round(current_prob - p_new, 1)
            if drop >= 5.0:
                candidates.append((drop, f"SI OPTIMAL ACTUATOR: Throttle HVAC airflow from {data.air_flow:.1f} -> {target_flow:.1f} cm/s (cuts risk by -{drop}% to {p_new:.1f}%)."))
                break

    for target_o2 in [22.5, 21.0, 19.5]:
        if target_o2 < data.initial_o2:
            p_new = raw_ml_probability(data.air_flow, target_o2, data.initial_co, data.initial_co2, multiplier)
            drop = round(current_prob - p_new, 1)
            if drop >= 5.0:
                candidates.append((drop, f"SI ATMOSPHERE DILUTION: Inject N2 to reduce O2 from {data.initial_o2:.1f}% -> {target_o2:.1f}% (cuts risk by -{drop}% to {p_new:.1f}%)."))
                break

    if data.initial_co > 25.0:
        p_new = raw_ml_probability(data.air_flow, data.initial_o2, 10.0, data.initial_co2, multiplier)
        drop = round(current_prob - p_new, 1)
        if drop >= 5.0:
            candidates.append((drop, f"SI PYROLYSIS SCRUB: Isolate avionics bus & activate TCCS CO scrubber to drop CO from {data.initial_co:.0f} -> 10 ppm (cuts risk by -{drop}% to {p_new:.1f}%)."))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in candidates[:3]] or ["Isolate compartment ventilation and don emergency breathing apparatus (PBA)."]

def score_zone(data: EnvironmentInput) -> dict:
    mat = MATERIAL_PROFILES.get(data.material_type, MATERIAL_PROFILES["general_cabin"])
    final_prob = raw_ml_probability(data.air_flow, data.initial_o2, data.initial_co, data.initial_co2, mat["risk_multiplier"])

    is_guardrail = (
        data.initial_o2 <= 10.0
        or (data.initial_o2 >= 26.5 and data.air_flow >= 10.0)
        or data.initial_o2 >= 30.0
        or data.initial_co >= 1200.0
    )
    engine_mode = "physics_guardrail" if is_guardrail else "calibrated_random_forest"

    if final_prob >= 75.0:
        status = "HAZARDOUS"
    elif final_prob >= 45.0:
        status = "FLAMMABLE"
    elif final_prob >= 20.0:
        status = "CAUTION"
    else:
        status = "SAFE"

    mitigations = compute_si_counterfactuals(data, final_prob, mat["risk_multiplier"])
    interpretation = f"{status} ({final_prob}%): {mat['label']} (MOC {mat['moc_o2']}% O2) evaluated at {data.initial_o2:.1f}% O2, {data.air_flow:.1f} cm/s forced flow, and {data.initial_co:.0f} ppm CO under microgravity."

    return {
        "zone_name": data.zone_name,
        "material": mat["label"],
        "material_source": mat["source"],
        "engine_mode": engine_mode,
        "status": status,
        "hazard_probability": final_prob,
        "message": interpretation,
        "interpretation": interpretation,
        "mitigation_actions": mitigations,
        "telemetry": data.model_dump(),
    }

@app.get("/")
def read_root():
    return {
        "system": "ACCENSIO // Spacecraft Microgravity Fire Super Intelligence",
        "version": "3.1.2",
        "ai_layers": ["Calibrated Random Forest (0.9709 ROC-AUC)", "SI Counterfactual Optimizer", "Gemini 2.5 Flash Grounded Synthesis"],
        "datasets": list(DATASET_PROVENANCE.keys()),
    }

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/ready")
def readiness_check():
    csv_files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    return {
        "ready": model is not None and len(model_features) > 0,
        "model_loaded": model is not None,
        "feature_count": len(model_features),
        "datasets_present": len(csv_files),
        "gemini_connected": gemini_client is not None,
        "llm_provider": "gemini-2.5-flash" if gemini_client is not None else "offline-fallback",
        "error": model_load_error,
    }

@app.post("/predict")
def predict_flammability(data: EnvironmentInput):
    return score_zone(data)

@app.post("/dashboard/analyze")
def analyze_spacecraft_dashboard(payload: MultiZoneRequest):
    evaluated = [score_zone(z) for z in payload.zones]
    ranked = sorted(evaluated, key=lambda x: x["hazard_probability"], reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["risk_rank"] = idx

    critical_count = sum(1 for z in ranked if z["status"] in {"HAZARDOUS", "FLAMMABLE"})
    caution_count = sum(1 for z in ranked if z["status"] == "CAUTION")
    top_zone = ranked[0]

    base_summary = (
        f"ALERT: {critical_count}/{len(ranked)} compartments elevated. Top priority: {top_zone['zone_name']} "
        f"({top_zone['hazard_probability']}% — {top_zone['status']})."
        if critical_count > 0
        else (
            f"ADVISORY: {caution_count} compartment(s) in CAUTION. Highest sensitivity: {top_zone['zone_name']} ({top_zone['hazard_probability']}%)."
            if caution_count > 0
            else f"ALL NOMINAL: All {len(ranked)} spacecraft compartments within safe microgravity envelopes."
        )
    )

    return {
        "executive_summary": base_summary,
        "fleet_metrics": {
            "total_zones": len(ranked),
            "critical_or_flammable": critical_count,
            "caution": caution_count,
            "safe": len(ranked) - critical_count - caution_count,
            "max_hazard_probability": top_zone["hazard_probability"],
        },
        "ranked_zones": ranked,
    }

@app.post("/assistant/ask")
def ask_safety_assistant(query: AssistantQuery):
    citations = ["NASA PSI-25 (BASS-II)", "NASA PSI-100 (SAFFIRE-3)", "NASA PSI-102 (SAME)"]
    live_summary = "No active compartment telemetry provided."

    if query.current_telemetry is not None:
        live = score_zone(query.current_telemetry)
        live_summary = (
            f"Zone: {live['zone_name']} | Material: {live['material']} | ML Score: {live['hazard_probability']}% ({live['status']}) | "
            f"O2={query.current_telemetry.initial_o2}%, Flow={query.current_telemetry.air_flow} cm/s, "
            f"CO={query.current_telemetry.initial_co} ppm, CO2={query.current_telemetry.initial_co2}% | "
            f"SI Optimal Interventions: {' ; '.join(live['mitigation_actions'])}"
        )
        citations.insert(0, live["material_source"])

    unique_citations = list(dict.fromkeys(citations))
    prompt = (
        "You are ACCENSIO, a NASA Spacecraft Microgravity Fire Super Intelligence assistant. "
        "Answer concisely in 3-4 sentences using ONLY the live ML telemetry, SI counterfactual actions, and NASA PSI provenance below.\n\n"
        f"LIVE TELEMETRY & SI COUNTERFACTUALS:\n{live_summary}\n\n"
        f"NASA PSI PROVENANCE:\n{json.dumps(DATASET_PROVENANCE)}\n\n"
        f"QUESTION: {query.question}"
    )

    llm_answer, provider = call_gemini_grounded(prompt)
    if llm_answer is None:
        llm_answer = (
            f"{live_summary} Under 0g, buoyancy is absent so forced HVAC airflow (cm/s) and O2 (% vol) govern flame spread "
            "(NASA PSI-25/PSI-100), while CO (ppm) tracks early avionics pyrolysis (NASA PSI-102)."
        )

    return {
        "question": query.question,
        "answer": llm_answer,
        "citations": unique_citations,
        "provider": provider,
        "grounded": True,
    }

# Serve frontend automatically if hosted together
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    @app.get("/app", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)