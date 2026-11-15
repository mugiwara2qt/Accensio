# SPACE STORIES: ACCENSIO
**Spacecraft Microgravity Fire Safety AI & 0g Combustion Physics Simulator**
Built for the **NASA Space Apps Challenge**, **ACCENSIO** is a real-time microgravity combustion simulator and spacecraft cabin risk assessment platform. It combines an isotonic-calibrated machine learning model trained on open NASA Physical Sciences Informatics (PSI) datasets with a 60 FPS fluid/flame visualizer and a Gemini-powered safety assistant.

---

## Why Microgravity Fire Safety Matters
On Earth, hot air rises due to buoyancy, pulling fresh oxygen into the base of a flame and carrying smoke up to ceiling detectors. In zero gravity ($g = 0\text{ m/s}^2$), buoyancy disappears:
* **Hidden Diffusive Flames:** In still cabin air ($Pe < 35$), flames form dim, near-invisible spherical blue balls that burn slowly without triggering standard smoke alarms.
* **Convective Blowtorches:** When cabin ventilation (HVAC) fans blow across a heated surface or oxygen levels increase during EVA prep, flames stretch into fast-moving convective wakes.
* **Silent Wire Pyrolysis:** Overheating electrical insulation (like Kapton) can smolder behind avionics panels and release toxic Carbon Monoxide ($CO$) long before open flames appear.

ACCENSIO lets operators test these exact scenarios across five station compartments and get immediate, actionable steps to stabilize the cabin.

---

## Key Features
* **60 FPS 0g Physics Chamber:** Real-time HTML5 Canvas simulation showing flame morphology morphing from spherical diffusion to convective wakes as airflow ($U$) and $O_2$ sliders move, complete with live Péclet number ($Pe$), spread rate ($V_f$), flame temperature ($T_f$), and Heat Release Rate ($HRR$) readouts.
* **Instant ML Hazard Verdicts:** Evaluates cabin telemetry across 5 real spacecraft materials (`Cabin Composite`, `Cotton/Cellulose`, `PMMA Acrylic`, `Kapton Wire`, and `Nomex`) using a calibrated Random Forest classifier.
* **Counterfactual Crew Interventions & Auto-Stabilizer:** Recommends specific fixes—such as throttling local HVAC fans, isolating $O_2$ supply lines, or activating charcoal scrubbers—with a one-click **SI Auto-Stabilize** control.
* **Station-Wide Risk Ranking:** Remembers independent telemetry states across 5 spacecraft compartments (`Destiny Lab`, `Quest Airlock`, `Avionics Bay 2`, `Crew Habitation`, and `Cargo Vault`) and ranks them by live hazard priority.
* **ACCENSIO Super Intelligence (SI):** Integrated `gemini-2.5-flash` assistant grounded in NASA PSI research that answers crew questions using live cabin sensor context.

---

## System Architecture & Tech Stack
* **Frontend:** HTML5 Canvas, CSS3, hosted on **Netlify** (`https://accensio-ai.netlify.app`).
* **Backend & ML Engine:** Python, FastAPI, Uvicorn, Scikit-Learn (Isotonic-Calibrated Random Forest), NumPy, Joblib, hosted on **Render** (`https://accensio.onrender.com`).
* **AI Assistant:** Google GenAI SDK (`gemini-2.5-flash`) with multi-key rotation fallback.
* **NASA Open-Science Datasets ([psi.nasa.gov](https://psi.nasa.gov)):**
  * **PSI-25 (BASS-II):** Burning and Suppression of Solids – II
  * **PSI-100 (SAFFIRE-III):** Spacecraft Fire Experiment – III
  * **PSI-102 (SAME):** Smoke and Aerosol Measurement Experiment
