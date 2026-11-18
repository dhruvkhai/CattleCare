import os
import sys
import json
import random
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

sys.path.append('src')
from predict import CattleCarePredictor

app = FastAPI(title="CattleCare AI Monitoring Platform")

predictor = CattleCarePredictor()

# Load test dataset sample for fast serving
candidates = [
    r"data\processed\hardware_compatible\behavior_training_windowed_1s.csv",
    r"data\processed\hardware_compatible\behavior_training_windowed_1s.csv.gz",
    r"data\processed\hardware_compatible\sample_behavior_data.csv"
]
DATA_PATH = None
for c in candidates:
    if os.path.exists(c):
        DATA_PATH = c
        break

if DATA_PATH and os.path.exists(DATA_PATH):
    df_data = pd.read_csv(DATA_PATH)
else:
    df_data = None

class SensorPacket(BaseModel):
    accel_x_mean: float = -4.2
    accel_x_std: float = 0.5
    accel_x_min: float = -5.0
    accel_x_max: float = -3.5
    accel_y_mean: float = 1.2
    accel_y_std: float = 0.8
    accel_y_min: float = 0.5
    accel_y_max: float = 2.0
    accel_z_mean: float = 8.5
    accel_z_std: float = 0.6
    accel_z_min: float = 7.5
    accel_z_max: float = 9.5
    accel_norm_mean: float = 9.8
    accel_norm_std: float = 0.7
    accel_norm_max: float = 10.5
    pitch_mean: float = -62.0
    pitch_std: float = 5.2
    pitch_min: float = -70.0
    pitch_max: float = -55.0
    roll_mean: float = -5.0
    roll_std: float = 3.0
    collar_temp_C: float = 27.5
    barn_temp_C: float = 24.0
    barn_humidity_per: float = 65.0
    THI: float = 73.5

@app.get("/api/cows")
def get_cows():
    if df_data is None:
        return {"cows": ["C09", "C10"]}
    return {"cows": sorted(df_data['cow_id'].unique().tolist())}

@app.get("/api/cow_stats/{cow_id}")
def get_cow_stats(cow_id: str):
    if df_data is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    cow_df = df_data[df_data['cow_id'] == cow_id]
    if len(cow_df) == 0:
        raise HTTPException(status_code=404, detail="Cow not found")
        
    counts = cow_df['behavior_name'].value_counts().to_dict()
    total = len(cow_df)
    return {
        "cow_id": cow_id,
        "total_records": total,
        "behavior_breakdown": {k: round(v / total * 100, 1) for k, v in counts.items()},
        "avg_thi": round(float(cow_df['THI'].mean()), 1),
        "avg_pitch": round(float(cow_df['pitch_mean'].mean()), 1),
        "avg_collar_temp": round(float(cow_df['collar_temp_C'].mean()), 1)
    }

@app.get("/api/stream/{cow_id}")
def stream_sample(cow_id: str, count: int = 1):
    if df_data is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    cow_df = df_data[df_data['cow_id'] == cow_id]
    if len(cow_df) == 0:
        raise HTTPException(status_code=404, detail="Cow not found")
        
    samples = cow_df.sample(min(count, len(cow_df)))
    results = []
    for _, row in samples.iterrows():
        pred = predictor.predict_sample(row.to_dict())
        results.append({
            "timestamp": int(row['timestamp']),
            "cow_id": cow_id,
            "ground_truth": row['behavior_name'],
            "pitch": round(float(row['pitch_mean']), 1),
            "accel_norm": round(float(row['accel_norm_mean']), 2),
            "prediction": pred
        })
    return results

@app.post("/api/predict")
def predict_custom(packet: SensorPacket):
    res = predictor.predict_sample(packet.dict())
    return res

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CattleCare - AI Livestock Monitoring Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen p-6 font-sans">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center bg-slate-800 p-6 rounded-2xl border border-slate-700 shadow-xl gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="text-3xl">🐄</span>
                    <h1 class="text-2xl font-bold tracking-tight text-emerald-400">CattleCare AI Edge Monitor</h1>
                </div>
                <p class="text-sm text-slate-400 mt-1">Real-time Behavior Recognition & Microclimate Health Analytics (ESP32 + MPU6050 + DHT11)</p>
            </div>
            <div class="flex items-center gap-3">
                <label class="text-sm font-semibold text-slate-300">Select Cow:</label>
                <select id="cowSelect" class="bg-slate-900 border border-slate-600 rounded-lg px-4 py-2 font-mono text-emerald-400 text-sm focus:outline-none focus:border-emerald-500">
                    <option value="C09">Cow C09 (Unseen Test)</option>
                    <option value="C10">Cow C10 (Unseen Test)</option>
                    <option value="C01">Cow C01 (Training)</option>
                    <option value="C02">Cow C02 (Training)</option>
                    <option value="C03">Cow C03 (Training)</option>
                </select>
                <button onclick="fetchSample()" class="bg-emerald-600 hover:bg-emerald-500 px-4 py-2 rounded-lg font-medium text-sm transition shadow-lg shadow-emerald-900/40">
                    ⚡ Live Step
                </button>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
                <div class="text-xs text-slate-400 uppercase font-semibold">Predicted Behavior</div>
                <div id="kpiBehavior" class="text-2xl font-black text-emerald-400 mt-2">--</div>
                <div id="kpiConfidence" class="text-xs text-slate-400 mt-1">Confidence: --%</div>
            </div>
            <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
                <div class="text-xs text-slate-400 uppercase font-semibold">Neck Pitch Angle</div>
                <div id="kpiPitch" class="text-2xl font-black text-cyan-400 mt-2">--°</div>
                <div id="kpiPitchStatus" class="text-xs text-slate-400 mt-1">Biomechanical Tilt</div>
            </div>
            <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
                <div class="text-xs text-slate-400 uppercase font-semibold">Movement Intensity</div>
                <div id="kpiNorm" class="text-2xl font-black text-amber-400 mt-2">-- m/s²</div>
                <div class="text-xs text-slate-400 mt-1">MPU6050 Dynamic Norm</div>
            </div>
            <div class="bg-slate-800 p-5 rounded-xl border border-slate-700">
                <div class="text-xs text-slate-400 uppercase font-semibold">Barn THI & Heat Stress</div>
                <div id="kpiTHI" class="text-2xl font-black text-rose-400 mt-2">--</div>
                <div id="kpiStress" class="text-xs text-slate-400 mt-1">Status: --</div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div class="lg:col-span-2 bg-slate-800 p-6 rounded-2xl border border-slate-700 space-y-4">
                <h2 class="text-lg font-bold text-slate-200">Live Collar Telemetry Log</h2>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="bg-slate-900/60 text-slate-400 text-xs uppercase font-mono border-b border-slate-700">
                            <tr>
                                <th class="p-3">Time</th>
                                <th class="p-3">Pitch</th>
                                <th class="p-3">Movement</th>
                                <th class="p-3">Ground Truth</th>
                                <th class="p-3">Prediction</th>
                                <th class="p-3">Confidence</th>
                            </tr>
                        </thead>
                        <tbody id="telemetryBody" class="divide-y divide-slate-700/50 font-mono text-xs">
                            <tr><td colspan="6" class="p-4 text-center text-slate-500">Click "Live Step" to stream sensor data...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="bg-slate-800 p-6 rounded-2xl border border-slate-700 flex flex-col justify-between">
                <h2 class="text-lg font-bold text-slate-200 mb-4">Daily Behavioral Time-Budget</h2>
                <div class="relative w-full aspect-square max-h-72 mx-auto">
                    <canvas id="budgetChart"></canvas>
                </div>
                <div id="budgetLegend" class="mt-4 text-xs text-slate-400 space-y-1"></div>
            </div>
        </div>
    </div>

    <script>
        let chartInstance = null;

        async function updateStats() {
            const cow = document.getElementById('cowSelect').value;
            const res = await fetch(`/api/cow_stats/${cow}`);
            if (!res.ok) return;
            const data = await res.json();
            
            const labels = Object.keys(data.behavior_breakdown);
            const values = Object.values(data.behavior_breakdown);
            
            if (chartInstance) chartInstance.destroy();
            const ctx = document.getElementById('budgetChart').getContext('2d');
            chartInstance = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: ['#10b981', '#3b82f6', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#64748b'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } }
                }
            });

            const leg = document.getElementById('budgetLegend');
            leg.innerHTML = labels.map((l, i) => `
                <div class="flex justify-between py-0.5">
                    <span class="text-slate-300 font-medium">${l}</span>
                    <span class="font-mono text-emerald-400">${values[i]}%</span>
                </div>
            `).join('');
        }

        async function fetchSample() {
            const cow = document.getElementById('cowSelect').value;
            const res = await fetch(`/api/stream/${cow}?count=1`);
            if (!res.ok) return;
            const data = await res.json();
            if (data.length === 0) return;
            const s = data[0];

            document.getElementById('kpiBehavior').innerText = s.prediction.predicted_behavior;
            document.getElementById('kpiConfidence').innerText = `Confidence: ${s.prediction.confidence}%`;
            document.getElementById('kpiPitch').innerText = `${s.pitch}°`;
            document.getElementById('kpiPitchStatus').innerText = s.pitch < -45 ? 'Feeding / Head Down' : (s.pitch > -15 ? 'Resting / Lying' : 'Upright Standing');
            document.getElementById('kpiNorm').innerText = `${s.accel_norm} m/s²`;
            document.getElementById('kpiTHI').innerText = `${s.prediction.environmental_status.THI}`;
            document.getElementById('kpiStress').innerText = `Status: ${s.prediction.environmental_status.heat_stress_level}`;

            const tbody = document.getElementById('telemetryBody');
            const row = document.createElement('tr');
            const match = s.ground_truth === s.prediction.predicted_behavior;
            row.innerHTML = `
                <td class="p-3 text-slate-400">${new Date(s.timestamp * 1000).toLocaleTimeString()}</td>
                <td class="p-3 font-semibold text-cyan-300">${s.pitch}°</td>
                <td class="p-3 text-amber-300">${s.accel_norm}</td>
                <td class="p-3 text-slate-300">${s.ground_truth}</td>
                <td class="p-3 font-bold ${match ? 'text-emerald-400' : 'text-rose-400'}">${s.prediction.predicted_behavior}</td>
                <td class="p-3">${s.prediction.confidence}% ${match ? '✓' : '✗'}</td>
            `;
            if (tbody.children[0] && tbody.children[0].children.length === 1) {
                tbody.innerHTML = '';
            }
            tbody.insertBefore(row, tbody.firstChild);
            if (tbody.children.length > 8) tbody.removeChild(tbody.lastChild);
        }

        document.getElementById('cowSelect').addEventListener('change', () => {
            updateStats();
            document.getElementById('telemetryBody').innerHTML = '<tr><td colspan="6" class="p-4 text-center text-slate-500">Switched cow. Click "Live Step" to stream...</td></tr>';
        });

        updateStats();
        fetchSample();
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
