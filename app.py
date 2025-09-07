import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Loader2, Settings, Upload, Wand2, CheckCircle2, XCircle } from "lucide-react";

// === IMPORTANT ===
// This UI targets the common scikit-learn Breast Cancer Wisconsin feature set (30 features)
// and the notebook appears to use an SVC trained on load_breast_cancer().
// It sends features in the exact order of this FEATURES array.
// Configure your backend endpoint (FastAPI/Flask/etc.) to accept:
// {
//   feature_names: string[],
//   features: number[],
//   meta?: { source?: string }
// }
// and respond with something like:
// {
//   prediction: 0 | 1 | "benign" | "malignant",
//   proba?: number[] // [p_benign, p_malignant]
// }

const FEATURES = [
  "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
  "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
  "radius error","texture error","perimeter error","area error","smoothness error",
  "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
  "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
  "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension",
] as const;

const toKey = (name: string) => name.replace(/[^a-z0-9]+/gi, "_").toLowerCase();

export default function BreastCancerPredictorUI() {
  const [endpoint, setEndpoint] = useState<string>("/predict");
  const [values, setValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    FEATURES.forEach(f => (init[toKey(f)] = ""));
    return init;
  });
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<null | { prediction: string; proba?: number[] }>(null);
  const [error, setError] = useState<string | null>(null);
  const [bulk, setBulk] = useState("");

  const orderedVector = useMemo(() => FEATURES.map(f => Number(values[toKey(f)] ?? "")), [values]);
  const ready = useMemo(() => orderedVector.every(v => Number.isFinite(v)), [orderedVector]);

  function updateValue(key: string, raw: string) {
    // allow "" while typing; otherwise coerce to numeric-friendly string
    if (raw.trim() === "") return setValues(v => ({ ...v, [key]: "" }));
    const n = Number(raw);
    if (Number.isFinite(n)) setValues(v => ({ ...v, [key]: raw }));
  }

  function fillRandom() {
    // Reasonable randoms based on dataset rough scales
    const ranges: Record<string, [number, number]> = {
      mean_radius: [6, 28], mean_texture: [9, 40], mean_perimeter: [40, 190], mean_area: [140, 2500],
      mean_smoothness: [0.05, 0.2], mean_compactness: [0.0, 0.4], mean_concavity: [0.0, 0.5], mean_concave_points: [0.0, 0.3], mean_symmetry: [0.1, 0.4], mean_fractal_dimension: [0.04, 0.1],
      radius_error: [0.1, 3.0], texture_error: [0.2, 5.0], perimeter_error: [0.5, 25], area_error: [5, 550], smoothness_error: [0.001, 0.02],
      compactness_error: [0.0, 0.1], concavity_error: [0.0, 0.3], concave_points_error: [0.0, 0.07], symmetry_error: [0.005, 0.08], fractal_dimension_error: [0.001, 0.03],
      worst_radius: [7, 40], worst_texture: [10, 50], worst_perimeter: [50, 260], worst_area: [180, 4500], worst_smoothness: [0.07, 0.25],
      worst_compactness: [0.02, 1.5], worst_concavity: [0.02, 1.5], worst_concave_points: [0.0, 0.5], worst_symmetry: [0.1, 0.6], worst_fractal_dimension: [0.04, 0.2],
    };
    const next: Record<string, string> = {};
    FEATURES.forEach(f => {
      const k = toKey(f);
      const [lo, hi] = ranges[k] || [0, 1];
      const val = lo + Math.random() * (hi - lo);
      next[k] = String(Number(val.toFixed(4)));
    });
    setValues(next);
  }

  function clearAll() {
    const cleared: Record<string, string> = {};
    FEATURES.forEach(f => (cleared[toKey(f)] = ""));
    setValues(cleared);
    setResult(null);
    setError(null);
  }

  function parseBulkPaste(text: string) {
    // Accept a single CSV row or JSON array of 30 numbers
    try {
      const t = text.trim();
      if (t.startsWith("[")) {
        const arr = JSON.parse(t);
        if (Array.isArray(arr) && arr.length === FEATURES.length && arr.every(x => Number.isFinite(Number(x)))) {
          const next: Record<string, string> = {};
          FEATURES.forEach((f, i) => (next[toKey(f)] = String(arr[i])));
          setValues(next);
          return { ok: true };
        }
      } else {
        // CSV: either raw row or line with headers+row
        const lines = t.split(/\r?\n/).filter(Boolean);
        const last = lines[lines.length - 1];
        const parts = last.split(/,|\t|;|\s+/).filter(Boolean);
        if (parts.length === FEATURES.length && parts.every(x => Number.isFinite(Number(x)))) {
          const next: Record<string, string> = {};
          FEATURES.forEach((f, i) => (next[toKey(f)] = String(Number(parts[i]))));
          setValues(next);
          return { ok: true };
        }
      }
      return { ok: false, reason: "Paste 30 numeric values as JSON array or single CSV row." };
    } catch (e: any) {
      return { ok: false, reason: e?.message || "Invalid input" };
    }
  }

  async function predict() {
    setPending(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        feature_names: FEATURES,
        features: orderedVector,
        meta: { source: "ui-react" },
      };
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`Backend ${res.status} ${res.statusText}`);
      const data = await res.json();
      const pred = String(
        data.prediction === 0 ? "benign" : data.prediction === 1 ? "malignant" : data.prediction
      );
      setResult({ prediction: pred, proba: Array.isArray(data.proba) ? data.proba : undefined });
    } catch (e: any) {
      setError(e?.message || "Failed to predict");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">Breast Cancer SVC – Predictor</h1>
            <p className="text-gray-600 mt-1">Enter the 30 features used by the scikit-learn Breast Cancer dataset. The vector is sent in the exact order expected by most notebooks.</p>
          </div>
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-gray-500" />
            <Input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} className="w-[320px]" placeholder="https://your-api.example.com/predict" />
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="col-span-2 shadow-sm">
            <CardContent className="p-4 sm:p-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                {FEATURES.map((name) => {
                  const key = toKey(name);
                  return (
                    <div key={key} className="space-y-1">
                      <label className="text-sm font-medium text-gray-700">{name}</label>
                      <Input
                        inputMode="decimal"
                        placeholder="0"
                        value={values[key]}
                        onChange={(e) => updateValue(key, e.target.value)}
                      />
                    </div>
                  );
                })}
              </div>

              <div className="flex flex-wrap gap-2 mt-5">
                <Button variant="secondary" onClick={fillRandom}><Wand2 className="w-4 h-4 mr-2"/>Fill Random</Button>
                <Button variant="secondary" onClick={clearAll}><XCircle className="w-4 h-4 mr-2"/>Clear</Button>
                <Button onClick={predict} disabled={!ready || pending}>
                  {pending ? (<><Loader2 className="w-4 h-4 mr-2 animate-spin"/>Predicting…</>) : "Predict"}
                </Button>
              </div>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card className="shadow-sm">
              <CardContent className="p-4 sm:p-6 space-y-3">
                <h3 className="font-medium">Bulk paste</h3>
                <p className="text-sm text-gray-600">Paste a single CSV row or a JSON array of 30 numbers. We’ll map it to the correct order.</p>
                <Textarea value={bulk} onChange={(e) => setBulk(e.target.value)} placeholder="[17.99, 10.38, … 0.1189]" className="min-h-[120px]"/>
                <div className="flex gap-2">
                  <Button variant="secondary" onClick={() => {
                    const r = parseBulkPaste(bulk);
                    if (!r.ok) setError(r.reason || "Could not parse paste"); else setError(null);
                  }}><Upload className="w-4 h-4 mr-2"/>Use Values</Button>
                </div>
              </CardContent>
            </Card>

            <Card className="shadow-sm">
              <CardContent className="p-4 sm:p-6">
                <h3 className="font-medium mb-2">Result</h3>
                {!result && !error && (
                  <p className="text-sm text-gray-600">Run a prediction to see the output.</p>
                )}
                {error && (
                  <div className="text-sm text-red-600">{error}</div>
                )}
                {result && (
                  <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
                    <div className="flex items-center gap-2">
                      {String(result.prediction).toLowerCase().includes("malig") ? (
                        <XCircle className="w-5 h-5 text-red-600"/>
                      ) : (
                        <CheckCircle2 className="w-5 h-5 text-green-600"/>
                      )}
                      <div className="text-lg font-semibold">Prediction: {String(result.prediction)}</div>
                    </div>
                    {Array.isArray(result.proba) && result.proba.length >= 2 && (
                      <div className="text-sm text-gray-700">
                        Probabilities → Benign: {result.proba[0].toFixed?.(3) ?? result.proba[0]} · Malignant: {result.proba[1].toFixed?.(3) ?? result.proba[1]}
                      </div>
                    )}
                    <p className="text-xs text-gray-500">This tool is for demonstration only and not a medical device.</p>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        <Card className="shadow-sm">
          <CardContent className="p-4 sm:p-6">
            <h3 className="font-medium mb-2">How to hook up a backend</h3>
            <p className="text-sm text-gray-700">
              Point the endpoint to your API (default <code>/predict</code>). The API should scale/transform the vector exactly as your notebook did (e.g., StandardScaler) before calling <code>model.predict</code>.
              If you used a scikit-learn Pipeline, just call <code>pipeline.predict</code> directly.
            </p>
            <pre className="mt-3 text-xs bg-gray-900 text-gray-100 p-3 rounded-2xl overflow-auto">
{`// Example FastAPI (Python)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
pipe = joblib.load("model_or_pipeline.joblib")  # Pipeline[Sscaler -> SVC], or manually apply scaler

class Inp(BaseModel):
    feature_names: list[str]
    features: list[float]
    meta: dict | None = None

@app.post("/predict")
def predict(inp: Inp):
    x = np.array(inp.features, dtype=float).reshape(1, -1)
    prob = getattr(pipe, "predict_proba", None)
    y = int(pipe.predict(x)[0])
    resp = {"prediction": y}
    if prob is not None:
        p = prob(x)[0].tolist()
        # Ensure [p_benign, p_malignant] order if your model uses labels {0,1}
        resp["proba"] = p
    return resp
`}
            </pre>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
