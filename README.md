# TensorFlow Best Practice (Real-Time Scenario)

This mini codebase demonstrates **TensorFlow best practices** using a realistic use case:

## Use case
A customer-success team wants to estimate **churn risk in near real time** when an agent opens a customer profile.

- **Python + TensorFlow**: model training + inference API
- **Node.js**: lightweight gateway (frontend-friendly API layer)
- **React**: simple UI to test live predictions

---

## Why this is a best-practice starter

In `src/ml/train.py`, you get practical patterns:

1. **Reproducibility**
   - fixed random seed via `tf.keras.utils.set_random_seed(...)`
2. **Data leakage prevention**
   - normalization layer is `adapt(...)` on train split only
3. **Efficient input pipeline**
   - `tf.data` with `shuffle`, `batch`, `prefetch`
4. **Generalization controls**
   - dropout + early stopping
5. **Model lifecycle basics**
   - best checkpoint saved with `ModelCheckpoint`

---

## Project structure

- `src/ml/train.py` → train and save churn model
- `src/ml/serve.py` → Flask inference API (`/predict`)
- `src/index.js` → Node gateway (`/api/predict`)
- `src/frontend/*` → React app for quick testing

---

## Quick start

## 1) Train model (Python)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install tensorflow flask numpy
python src/ml/train.py
```

Artifacts saved under:
- `artifacts/churn_model.keras`
- `artifacts/norm_weights.npz`

## 2) Run TensorFlow API

```bash
python src/ml/serve.py
```

Runs at `http://localhost:8000`

## 3) Run Node gateway

```bash
npm init -y
npm install express cors
node src/index.js
```

Runs at `http://localhost:3001`

## 4) Run React frontend

```bash
cd src/frontend
npm install
npm run dev
```

Open `http://localhost:5173`

---

## Example prediction request

```bash
curl -X POST http://localhost:3001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_spend": 80,
    "support_tickets": 3,
    "contract_months": 6,
    "logins_per_week": 2
  }'
```

---

## Guidance for real production usage

To evolve this into production safely:

1. **Add feature validation**
   - enforce ranges/types at API boundary
2. **Track model versions**
   - version model artifacts and inference metadata
3. **Monitor drift**
   - compare live feature distribution vs training distribution
4. **Observe business KPIs**
   - calibration, precision@top-k risk buckets, retention uplift
5. **Set retraining cadence**
   - scheduled retraining + offline evaluation gate
6. **Add auth/rate-limit**
   - secure Node and Python APIs
7. **Containerize**
   - deploy as separate services with health checks

---

## Notes

- This is intentionally lightweight (not heavy code).
- Data is synthetic, but pipeline mirrors a real workflow.
- You can replace synthetic data with your own CSV/warehouse pipeline using the same structure.