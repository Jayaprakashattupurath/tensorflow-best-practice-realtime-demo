const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

const PY_API_URL = process.env.PY_API_URL || "http://127.0.0.1:8000";

app.get("/health", async (_, res) => {
  res.json({ status: "ok", gateway: "node" });
});

app.post("/api/predict", async (req, res) => {
  try {
    const response = await fetch(`${PY_API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (err) {
    res.status(500).json({ error: "Prediction service unavailable", details: err.message });
  }
});

app.listen(3001, () => {
  console.log("Node gateway running on http://localhost:3001");
});