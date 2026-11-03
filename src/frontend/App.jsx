import { useState } from "react";

const initial = {
  monthly_spend: 80,
  support_tickets: 3,
  contract_months: 6,
  logins_per_week: 2,
};

export default function App() {
  const [form, setForm] = useState(initial);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onChange = (key, value) => {
    setForm((p) => ({ ...p, [key]: Number(value) }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://localhost:3001/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <main style={{ fontFamily: "Arial", maxWidth: 560, margin: "2rem auto" }}>
      <h2>Customer Churn Risk Checker</h2>
      <p>TensorFlow best-practice demo for near real-time support workflow.</p>

      <form onSubmit={onSubmit}>
        {Object.keys(form).map((k) => (
          <label key={k} style={{ display: "block", marginBottom: 10 }}>
            {k}
            <input
              type="number"
              value={form[k]}
              onChange={(e) => onChange(k, e.target.value)}
              style={{ marginLeft: 10 }}
            />
          </label>
        ))}
        <button disabled={loading}>{loading ? "Predicting..." : "Predict Churn"}</button>
      </form>

      {result && (
        <pre style={{ marginTop: 16, background: "#f5f5f5", padding: 12 }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </main>
  );
}