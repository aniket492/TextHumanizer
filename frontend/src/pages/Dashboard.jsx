import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Alert from "../components/Alert";
import Humanizer from "../components/Humanizer";
import { INTENSITIES, TONES, formatDate, label } from "../components/options";
import { useAuth } from "../hooks/useAuth";
import { api } from "../services/api";

export default function Dashboard() {
  const { user } = useAuth();
  const [usage, setUsage] = useState(null);
  const [recent, setRecent] = useState([]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [u, h] = await Promise.all([api.usage(), api.history(1, 5)]);
      setUsage(u);
      setRecent(h.items);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <>
      <h1 className="page-title">Hi, {user.name.split(" ")[0]}</h1>
      <Alert onClose={() => setError("")}>{error}</Alert>
      <section className="stats" aria-label="Usage">
        <div className="card stat">
          <span className="stat-value">{usage?.request_count ?? "–"}</span>
          <span className="stat-label">Requests</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{usage?.words_processed?.toLocaleString() ?? "–"}</span>
          <span className="stat-label">Words processed</span>
        </div>
        <div className="card stat">
          <span className="stat-value">{usage?.history_count ?? "–"}</span>
          <span className="stat-label">Saved rewrites</span>
        </div>
      </section>

      <div className="dashboard-grid">
        <Humanizer onComplete={refresh} />
        <aside className="card recent" aria-label="Recent history">
          <div className="output-head">
            <h2>Recent</h2>
            <Link to="/history" className="small">View all</Link>
          </div>
          {recent.length === 0 ? (
            <p className="muted">No rewrites yet.</p>
          ) : (
            <ul className="history-list compact">
              {recent.map((item) => (
                <li key={item.id}>
                  <Link to={`/history/${item.id}`}>
                    <span className="history-preview">{item.preview}</span>
                    <span className="muted small">
                      {label(TONES, item.tone)} &middot; {label(INTENSITIES, item.intensity)} &middot;{" "}
                      {formatDate(item.created_at)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </div>
    </>
  );
}
