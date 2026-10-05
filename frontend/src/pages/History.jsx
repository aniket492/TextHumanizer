import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Alert from "../components/Alert";
import Spinner from "../components/Spinner";
import { INTENSITIES, TONES, formatDate, label } from "../components/options";
import { api } from "../services/api";

const PAGE_SIZE = 10;

export default function History() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [deleting, setDeleting] = useState(null);

  const load = useCallback(async (p) => {
    try {
      setData(await api.history(p, PAGE_SIZE));
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    load(page);
  }, [page, load]);

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this rewrite from your history? This cannot be undone.")) return;
    setDeleting(id);
    try {
      await api.deleteHistory(id);
      const remaining = (data?.items.length ?? 1) - 1;
      if (remaining === 0 && page > 1) setPage(page - 1);
      else await load(page);
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(null);
    }
  };

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <>
      <h1 className="page-title">History</h1>
      <Alert onClose={() => setError("")}>{error}</Alert>
      {!data && !error && <Spinner label="Loading history" block />}
      {data && data.items.length === 0 && (
        <div className="card center">
          <p>No saved rewrites yet.</p>
          <Link to="/dashboard" className="btn btn-primary">Humanize some text</Link>
        </div>
      )}
      {data && data.items.length > 0 && (
        <ul className="history-list">
          {data.items.map((item) => (
            <li key={item.id} className="card history-item">
              <div>
                <p className="history-preview">{item.preview}</p>
                <p className="muted small">
                  {label(TONES, item.tone)} &middot; {label(INTENSITIES, item.intensity)} &middot; {item.word_count} words
                  &middot; {formatDate(item.created_at)}
                </p>
              </div>
              <div className="actions">
                <Link to={`/history/${item.id}`} className="btn btn-ghost btn-sm">View</Link>
                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => handleDelete(item.id)}
                  disabled={deleting === item.id}
                >
                  {deleting === item.id ? "Deleting..." : "Delete"}
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {data && totalPages > 1 && (
        <nav className="pagination" aria-label="History pages">
          <button className="btn btn-ghost btn-sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
            Previous
          </button>
          <span className="muted">Page {page} of {totalPages}</span>
          <button className="btn btn-ghost btn-sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
            Next
          </button>
        </nav>
      )}
    </>
  );
}
