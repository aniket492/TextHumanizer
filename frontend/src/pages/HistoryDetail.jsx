import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Alert from "../components/Alert";
import QualityReport from "../components/QualityReport";
import Spinner from "../components/Spinner";
import { INTENSITIES, TONES, formatDate, label } from "../components/options";
import { useClipboard } from "../hooks/useClipboard";
import { api } from "../services/api";

export default function HistoryDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [item, setItem] = useState(null);
  const [error, setError] = useState("");
  const { copied, copy } = useClipboard();

  useEffect(() => {
    api.historyItem(id).then(setItem).catch((err) => setError(err.message));
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm("Delete this rewrite from your history? This cannot be undone.")) return;
    try {
      await api.deleteHistory(id);
      navigate("/history", { replace: true });
    } catch (err) {
      setError(err.message);
    }
  };

  if (error) {
    return (
      <>
        <Alert>{error}</Alert>
        <Link to="/history">Back to history</Link>
      </>
    );
  }
  if (!item) return <Spinner label="Loading rewrite" block />;

  return (
    <>
      <Link to="/history" className="small">&larr; Back to history</Link>
      <h1 className="page-title">Rewrite #{item.id}</h1>
      <p className="muted">
        {label(TONES, item.tone)} &middot; {label(INTENSITIES, item.intensity)} &middot; {item.word_count} words &middot;{" "}
        {formatDate(item.created_at)}
        {item.quality?.engine && <> &middot; engine: {item.quality.engine}</>}
      </p>
      <div className="compare">
        <section className="card">
          <h2>Original</h2>
          <div className="output-text">
            {item.original_text.split("\n\n").map((p, i) => <p key={i}>{p}</p>)}
          </div>
        </section>
        <section className="card">
          <div className="output-head">
            <h2>Output</h2>
            <button className="btn btn-ghost btn-sm" onClick={() => copy(item.output_text)}>
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>
          <div className="output-text">
            {item.output_text.split("\n\n").map((p, i) => <p key={i}>{p}</p>)}
          </div>
        </section>
      </div>
      {item.quality && <div className="card"><QualityReport quality={item.quality} /></div>}
      <button className="btn btn-danger" onClick={handleDelete}>Delete</button>
    </>
  );
}
