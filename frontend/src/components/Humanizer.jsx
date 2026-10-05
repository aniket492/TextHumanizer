import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { useClipboard } from "../hooks/useClipboard";
import { api } from "../services/api";
import Alert from "./Alert";
import AnalysisPanel from "./AnalysisPanel";
import QualityReport from "./QualityReport";
import Spinner from "./Spinner";
import { INTENSITIES, MAX_WORDS, TONES, countWords } from "./options";

const DRAFT_KEY = "humanizer_draft";

function readDraft() {
  try {
    return sessionStorage.getItem(DRAFT_KEY) || "";
  } catch {
    return "";
  }
}

export default function Humanizer({ onComplete }) {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [text, setText] = useState(readDraft);
  const [tone, setTone] = useState("natural");
  const [intensity, setIntensity] = useState("medium");
  const [result, setResult] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const { copied, copy } = useClipboard();
  const abortRef = useRef(null);

  useEffect(() => {
    try {
      sessionStorage.setItem(DRAFT_KEY, text);
    } catch {
      /* ignore */
    }
  }, [text]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const words = countWords(text);
  const tooLong = words > MAX_WORDS;

  const handleHumanize = async (e) => {
    e.preventDefault();
    setError("");
    if (!text.trim()) return setError("Please enter some text to humanize.");
    if (tooLong) return setError(`Please shorten your text to ${MAX_WORDS.toLocaleString()} words or fewer.`);
    if (!user) return navigate("/login", { state: { from: "/dashboard", reason: "Log in to humanize text." } });

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setResult(null);
    setAnalysis(null);
    try {
      const data = await api.humanize(text, tone, intensity, { signal: controller.signal });
      setResult(data);
      onComplete?.(data);
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!result) return;
    setAnalyzing(true);
    setError("");
    try {
      setAnalysis(await api.analyze(result.output_text, text));
    } catch (err) {
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleClear = () => {
    abortRef.current?.abort();
    setText("");
    setResult(null);
    setAnalysis(null);
    setError("");
  };

  return (
    <div className="humanizer">
      <form className="card editor" onSubmit={handleHumanize} aria-busy={loading}>
        <label htmlFor="input-text" className="field-label">
          Your text
        </label>
        <textarea
          id="input-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste text that sounds stiff or mechanical..."
          rows={12}
          aria-describedby="word-count"
          disabled={loading}
        />
        <div className="editor-meta">
          <span id="word-count" className={tooLong ? "text-danger" : "muted"}>
            {words.toLocaleString()} / {MAX_WORDS.toLocaleString()} words
          </span>
        </div>

        <div className="controls">
          <div className="control">
            <label htmlFor="tone" className="field-label">
              Tone
            </label>
            <select id="tone" value={tone} onChange={(e) => setTone(e.target.value)} disabled={loading}>
              {TONES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <fieldset className="control">
            <legend className="field-label">Intensity</legend>
            <div className="segmented">
              {INTENSITIES.map((i) => (
                <label key={i.value} className={intensity === i.value ? "active" : ""} title={i.hint}>
                  <input
                    type="radio"
                    name="intensity"
                    value={i.value}
                    checked={intensity === i.value}
                    onChange={() => setIntensity(i.value)}
                    disabled={loading}
                  />
                  {i.label}
                </label>
              ))}
            </div>
          </fieldset>
        </div>

        <div className="actions">
          <button type="submit" className="btn btn-primary" disabled={loading || !text.trim()}>
            {loading ? (
              <>
                <span className="spinner spinner-sm" aria-hidden="true" /> Humanizing...
              </>
            ) : (
              "Humanize"
            )}
          </button>
          <button type="button" className="btn btn-ghost" onClick={handleClear} disabled={!text && !result}>
            Clear
          </button>
        </div>
        {loading && (
          <p className="muted small" role="status">
            Rewriting and checking meaning. Longer texts can take up to a minute on CPU.
          </p>
        )}
      </form>

      <Alert onClose={() => setError("")}>{error}</Alert>

      <section className="card output" aria-live="polite" aria-label="Humanized output">
        <div className="output-head">
          <h2>Output</h2>
          {result && (
            <div className="actions">
              <button className="btn btn-ghost btn-sm" onClick={() => copy(result.output_text)}>
                {copied ? "Copied!" : "Copy"}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={handleAnalyze} disabled={analyzing}>
                {analyzing ? "Analyzing..." : "Analyze output"}
              </button>
            </div>
          )}
        </div>
        {loading && <Spinner label="Humanizing your text" block />}
        {!loading && !result && <p className="muted">Your rewritten text will appear here.</p>}
        {result && (
          <>
            <div className="output-text">
              {result.output_text.split("\n\n").map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>
            <p className="muted small">
              {result.output_word_count} words &middot; {(result.processing_ms / 1000).toFixed(1)}s &middot; engine:{" "}
              {result.engine}
            </p>
            {result.notes.length > 0 && (
              <ul className="notes">
                {result.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            <QualityReport quality={result.quality} />
          </>
        )}
      </section>
      <AnalysisPanel analysis={analysis} />
    </div>
  );
}
