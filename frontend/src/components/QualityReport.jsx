const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
const num = (v, d = 1) => (typeof v === "number" ? v.toFixed(d) : "–");

export default function QualityReport({ quality }) {
  if (!quality) return null;
  const items = [
    { label: "Meaning preserved", value: pct(quality.semantic_similarity), hint: `Semantic similarity (${quality.similarity_method})` },
    { label: "Key details kept", value: pct(quality.entity_preservation), hint: "Names, numbers, dates, citations, URLs" },
    { label: "Factual consistency", value: pct(quality.factual_consistency), hint: "Numbers and negations unchanged" },
    { label: "Grammar check", value: pct(quality.grammar_score), hint: "Heuristic grammar and fluency score" },
    { label: "Readability", value: `${num(quality.readability_before)} → ${num(quality.readability_after)}`, hint: "Flesch reading ease (higher is easier)" },
    { label: "Repetition", value: `${pct(quality.repetition_before)} → ${pct(quality.repetition_after)}`, hint: "Share of text in repeated phrases" },
    { label: "Length", value: `${pct(quality.length_ratio)} of original`, hint: "Output words / input words" },
  ];
  return (
    <section className="quality" aria-label="Quality report">
      <div className="quality-head">
        <h3>Quality checks</h3>
        <span className={`badge ${quality.passed ? "badge-ok" : "badge-warn"}`}>
          {quality.passed ? "Passed" : "Review carefully"}
        </span>
      </div>
      <dl className="metric-grid">
        {items.map((it) => (
          <div className="metric" key={it.label} title={it.hint}>
            <dt>{it.label}</dt>
            <dd>{it.value}</dd>
          </div>
        ))}
      </dl>
      {quality.missing_items?.length > 0 && (
        <p className="muted small">
          Possibly changed details: {quality.missing_items.slice(0, 8).join(", ")}
        </p>
      )}
    </section>
  );
}
