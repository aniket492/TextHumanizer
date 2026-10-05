export default function AnalysisPanel({ analysis }) {
  if (!analysis) return null;
  const stats = [
    ["Words", analysis.word_count],
    ["Sentences", analysis.sentence_count],
    ["Paragraphs", analysis.paragraph_count],
    ["Avg. sentence length", analysis.average_sentence_length],
    ["Sentence-length variation", analysis.sentence_length_variation],
    ["Vocabulary diversity", analysis.vocabulary_diversity],
    ["Readability", `${analysis.readability_score} (${analysis.readability.label})`],
    ["Grade level", analysis.readability.flesch_kincaid_grade],
    ["Repetition score", analysis.repetition_score],
  ];
  if (analysis.semantic_similarity != null) stats.push(["Similarity to original", analysis.semantic_similarity]);

  return (
    <section className="card analysis" aria-label="Text analysis">
      <h3>Output analysis</h3>
      <dl className="metric-grid">
        {stats.map(([k, v]) => (
          <div className="metric" key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {analysis.repeated_phrases.length > 0 && (
        <p className="small">
          <strong>Repeated phrases:</strong>{" "}
          {analysis.repeated_phrases.map((p) => `“${p.text}” ×${p.count}`).join(", ")}
        </p>
      )}
      {analysis.repeated_words.length > 0 && (
        <p className="small">
          <strong>Frequent words:</strong> {analysis.repeated_words.map((w) => `${w.text} ×${w.count}`).join(", ")}
        </p>
      )}
    </section>
  );
}
