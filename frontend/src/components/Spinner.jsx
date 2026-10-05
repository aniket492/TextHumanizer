export default function Spinner({ label = "Loading", block = false }) {
  return (
    <div className={block ? "spinner-block" : "spinner-inline"} role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className={block ? "" : "sr-only"}>{label}</span>
    </div>
  );
}
