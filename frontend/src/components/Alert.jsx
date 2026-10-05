export default function Alert({ kind = "error", children, onClose }) {
  if (!children) return null;
  return (
    <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <span>{children}</span>
      {onClose && (
        <button className="alert-close" onClick={onClose} aria-label="Dismiss">
          {"✕"}
        </button>
      )}
    </div>
  );
}
