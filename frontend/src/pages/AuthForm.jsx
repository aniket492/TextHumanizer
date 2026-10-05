import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import Alert from "../components/Alert";
import { useAuth } from "../hooks/useAuth";

export default function AuthForm({ mode }) {
  const isRegister = mode === "register";
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({ name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const update = (field) => (e) => setForm((f) => ({ ...f, [field]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (isRegister && form.password.length < 8) return setError("Password must be at least 8 characters.");
    setSubmitting(true);
    try {
      if (isRegister) await register(form.name.trim(), form.email.trim(), form.password);
      else await login(form.email.trim(), form.password);
      navigate(location.state?.from || "/dashboard", { replace: true });
    } catch (err) {
      setError(err.message);
      setSubmitting(false);
    }
  };

  return (
    <section className="auth card">
      <h1>{isRegister ? "Create your account" : "Welcome back"}</h1>
      {location.state?.reason && <Alert kind="info">{location.state.reason}</Alert>}
      <form onSubmit={handleSubmit} noValidate={false}>
        {isRegister && (
          <div className="field">
            <label htmlFor="name">Name</label>
            <input id="name" autoComplete="name" required maxLength={120} value={form.name} onChange={update("name")} />
          </div>
        )}
        <div className="field">
          <label htmlFor="email">E-mail</label>
          <input id="email" type="email" autoComplete="email" required value={form.email} onChange={update("email")} />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoComplete={isRegister ? "new-password" : "current-password"}
            required
            minLength={isRegister ? 8 : 1}
            maxLength={72}
            value={form.password}
            onChange={update("password")}
            aria-describedby={isRegister ? "password-hint" : undefined}
          />
          {isRegister && (
            <small id="password-hint" className="muted">
              At least 8 characters, including a letter and a digit.
            </small>
          )}
        </div>
        <Alert onClose={() => setError("")}>{error}</Alert>
        <button className="btn btn-primary btn-block" type="submit" disabled={submitting}>
          {submitting ? "Please wait..." : isRegister ? "Sign up" : "Log in"}
        </button>
      </form>
      <p className="muted small center">
        {isRegister ? (
          <>Already have an account? <Link to="/login">Log in</Link></>
        ) : (
          <>New here? <Link to="/register" state={location.state}>Create an account</Link></>
        )}
      </p>
    </section>
  );
}
