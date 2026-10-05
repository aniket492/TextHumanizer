import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export default function Header() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  const handleLogout = () => {
    logout();
    setOpen(false);
    navigate("/");
  };
  const close = () => setOpen(false);

  return (
    <header className="site-header">
      <div className="container header-inner">
        <Link to="/" className="brand" onClick={close}>
          <span className="brand-mark" aria-hidden="true">H</span>
          <span>AI Humanizer</span>
        </Link>
        <button
          className="menu-toggle"
          aria-expanded={open}
          aria-controls="primary-nav"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="sr-only">Toggle navigation</span>
          <span aria-hidden="true">{open ? "✕" : "☰"}</span>
        </button>
        <nav id="primary-nav" className={`nav ${open ? "is-open" : ""}`} aria-label="Primary">
          {user ? (
            <>
              <NavLink to="/dashboard" onClick={close}>Dashboard</NavLink>
              <NavLink to="/history" onClick={close}>History</NavLink>
              <span className="nav-user" title={user.email}>{user.name}</span>
              <button className="btn btn-ghost btn-sm" onClick={handleLogout}>Log out</button>
            </>
          ) : (
            <>
              <NavLink to="/login" onClick={close}>Log in</NavLink>
              <Link to="/register" className="btn btn-primary btn-sm" onClick={close}>Sign up</Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}
