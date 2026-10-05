import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Header from "./components/Header";
import Spinner from "./components/Spinner";
import { useAuth } from "./hooks/useAuth";
import Dashboard from "./pages/Dashboard";
import History from "./pages/History";
import HistoryDetail from "./pages/HistoryDetail";
import Home from "./pages/Home";
import Login from "./pages/Login";
import NotFound from "./pages/NotFound";
import Register from "./pages/Register";

function RequireAuth({ children }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <Spinner label="Checking your session" block />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return children;
}

function GuestOnly({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <Spinner label="Checking your session" block />;
  return user ? <Navigate to="/dashboard" replace /> : children;
}

export default function App() {
  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Header />
      <main id="main" className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<GuestOnly><Login /></GuestOnly>} />
          <Route path="/register" element={<GuestOnly><Register /></GuestOnly>} />
          <Route path="/dashboard" element={<RequireAuth><Dashboard /></RequireAuth>} />
          <Route path="/history" element={<RequireAuth><History /></RequireAuth>} />
          <Route path="/history/:id" element={<RequireAuth><HistoryDetail /></RequireAuth>} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <footer className="site-footer container">
        <p>
          Rewrites aim for clear, natural writing while preserving meaning and facts. Always review the output
          before using it. No tool can guarantee any particular AI-detector result.
        </p>
      </footer>
    </>
  );
}
