import { Link } from "react-router-dom";
import Humanizer from "../components/Humanizer";
import { useAuth } from "../hooks/useAuth";

export default function Home() {
  const { user } = useAuth();
  return (
    <>
      <section className="hero">
        <h1>Turn stiff, mechanical text into natural writing</h1>
        <p className="lead">
          Rewrite drafts in seven tones while keeping your meaning, names, numbers, dates, citations and links intact.
          Every result comes with transparent quality checks.
        </p>
        {!user && (
          <div className="actions hero-actions">
            <Link to="/register" className="btn btn-primary">
              Create a free account
            </Link>
            <Link to="/login" className="btn btn-ghost">
              Log in
            </Link>
          </div>
        )}
        <ul className="feature-list">
          <li>Meaning-preservation check</li>
          <li>Facts and entities protected</li>
          <li>Readability and repetition analysis</li>
        </ul>
      </section>
      <Humanizer />
    </>
  );
}
