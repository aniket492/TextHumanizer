import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { countWords } from "./components/options";
import { AuthProvider } from "./hooks/useAuth";
import { ApiError, api } from "./services/api";

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
  sessionStorage.clear();
});

describe("countWords", () => {
  it("counts words, contractions and hyphenated words", () => {
    expect(countWords("")).toBe(0);
    expect(countWords("It's a well-known fact, 2024.")).toBe(5);
  });
});

describe("App", () => {
  it("renders the landing page with editor controls", () => {
    renderAt("/");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/natural writing/i);
    expect(screen.getByLabelText(/your text/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/tone/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /humanize/i })).toBeDisabled();
  });

  it("updates the live word count", () => {
    renderAt("/");
    fireEvent.change(screen.getByLabelText(/your text/i), { target: { value: "one two three" } });
    expect(screen.getByText(/3 \/ 1,500 words/)).toBeInTheDocument();
  });

  it("redirects protected pages to login", () => {
    renderAt("/history");
    expect(screen.getByRole("heading", { name: /welcome back/i })).toBeInTheDocument();
  });

  it("shows API errors on failed login", async () => {
    vi.spyOn(api, "login").mockRejectedValue(new ApiError("Incorrect e-mail or password.", { status: 401 }));
    renderAt("/login");
    fireEvent.change(screen.getByLabelText(/e-mail/i), { target: { value: "a@example.com" } });
    fireEvent.change(screen.getByLabelText(/password/i), { target: { value: "wrongpass1" } });
    fireEvent.click(screen.getByRole("button", { name: /log in/i }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/incorrect/i));
  });
});
