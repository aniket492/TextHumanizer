// Centralised API client. The base URL comes from VITE_API_BASE_URL (empty = same origin).
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");

const TOKEN_KEY = "humanizer_token";

export const tokenStore = {
  get() {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(token) {
    try {
      localStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* storage unavailable (private mode); token lives only in memory */
    }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* ignore */
    }
  },
};

export class ApiError extends Error {
  constructor(message, { status = 0, code = "network_error", details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

let unauthorizedHandler = null;
export function onUnauthorized(handler) {
  unauthorizedHandler = handler;
}

async function request(path, { method = "GET", body, auth = true, signal } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const token = auth ? tokenStore.get() : null;
  if (token) headers.Authorization = `Bearer ${token}`;

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError("Cannot reach the server. Check your connection and try again.");
  }

  if (response.status === 204) return null;
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const error = data?.error ?? {};
    let message = error.message || `Request failed (${response.status}).`;
    if (error.details?.length) {
      message = error.details.map((d) => (d.field ? `${d.field}: ${d.message}` : d.message)).join(" ");
    }
    if (response.status === 401 && token && unauthorizedHandler) unauthorizedHandler();
    throw new ApiError(message, { status: response.status, code: error.code, details: error.details });
  }
  return data;
}

export const api = {
  health: () => request("/health", { auth: false }),
  register: (name, email, password) =>
    request("/api/auth/register", { method: "POST", body: { name, email, password }, auth: false }),
  login: (email, password) => request("/api/auth/login", { method: "POST", body: { email, password }, auth: false }),
  me: () => request("/api/auth/me"),
  humanize: (text, tone, intensity, { signal } = {}) =>
    request("/api/humanize", { method: "POST", body: { text, tone, intensity }, signal }),
  analyze: (text, compareText) =>
    request("/api/analyze", { method: "POST", body: { text, compare_text: compareText || null } }),
  history: (page = 1, pageSize = 20) => request(`/api/history?page=${page}&page_size=${pageSize}`),
  historyItem: (id) => request(`/api/history/${encodeURIComponent(id)}`),
  deleteHistory: (id) => request(`/api/history/${encodeURIComponent(id)}`, { method: "DELETE" }),
  usage: () => request("/api/usage"),
};
