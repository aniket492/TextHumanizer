import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { api, onUnauthorized, tokenStore } from "../services/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(() => Boolean(tokenStore.get()));

  const logout = useCallback(() => {
    tokenStore.clear();
    setUser(null);
  }, []);

  useEffect(() => {
    onUnauthorized(logout);
    if (!tokenStore.get()) return;
    let active = true;
    api
      .me()
      .then((me) => active && setUser(me))
      .catch(() => active && logout())
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [logout]);

  const handleToken = useCallback((data) => {
    tokenStore.set(data.access_token);
    setUser(data.user);
    return data.user;
  }, []);

  const login = useCallback((email, password) => api.login(email, password).then(handleToken), [handleToken]);
  const register = useCallback(
    (name, email, password) => api.register(name, email, password).then(handleToken),
    [handleToken],
  );

  const value = useMemo(() => ({ user, loading, login, register, logout }), [user, loading, login, register, logout]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
