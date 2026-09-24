import React, { useState } from "react";
import { X, Lock, Mail, ArrowRight } from "lucide-react";
import { authAPI } from "../services/api";

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isLogin) {
        const res = await authAPI.login(email, password);
        localStorage.setItem("token", res.data.access_token);
        onAuthSuccess(res.data.user);
      } else {
        await authAPI.register(email, password);
        const loginRes = await authAPI.login(email, password);
        localStorage.setItem("token", loginRes.data.access_token);
        onAuthSuccess(loginRes.data.user);
      }
      onClose();
    } catch (err) {
      setError(err.response?.data?.error || "Authentication failed. Check credentials.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ fontSize: "1.2rem", color: "#ffffff" }}>{isLogin ? "Sign In" : "Create Account"}</h3>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer" }}>
            <X size={20} />
          </button>
        </div>

        {error && (
          <div style={{ background: "rgba(239, 68, 68, 0.15)", border: "1px solid #ef4444", color: "#fca5a5", padding: "8px 12px", borderRadius: "8px", fontSize: "0.85rem" }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>Email Address</label>
            <div style={{ position: "relative" }}>
              <Mail size={16} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-dim)" }} />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="developer@example.com"
                style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "1px solid var(--border-color)", padding: "10px 10px 10px 38px", borderRadius: "8px", color: "#fff", outline: "none" }}
              />
            </div>
          </div>

          <div>
            <label style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "block", marginBottom: "4px" }}>Password</label>
            <div style={{ position: "relative" }}>
              <Lock size={16} style={{ position: "absolute", left: "12px", top: "12px", color: "var(--text-dim)" }} />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{ width: "100%", background: "rgba(255,255,255,0.05)", border: "1px solid var(--border-color)", padding: "10px 10px 10px 38px", borderRadius: "8px", color: "#fff", outline: "none" }}
              />
            </div>
          </div>

          <button type="submit" disabled={loading} className="glass-button glass-button-primary" style={{ marginTop: "8px", justifyContent: "center" }}>
            {loading ? "Processing..." : isLogin ? "Sign In" : "Register Account"} <ArrowRight size={16} />
          </button>
        </form>

        <div style={{ textAlign: "center", fontSize: "0.8rem", color: "var(--text-dim)", marginTop: "8px" }}>
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button onClick={() => setIsLogin(!isLogin)} style={{ background: "none", border: "none", color: "#a5b4fc", cursor: "pointer", textDecoration: "underline" }}>
            {isLogin ? "Register" : "Sign In"}
          </button>
        </div>
      </div>
    </div>
  );
}
