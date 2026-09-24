import React from "react";
import { Activity, User, LogOut, ShieldCheck, Zap } from "lucide-react";

export default function Navbar({
  user,
  onOpenAuthModal,
  onLogout,
  onRunEval,
  currentProvider,
}) {
  return (
    <header className="navbar">
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div className="badge">
          <span className="status-dot"></span>
          <span>Groq Llama-3.3-70B Active</span>
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>
          Provider: <strong style={{ color: "#a5b4fc" }}>{currentProvider || "groq"}</strong>
        </div>
      </div>

      <div className="nav-actions">
        <button
          className="glass-button"
          onClick={onRunEval}
          title="Run AI Quality & Groundedness Evaluation Benchmark"
        >
          <Zap size={15} color="#f59e0b" /> AI Benchmark Eval
        </button>

        {user ? (
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>{user.email}</span>
            <button className="glass-button" onClick={onLogout} title="Log out">
              <LogOut size={15} />
            </button>
          </div>
        ) : (
          <button className="glass-button glass-button-primary" onClick={onOpenAuthModal}>
            <User size={15} /> Sign In
          </button>
        )}
      </div>
    </header>
  );
}
