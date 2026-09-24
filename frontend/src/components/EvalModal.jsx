import React, { useState, useEffect } from "react";
import { X, Zap, CheckCircle2, XCircle, Clock, ShieldCheck, Play, Layers, BarChart2, Server } from "lucide-react";
import { evalAPI } from "../services/api";

export default function EvalModal({ isOpen, onClose }) {
  const [loading, setLoading] = useState(false);
  const [evalData, setEvalData] = useState(null);
  const [systemMetrics, setSystemMetrics] = useState(null);
  const [pastExperiments, setPastExperiments] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");

  const [retrievalMode, setRetrievalMode] = useState("hybrid");
  const [useReranker, setUseReranker] = useState(false);
  const [promptVersion, setPromptVersion] = useState("v1.0");

  useEffect(() => {
    if (isOpen) {
      setErrorMsg("");
      fetchSystemMetrics();
      fetchPastExperiments();
    }
  }, [isOpen]);

  const fetchSystemMetrics = async () => {
    try {
      const res = await evalAPI.getSystemMetrics();
      setSystemMetrics(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const fetchPastExperiments = async () => {
    try {
      const res = await evalAPI.listExperiments();
      setPastExperiments(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const runExperiment = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const res = await evalAPI.runExperiment({
        retrievalMode,
        useReranker,
        promptVersion,
        topK: 4,
      });
      setEvalData(res.data);
      fetchSystemMetrics();
      fetchPastExperiments();
    } catch (err) {
      console.error(err);
      setErrorMsg(err.response?.data?.detail || err.message || "Failed to run evaluation benchmark.");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;


  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: "850px", width: "90%" }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Zap size={24} color="#f59e0b" />
            <h3 style={{ fontSize: "1.25rem", color: "#ffffff", margin: 0 }}>
              RAG Intelligence & Evaluation Dashboard
            </h3>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer" }}>
            <X size={20} />
          </button>
        </div>

        {/* Configuration Controls */}
        <div className="glass-panel" style={{ padding: "12px 16px", marginTop: "12px", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "4px" }}>Retrieval Strategy</label>
            <select
              value={retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value)}
              style={{ width: "100%", background: "#1e293b", color: "#fff", border: "1px solid #334155", borderRadius: "6px", padding: "6px 10px", fontSize: "0.85rem" }}
            >
              <option value="hybrid">Hybrid (Vector + BM25 + RRF)</option>
              <option value="vector">Dense Vector Search (FAISS)</option>
              <option value="bm25">Sparse Keyword Search (BM25)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "4px" }}>Cross-Encoder Reranker</label>
            <select
              value={useReranker ? "true" : "false"}
              onChange={(e) => setUseReranker(e.target.value === "true")}
              style={{ width: "100%", background: "#1e293b", color: "#fff", border: "1px solid #334155", borderRadius: "6px", padding: "6px 10px", fontSize: "0.85rem" }}
            >
              <option value="false">Disabled (Fast Rank)</option>
              <option value="true">Enabled (ms-marco CrossEncoder)</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "4px" }}>Prompt Version</label>
            <select
              value={promptVersion}
              onChange={(e) => setPromptVersion(e.target.value)}
              style={{ width: "100%", background: "#1e293b", color: "#fff", border: "1px solid #334155", borderRadius: "6px", padding: "6px 10px", fontSize: "0.85rem" }}
            >
              <option value="v1.0">v1.0 Standard Production</option>
              <option value="v1.1_strict">v1.1 Strict Groundedness</option>
              <option value="v2.0_cot">v2.0 Chain-of-Thought (CoT)</option>
            </select>
          </div>
        </div>

        {/* Trigger Button */}
        <div style={{ textAlign: "center", margin: "14px 0" }}>
          <button
            className="glass-button glass-button-primary"
            onClick={runExperiment}
            disabled={loading}
            style={{ fontSize: "0.95rem", padding: "10px 28px", opacity: loading ? 0.7 : 1 }}
          >
            {loading ? <Zap className="spin" size={18} /> : <Play size={18} />}
            {loading ? "Running Benchmark Experiment..." : "Run Evaluation Benchmark"}
          </button>
        </div>

        {errorMsg && (
          <div style={{ background: "rgba(239,68,68,0.2)", border: "1px solid #ef4444", color: "#fca5a5", borderRadius: "8px", padding: "10px 14px", marginBottom: "14px", fontSize: "0.85rem" }}>
            ⚠️ {errorMsg}
          </div>
        )}

        {/* System Latency Percentiles & Redis Stats */}
        {systemMetrics && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px", marginBottom: "16px" }}>
            <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>P50 Latency</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#60a5fa" }}>
                {systemMetrics.latency_percentiles_sec?.p50 ?? 0}s
              </div>
            </div>
            <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>P95 Latency</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#f59e0b" }}>
                {systemMetrics.latency_percentiles_sec?.p95 ?? 0}s
              </div>
            </div>
            <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>P99 Latency</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#ef4444" }}>
                {systemMetrics.latency_percentiles_sec?.p99 ?? 0}s
              </div>
            </div>
            <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
              <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Redis Cache Hit Rate</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "700", color: "#34d399" }}>
                {systemMetrics.cache_performance?.hit_rate_pct ?? 0}%
              </div>
            </div>
          </div>
        )}

        {/* Evaluation Results Section */}
        {evalData && (
          <div style={{ display: "flex", flexDirection: "column", gap: "14px", maxHeight: "320px", overflowY: "auto" }}>
            {/* Metric Cards Grid */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
              <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Recall@5</div>
                <div style={{ fontSize: "1.3rem", fontWeight: "700", color: "#34d399" }}>
                  {(((evalData.retrieval_metrics?.["recall@5"] ?? evalData.retrieval_metrics?.recall_at_5 ?? 0)) * 100).toFixed(0)}%
                </div>
              </div>

              <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>MRR</div>
                <div style={{ fontSize: "1.3rem", fontWeight: "700", color: "#818cf8" }}>
                  {evalData.retrieval_metrics.mrr.toFixed(2)}
                </div>
              </div>
              <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Faithfulness</div>
                <div style={{ fontSize: "1.3rem", fontWeight: "700", color: "#38bdf8" }}>
                  {(evalData.generation_metrics.faithfulness * 100).toFixed(0)}%
                </div>
              </div>
              <div className="glass-panel" style={{ padding: "10px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Answer Relevance</div>
                <div style={{ fontSize: "1.3rem", fontWeight: "700", color: "#a78bfa" }}>
                  {(evalData.generation_metrics.answer_relevance * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {/* Test Cases List */}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {evalData.test_cases.map((tc, idx) => (
                <div key={idx} className="glass-panel" style={{ padding: "10px 12px", fontSize: "0.82rem" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      {tc.passed ? <CheckCircle2 size={16} color="#34d399" /> : <XCircle size={16} color="#ef4444" />}
                      <strong style={{ color: "#fff" }}>{tc.question}</strong>
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>{tc.latency_sec}s</span>
                  </div>
                  <div style={{ marginTop: "4px", fontSize: "0.78rem", color: "var(--text-muted)" }}>
                    Answer: {tc.answer_snippet}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
