import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Database, Radio } from "lucide-react";

export default function ChatInput({
  onSendMessage,
  loading,
  useRag,
  setUseRag,
  stream,
  setStream,
  provider,
  setProvider,
}) {
  const [input, setInput] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input.trim());
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "48px";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="input-area">
      <form className="input-box" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="textarea-field"
          placeholder="Ask a technical question, retrieve context, or analyze docs..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />

        <div className="input-toolbar">
          <div className="tool-group">
            <label className="toggle-label" title="Enable FAISS Vector Context Retrieval">
              <input
                type="checkbox"
                checked={useRag}
                onChange={(e) => setUseRag(e.target.checked)}
                style={{ accentColor: "var(--primary)" }}
              />
              <Database size={14} color={useRag ? "#34d399" : "var(--text-dim)"} /> RAG Context
            </label>

            <label className="toggle-label" title="Stream response using Server-Sent Events (SSE)">
              <input
                type="checkbox"
                checked={stream}
                onChange={(e) => setStream(e.target.checked)}
                style={{ accentColor: "var(--primary)" }}
              />
              <Radio size={14} color={stream ? "#6366f1" : "var(--text-dim)"} /> Stream SSE
            </label>

            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              style={{
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid var(--border-color)",
                color: "var(--text-muted)",
                fontSize: "0.75rem",
                borderRadius: "6px",
                padding: "2px 6px",
                outline: "none"
              }}
            >
              <option value="groq">Groq (Llama-3.3)</option>
              <option value="mock">Local Mock Fallback</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="glass-button glass-button-primary"
            style={{ padding: "6px 14px", borderRadius: "10px", opacity: !input.trim() || loading ? 0.5 : 1 }}
          >
            {loading ? <Sparkles size={16} className="spin" /> : <Send size={16} />}
            <span>Send</span>
          </button>
        </div>
      </form>
    </div>
  );
}
