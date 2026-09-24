import React, { useState } from "react";
import { Plus, MessageSquare, Trash2, Upload, FileText, Cpu, CheckCircle } from "lucide-react";
import { documentAPI } from "../services/api";

export default function Sidebar({
  conversations,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  onDocUploaded,
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadMsg("Indexing...");

    try {
      const res = await documentAPI.uploadDocument(file, "word", 200, 40, activeSessionId);
      setUploadMsg("Indexed!");

      if (onDocUploaded) onDocUploaded(res.data.message);
      setTimeout(() => setUploadMsg(""), 3000);
    } catch (err) {
      setUploadMsg("Upload failed");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <Cpu className="brand-icon" size={24} />
          <span>RAG Engine v1</span>
        </div>
      </div>

      <button className="glass-button glass-button-primary" onClick={onNewChat} style={{ width: "100%", justifyContent: "center" }}>
        <Plus size={18} /> New Conversation
      </button>

      {/* Document Upload Widget */}
      <div className="glass-panel" style={{ padding: "12px", display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", display: "flex", alignItems: "center", gap: "6px" }}>
          <FileText size={14} /> Knowledge Base Ingestion
        </div>
        <label className="glass-button" style={{ fontSize: "0.8rem", width: "100%", justifyContent: "center", cursor: "pointer" }}>
          <Upload size={14} /> {uploading ? "Ingesting..." : "Upload PDF / TXT / MD"}
          <input type="file" accept=".pdf,.txt,.md" onChange={handleFileUpload} style={{ display: "none" }} disabled={uploading} />
        </label>
        {uploadMsg && (
          <div style={{ fontSize: "0.75rem", color: uploadMsg.includes("failed") ? "#ef4444" : "#34d399", display: "flex", alignItems: "center", gap: "4px" }}>
            <CheckCircle size={12} /> {uploadMsg}
          </div>
        )}
      </div>

      {/* Conversations List */}
      <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: "1px", marginTop: "8px" }}>
        Recent Chats
      </div>
      <div className="conversation-list">
        {conversations.length === 0 ? (
          <div style={{ fontSize: "0.85rem", color: "var(--text-dim)", padding: "12px 0" }}>No saved sessions yet.</div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              className={`conversation-item ${conv.id === activeSessionId ? "active" : ""}`}
              onClick={() => onSelectSession(conv.id)}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px", overflow: "hidden" }}>
                <MessageSquare size={16} />
                <span className="conversation-title">{conv.title}</span>
              </div>
              <button
                className="delete-btn"
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(conv.id);
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
