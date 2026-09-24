import React, { useState, useEffect, useRef } from "react";
import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import AuthModal from "./components/AuthModal";
import EvalModal from "./components/EvalModal";
import { chatAPI, streamChatMessage, authAPI } from "./services/api";
import { Sparkles, MessageSquare, Zap, Cpu, Terminal, Layers } from "lucide-react";

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState("default");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [useRag, setUseRag] = useState(true);
  const [stream, setStream] = useState(true);
  const [provider, setProvider] = useState("groq");
  
  const [user, setUser] = useState(null);
  const [authModalOpen, setAuthModalOpen] = useState(false);
  const [evalModalOpen, setEvalModalOpen] = useState(false);
  const [errorToast, setErrorToast] = useState("");

  const chatEndRef = useRef(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
    checkCurrentUser();
  }, []);

  // Load specific conversation messages when session changes manually
  const loadConversationMessages = async (id) => {
    if (!id || id === "default") {
      setMessages([]);
      return;
    }
    try {
      const res = await chatAPI.getConversation(id);
      if (res.data && res.data.messages) {
        setMessages(
          res.data.messages.map((m) => ({
            role: m.role,
            content: typeof m.content === "string" ? m.content : (m.content?.text || m.content?.content || JSON.stringify(m.content, null, 2)),
            sources: m.sources,
          }))
        );
      }
    } catch (err) {
      console.error("Failed to load conversation details", err);
    }
  };

  const handleSelectSession = (id) => {
    if (loading) return;
    setActiveSessionId(id);
    loadConversationMessages(id);
  };

  const checkCurrentUser = async () => {
    if (localStorage.getItem("token")) {
      try {
        const res = await authAPI.getMe();
        setUser(res.data);
      } catch (e) {
        localStorage.removeItem("token");
      }
    }
  };

  const loadConversations = async () => {
    try {
      const res = await chatAPI.listConversations();
      setConversations(res.data || []);
    } catch (err) {
      console.error("Failed to load conversations", err);
    }
  };

  const handleNewChat = () => {
    if (loading) return;
    const newId = `session-${Date.now()}`;
    setActiveSessionId(newId);
    setMessages([]);
  };

  const handleDeleteSession = async (id) => {
    try {
      await chatAPI.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeSessionId === id) {
        handleNewChat();
      }
    } catch (err) {
      console.error("Failed to delete session", err);
    }
  };

  const handleSendMessage = async (queryText) => {
    if (!queryText || loading) return;
    
    setErrorToast("");
    const userMsg = { role: "user", content: queryText };
    
    // Append user message immediately to state
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    const currentSession = activeSessionId;

    if (stream) {
      // SSE Streaming Mode
      let assistantMsg = { role: "assistant", content: "", sources: [] };
      setMessages((prev) => [...prev, assistantMsg]);

      await streamChatMessage(
        queryText,
        currentSession,
        { provider, useRag },
        (meta) => {
          if (meta.conversation_id && meta.conversation_id !== activeSessionId) {
            setActiveSessionId(meta.conversation_id);
          }
          if (meta.sources) {
            assistantMsg.sources = meta.sources;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { ...assistantMsg };
              return updated;
            });
          }
        },
        (token) => {
          const tokenStr = typeof token === "string" ? token : (token?.token || token?.content || JSON.stringify(token));
          assistantMsg.content += tokenStr;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { ...assistantMsg };
            return updated;
          });
        },
        () => {
          setLoading(false);
          loadConversations();
        },
        (err) => {
          setLoading(false);
          setErrorToast(`Connection Error: ${err}`);
          setMessages((prev) => {
            const updated = [...prev];
            const lastIdx = updated.length - 1;
            if (lastIdx >= 0 && updated[lastIdx].role === "assistant") {
              const prevContent = updated[lastIdx].content || "";
              updated[lastIdx] = {
                ...updated[lastIdx],
                content: prevContent
                  ? `${prevContent}\n\n⚠️ [Stream interrupted: ${err}]`
                  : `⚠️ Failed to receive stream response: ${err}`,
              };
            } else {
              updated.push({ role: "assistant", content: `⚠️ Failed to receive stream response: ${err}` });
            }
            return updated;
          });
        }
      );
    } else {
      // Standard REST Mode
      try {
        const res = await chatAPI.sendMessage(queryText, currentSession, { provider, useRag });
        const answerRaw = res.data.answer;
        const answerStr = typeof answerRaw === "string" ? answerRaw : (answerRaw?.text || answerRaw?.content || JSON.stringify(answerRaw, null, 2));
        
        const botMsg = {
          role: "assistant",
          content: answerStr,
          sources: res.data.sources || [],
        };
        setMessages((prev) => [...prev, botMsg]);

        if (res.data.conversation_id) {
          setActiveSessionId(res.data.conversation_id);
        }
        loadConversations();
      } catch (err) {
        setErrorToast("Error connecting to backend server.");
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "⚠️ Could not connect to backend API server. Make sure FastAPI server is running on http://127.0.0.1:8000." },
        ]);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        conversations={conversations}
        activeSessionId={activeSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        onDocUploaded={loadConversations}
      />

      <main className="main-content">
        <Navbar
          user={user}
          onOpenAuthModal={() => setAuthModalOpen(true)}
          onLogout={() => {
            localStorage.removeItem("token");
            setUser(null);
          }}
          onRunEval={() => setEvalModalOpen(true)}
          currentProvider={provider}
        />

        {errorToast && (
          <div style={{ background: "rgba(239,68,68,0.2)", borderBottom: "1px solid #ef4444", color: "#fca5a5", padding: "8px 24px", fontSize: "0.85rem", textAlign: "center" }}>
            {errorToast}
          </div>
        )}

        <div className="chat-window">
          {messages.length === 0 ? (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", textAlign: "center", gap: "16px", padding: "40px 20px" }}>
              <div style={{ width: "64px", height: "64px", borderRadius: "20px", background: "rgba(99,102,241,0.15)", border: "1px solid var(--border-glow)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 30px var(--primary-glow)" }}>
                <Sparkles size={32} color="#a5b4fc" />
              </div>

              <h2 style={{ color: "#ffffff", fontSize: "1.6rem", fontWeight: "700", letterSpacing: "-0.5px" }}>
                Enterprise RAG & AI Intelligence Engine
              </h2>
              <p style={{ maxWidth: "520px", color: "var(--text-muted)", fontSize: "0.92rem", lineHeight: "1.6" }}>
                Powered by Groq Llama-3.3, FAISS Vector Similarity Search, Pydantic validation, and SSE streaming. Ask a query below or select a sample prompt to start.
              </p>

              {/* Sample Prompt Suggestion Cards */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px", width: "100%", maxWidth: "600px", marginTop: "12px" }}>
                {[
                  { title: "Platform Architecture", text: "What architecture does this RAG platform use?" },
                  { title: "Vector Search Engine", text: "Explain how FAISS vector search operates in this app." },
                  { title: "LLM Retry Strategy", text: "How does LLM provider retry & fallback work?" },
                  { title: "Security & Auth", text: "What security and JWT authentication features are active?" },
                ].map((preset, i) => (
                  <div
                    key={i}
                    className="glass-panel glass-panel-interactive"
                    onClick={() => handleSendMessage(preset.text)}
                    style={{ padding: "14px", textAlign: "left" }}
                  >
                    <div style={{ fontSize: "0.85rem", fontWeight: "600", color: "#ffffff", marginBottom: "4px" }}>
                      💡 {preset.title}
                    </div>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted)", lineHeight: "1.4" }}>
                      "{preset.text}"
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => <ChatMessage key={idx} message={msg} />)
          )}

          {loading && !stream && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--text-muted)", fontSize: "0.85rem", padding: "8px 0" }}>
              <Sparkles size={16} className="spin" color="var(--primary)" />
              <span>Generating response with Groq Llama-3.3...</span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <ChatInput
          onSendMessage={handleSendMessage}
          loading={loading}
          useRag={useRag}
          setUseRag={setUseRag}
          stream={stream}
          setStream={setStream}
          provider={provider}
          setProvider={setProvider}
        />
      </main>

      <AuthModal
        isOpen={authModalOpen}
        onClose={() => setAuthModalOpen(false)}
        onAuthSuccess={(u) => setUser(u)}
      />

      <EvalModal
        isOpen={evalModalOpen}
        onClose={() => setEvalModalOpen(false)}
      />
    </div>
  );
}