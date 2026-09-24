import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const authAPI = {
  register: (email, password) => apiClient.post("/auth/register", { email, password }),
  login: (email, password) => apiClient.post("/auth/token", { email, password }),
  getMe: () => apiClient.get("/auth/me"),
};

export const chatAPI = {
  sendMessage: (query, sessionId, options = {}) =>
    apiClient.post("/chat/", {
      query,
      session_id: sessionId,
      provider: options.provider || "groq",
      model: options.model,
      use_rag: options.useRag !== false,
      retrieval_mode: options.retrievalMode || "hybrid",
      use_reranker: !!options.useReranker,
      top_k: options.topK || 4,
      prompt_version: options.promptVersion || "v1.0",
    }),

  listConversations: () => apiClient.get("/chat/conversations"),
  getConversation: (conversationId) => apiClient.get(`/chat/conversations/${conversationId}`),
  deleteConversation: (conversationId) => apiClient.delete(`/chat/conversations/${conversationId}`),
};

export const documentAPI = {
  uploadDocument: (file, chunkStrategy = "word", chunkSize = 200, chunkOverlap = 40, sessionId = null) => {
    const formData = new FormData();
    formData.append("file", file);
    const sessionParam = sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : "";
    return apiClient.post(
      `/upload-doc?chunk_strategy=${chunkStrategy}&chunk_size=${chunkSize}&chunk_overlap=${chunkOverlap}${sessionParam}`,
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
  },
  clearDocuments: (sessionId = null) => {
    const param = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return apiClient.delete(`/clear-documents${param}`);
  },
};


export const evalAPI = {
  runExperiment: (config = {}) =>
    apiClient.post("/eval/experiment", {
      retrieval_mode: config.retrievalMode || "hybrid",
      use_reranker: !!config.useReranker,
      top_k: config.topK || 4,
      prompt_version: config.promptVersion || "v1.0",
      model: config.model || null,
    }),

  listExperiments: () => apiClient.get("/eval/experiments"),
  getSystemMetrics: () => apiClient.get("/eval/metrics"),
};

export const streamChatMessage = async (
  query,
  sessionId,
  options = {},
  onMetadata,
  onToken,
  onDone,
  onError
) => {
  let isDoneCalled = false;
  const triggerDone = (data = { done: true }) => {
    if (!isDoneCalled) {
      isDoneCalled = true;
      if (onDone) onDone(data);
    }
  };

  try {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(localStorage.getItem("token")
          ? { Authorization: `Bearer ${localStorage.getItem("token")}` }
          : {}),
      },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        provider: options.provider || "groq",
        model: options.model,
        use_rag: options.useRag !== false,
        retrieval_mode: options.retrievalMode || "hybrid",
        use_reranker: !!options.useReranker,
        top_k: options.topK || 4,
        prompt_version: options.promptVersion || "v1.0",
      }),
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    if (!response.body) {
      throw new Error("ReadableStream not supported by browser or empty response body.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    const processLine = (rawLine) => {
      const line = rawLine.trim();
      if (!line || !line.startsWith("data:")) return;

      const jsonStr = line.replace(/^data:\s*/, "").trim();
      if (!jsonStr) return;

      if (jsonStr === "[DONE]" || jsonStr === "done") {
        triggerDone();
        return;
      }

      try {
        const data = JSON.parse(jsonStr);
        if (data.type === "metadata" && onMetadata) {
          onMetadata(data.data || data);
        } else if (data.type === "token" && onToken) {
          const textToken = data.token !== undefined ? data.token : (data.content || data.text || "");
          if (textToken !== "") onToken(textToken);
        } else if (data.type === "done") {
          triggerDone(data);
        } else if (data.type === "error" && onError) {
          onError(data.error);
        } else if (data.token !== undefined && onToken) {
          onToken(data.token);
        } else if (data.content !== undefined && onToken) {
          onToken(data.content);
        }
      } catch (e) {
        // Fallback for non-JSON text stream tokens (ignoring raw SSE framing errors)
        if (onToken && !jsonStr.startsWith("{") && !jsonStr.startsWith("[")) {
          onToken(jsonStr);
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";

      for (const line of lines) {
        processLine(line);
      }
    }

    // Flush any remaining decoder bytes and process leftover buffer line
    buffer += decoder.decode();
    if (buffer.trim()) {
      const remainingLines = buffer.split(/\r?\n/);
      for (const line of remainingLines) {
        processLine(line);
      }
    }

    triggerDone();
  } catch (err) {
    if (onError) onError(err.message || String(err));
  }
};
