import { useState, useEffect, useRef } from "react";
import axios from "axios";

const API = "http://127.0.0.1:8000";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!input) return;

    const userMsg = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);

    setLoading(true);

    try {
      const res = await axios.post(`${API}/chat/`, {
        query: input,
        session_id: "default"
      });

      const botMsg = { role: "assistant", content: res.data.answer };
      setMessages(prev => [...prev, botMsg]);

    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Error connecting to server"
      }]);
    }

    setInput("");
    setLoading(false);
  };

  // FILE UPLOAD
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);

    await axios.post(`${API}/upload-doc`, formData);

    alert("File uploaded!");
  };

  return (
    <div style={styles.container}>
      <h2> AI Chatbot</h2>

      <input type="file" onChange={handleUpload} />

      <div style={styles.chatBox}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            ...styles.message,
            alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
            background: msg.role === "user" ? "#007bff" : "#eee",
            color: msg.role === "user" ? "white" : "black"
          }}>
            {msg.content}
          </div>
        ))}

        {loading && <div>Typing...</div>}

        <div ref={chatEndRef} />
      </div>

      <div style={styles.inputArea}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          style={styles.input}
          placeholder="Ask something..."
        />
        <button onClick={sendMessage} style={styles.button}>
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "600px",
    margin: "auto",
    padding: "20px"
  },
  chatBox: {
    display: "flex",
    flexDirection: "column",
    height: "400px",
    overflowY: "auto",
    border: "1px solid #ccc",
    padding: "10px",
    marginBottom: "10px",
    background: "#f9f9f9"
  },
  message: {
    padding: "10px",
    borderRadius: "10px",
    margin: "5px",
    maxWidth: "70%"
  },
  inputArea: {
    display: "flex"
  },
  input: {
    flex: 1,
    padding: "10px"
  },
  button: {
    padding: "10px"
  }
};

export default App;