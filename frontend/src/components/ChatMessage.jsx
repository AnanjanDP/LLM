import React, { useState } from "react";
import { User, Bot, Copy, Check, BookOpen, Layers } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import rehypeRaw from "rehype-raw";
import rehypeSanitize from "rehype-sanitize";

/**
 * Helper to safely extract renderable text from string or object content
 * and sanitize raw HTML tags (<br>, &lt;br&gt;) into actual line breaks.
 */
function getRenderableText(rawContent) {
  if (rawContent === null || rawContent === undefined) {
    return "";
  }
  let str = "";
  if (typeof rawContent === "string") {
    str = rawContent;
  } else if (typeof rawContent === "object") {
    if (typeof rawContent.content === "string") str = rawContent.content;
    else if (typeof rawContent.text === "string") str = rawContent.text;
    else if (typeof rawContent.answer === "string") str = rawContent.answer;
    else if (typeof rawContent.message === "string") str = rawContent.message;
    else {
      try {
        str = JSON.stringify(rawContent, null, 2);
      } catch (e) {
        str = String(rawContent);
      }
    }
  } else {
    str = String(rawContent);
  }

  // Pre-process HTML entities & line breaks to ensure clean markdown conversion
  str = str
    .replace(/&lt;br\s*\/?&gt;/gi, "<br />")
    .replace(/&nbsp;/g, " ")
    .replace(/<br\s*\/?>/gi, "<br />");

  return str;
}

export default function ChatMessage({ message }) {
  const [copied, setCopied] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const isUser = message.role === "user";
  const renderableText = getRenderableText(message.content || message.text || message.answer);

  const handleCopy = () => {
    navigator.clipboard.writeText(renderableText.replace(/<br\s*\/?>/gi, "\n"));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`message-bubble ${isUser ? "user" : "assistant"}`}>
      <div className={`avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>

      <div className="message-content">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "8px" }}>
          <div style={{ flex: 1 }}>
            {isUser ? (
              <p style={{ whiteSpace: "pre-wrap" }}>{renderableText.replace(/<br\s*\/?>/gi, "\n")}</p>
            ) : (
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkBreaks]}
                rehypePlugins={[rehypeRaw, rehypeSanitize]}
                components={{
                  br: () => <br />,
                  p: ({ children }) => <p style={{ marginBottom: "0.6em", lineHeight: "1.6" }}>{children}</p>,
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: "#a5b4fc", textDecoration: "underline" }}>
                      {children}
                    </a>
                  ),
                  code: ({ inline, className, children, ...props }) => (
                    inline ? (
                      <code style={{ background: "rgba(255,255,255,0.1)", padding: "2px 5px", borderRadius: "4px", fontSize: "0.88em" }} {...props}>
                        {children}
                      </code>
                    ) : (
                      <pre style={{ background: "rgba(15, 23, 42, 0.6)", padding: "12px", borderRadius: "8px", overflowX: "auto", margin: "8px 0" }}>
                        <code className={className} {...props}>{children}</code>
                      </pre>
                    )
                  )
                }}
              >
                {renderableText}
              </ReactMarkdown>
            )}
          </div>

          {!isUser && renderableText && (
            <button
              onClick={handleCopy}
              style={{ background: "none", border: "none", color: "var(--text-dim)", cursor: "pointer", padding: "2px" }}
              title="Copy message"
            >
              {copied ? <Check size={14} color="#34d399" /> : <Copy size={14} />}
            </button>
          )}
        </div>

        {/* Citations / Sources Accordion */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="citation-container">
            <button
              onClick={() => setShowSources(!showSources)}
              style={{
                background: "none",
                border: "none",
                color: "#a5b4fc",
                fontSize: "0.75rem",
                display: "flex",
                alignItems: "center",
                gap: "4px",
                cursor: "pointer",
                padding: "2px 0"
              }}
            >
              <BookOpen size={12} /> {message.sources.length} RAG Source Citation{message.sources.length > 1 ? "s" : ""}
            </button>

            {showSources && (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginTop: "6px" }}>
                {message.sources.map((src, i) => (
                  <div key={i} className="citation-tag">
                    <Layers size={12} style={{ marginTop: "2px", flexShrink: 0 }} />
                    <div>
                      <strong>[{src.id || i + 1}] {src.source_doc || "Knowledge Document"}</strong>
                      {src.score && <span style={{ opacity: 0.7, marginLeft: "6px" }}>(score: {src.score})</span>}
                      <p style={{ marginTop: "2px", fontSize: "0.72rem", color: "var(--text-muted)" }}>
                        {getRenderableText(src.text)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
