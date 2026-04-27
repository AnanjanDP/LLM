from fastapi import APIRouter
from app.schemas.chat import ChatRequest
from app.services.rag import get_context
from groq import Groq

from dotenv import load_dotenv
import os

router = APIRouter()

# ✅ Load env (supports secrets.env)
ENV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "secrets.env")
)

load_dotenv(dotenv_path=ENV_PATH, override=True)

# ✅ Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# ❗ Safety check
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Check your secrets.env file.")

# ✅ Init Groq client
client = Groq(api_key=GROQ_API_KEY)

# ✅ In-memory chat memory
chat_sessions = {}


@router.post("/")
async def chat(request: ChatRequest):
    try:
        query = request.query
        session_id = request.session_id

        # ✅ Init session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        history = chat_sessions[session_id]

        # ✅ Get RAG context (IMPORTANT: expects tuple)
        chunks, context = get_context(query)

        # ✅ Build messages
        messages = [
            {
                "role": "system",
                "content": (
    "Answer using the context but rephrase it clearly. Avoid repetition. "
    "Add examples when helpful, even if not explicitly in the context. "
    "Keep answers structured and easy to understand. "
    "Only say 'I don't know' if the topic is completely unrelated to the context."
)
            }
        ]

        messages.extend(history)

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        })

        # ✅ LLM call
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        # ✅ Store chat memory
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": chunks
        }

    except Exception as e:
        return {"error": str(e)}