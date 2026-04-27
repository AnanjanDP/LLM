def build_prompt(query, context, history):
    history_text = ""

    for h in history[-3:]:
        history_text += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    return f"""
You are a helpful AI assistant. Answer clearly and accurately.

Context:
{context if context else "No relevant context available"}

{history_text}

User: {query}
Assistant:
"""