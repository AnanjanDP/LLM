from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings


class PromptVersion:
    def __init__(self, version: str, system_prompt: str, context_format: str, description: str):
        self.version = version
        self.system_prompt = system_prompt
        self.context_format = context_format
        self.description = description


class PromptService:
    """Centralized, versioned prompt template and token cost management service."""

    PROMPT_VERSIONS: Dict[str, PromptVersion] = {
        "v1.0": PromptVersion(
            version="v1.0",
            system_prompt=(
                "You are an expert, reliable AI assistant specializing in technical knowledge retrieval and analysis. "
                "Answer questions clearly, accurately, and concisely based on the provided context.\n"
                "1. Prefer information directly supported by the provided context.\n"
                "2. If context is provided, cite key facts clearly.\n"
                "3. Maintain a professional tone with code snippets or bullet points when appropriate."
            ),
            context_format="### Context Information:\n{context}\n\n### User Question:\n{query}",
            description="Standard production RAG prompt focusing on accurate context citation."
        ),
        "v1.1_strict": PromptVersion(
            version="v1.1_strict",
            system_prompt=(
                "You are a strict, grounded AI assistant. Answer the user question ONLY using the provided context. "
                "If the information is not contained in the context, state: 'The provided document context does not contain sufficient information to answer this question.' "
                "Do NOT make assumptions or hallucinate."
            ),
            context_format="[DOCUMENT CONTEXT]\n{context}\n\n[USER QUERY]\n{query}",
            description="Strict groundedness prompt designed to eliminate hallucinations."
        ),
        "v2.0_cot": PromptVersion(
            version="v2.0_cot",
            system_prompt=(
                "You are an analytical AI reasoning engine. Solve the user query by breaking down your reasoning step-by-step "
                "based on the provided context before outputting your final summary answer."
            ),
            context_format="### Knowledge Context:\n{context}\n\n### Question:\n{query}\n\nThink step-by-step:",
            description="Chain-of-thought (CoT) prompt template for complex multi-step reasoning."
        )
    }

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "llama-3.3-70b-versatile": {"input_per_1m": 0.59, "output_per_1m": 0.79},
        "llama-3.1-8b-instant": {"input_per_1m": 0.05, "output_per_1m": 0.08},
        "mixtral-8x7b-32768": {"input_per_1m": 0.24, "output_per_1m": 0.24},
        "gemma2-9b-it": {"input_per_1m": 0.20, "output_per_1m": 0.20},
        "default": {"input_per_1m": 0.30, "output_per_1m": 0.50},
    }

    def get_prompt_version(self, version_name: str = None) -> PromptVersion:
        v = version_name or settings.DEFAULT_PROMPT_VERSION
        return self.PROMPT_VERSIONS.get(v, self.PROMPT_VERSIONS["v1.0"])

    def build_chat_messages(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        prompt_version_name: str = None
    ) -> Tuple[List[Dict[str, str]], str]:
        """Construct structured message payload and return (messages, version_used)."""
        pv = self.get_prompt_version(prompt_version_name)
        messages = [{"role": "system", "content": pv.system_prompt}]

        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant"):
                    messages.append({"role": role, "content": content})

        if context and context.strip():
            user_content = pv.context_format.format(context=context, query=query)
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})
        return messages, pv.version

    def estimate_tokens_and_cost(
        self, input_text: str, output_text: str, model_name: str = None
    ) -> Dict[str, Any]:
        """Estimate token usage and cost in USD."""
        model_key = (model_name or settings.DEFAULT_LLM_MODEL).lower()
        pricing = self.MODEL_PRICING.get(model_key, self.MODEL_PRICING["default"])

        # Word-based token heuristic (~1.33 tokens per word)
        input_tokens = int(len(input_text.split()) * 1.33) + 10
        output_tokens = int(len(output_text.split()) * 1.33) + 5
        total_tokens = input_tokens + output_tokens

        input_cost = (input_tokens / 1_000_000.0) * pricing["input_per_1m"]
        output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_1m"]
        total_cost = round(input_cost + output_cost, 6)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "model": model_key
        }


prompt_service = PromptService()
