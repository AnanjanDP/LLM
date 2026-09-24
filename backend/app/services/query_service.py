import re
from typing import Dict, Any, List, Optional
from app.core.logging import logger


class QueryService:
    """Service for query optimization, cleaning, expansion, and unsupported question detection."""

    UNSUPPORTED_PATTERNS = [
        r"^\s*$",
        r"^(hi|hello|hey|test|abc|xyz|foo|bar)\s*$",
        r"^what\s+is\s+the\s+meaning\s+of\s+life\??$",
    ]

    def clean_query(self, query: str) -> str:
        """Sanitize query string by removing redundant whitespace and special artifacts."""
        if not query:
            return ""
        cleaned = re.sub(r"\s+", " ", query.strip())
        return cleaned

    def detect_unsupported_or_vague(self, query: str) -> Dict[str, Any]:
        """Detect vague or unsupported queries before executing context retrieval."""
        cleaned = self.clean_query(query)
        if len(cleaned) < 3:
            return {
                "is_vague": True,
                "reason": "Query is too short to extract meaningful semantic context.",
                "action": "prompt_user_clarification"
            }

        for pattern in self.UNSUPPORTED_PATTERNS:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return {
                    "is_vague": True,
                    "reason": "Query is generic or out of scope.",
                    "action": "use_fallback_prompt"
                }

        return {"is_vague": False, "reason": None, "action": "proceed"}

    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for expanded multi-query retrieval."""
        cleaned = self.clean_query(query)
        variations = [cleaned]

        # Simple heuristic keyword-focused expansion
        keywords = [w for w in re.findall(r"\w+", cleaned) if len(w) > 3]
        if keywords and len(keywords) < len(cleaned.split()):
            variations.append(" ".join(keywords))

        return variations


query_service = QueryService()
