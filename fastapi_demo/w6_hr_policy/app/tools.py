"""Read-only tool schema and implementation for the synthetic handbook."""
from .rag import CATEGORIES, pack, retrieve

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_policy",
        "description": "按关键词检索当前合成制度。只读，不改变适用范围。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 160},
                "category": {"type": "string", "enum": list(CATEGORIES)}
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}]

async def search_policy(query, category="其他"):
    return pack(await retrieve(query, top_k=3, category=category))

IMPL = {"search_policy": search_policy}
