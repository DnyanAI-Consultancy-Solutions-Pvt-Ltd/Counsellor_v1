from langchain_groq import ChatGroq
from config.settings import get_settings


def get_llm(*, fast: bool = False, temperature: float = 0.0, max_tokens: int = 8000) -> ChatGroq:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing in .env")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_fast_model if fast else settings.groq_model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )
