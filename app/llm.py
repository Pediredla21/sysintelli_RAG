from langchain_community.llms import Ollama
from .config import settings

def get_llm() -> Ollama:
    """
    Initializes and returns the Ollama LLM instance based on settings.
    Temperature is 0 to ensure factual responses.
    num_predict is set high to prevent truncated answers.
    """
    llm = Ollama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.0,
        num_predict=2048
    )
    return llm
