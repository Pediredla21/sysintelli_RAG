from langchain_huggingface import HuggingFaceEmbeddings
from .config import settings

class EmbeddingService:
    _instance = None
    
    @classmethod
    def get_embeddings(cls) -> HuggingFaceEmbeddings:
        """
        Returns a singleton instance of the HuggingFaceEmbeddings model.
        This prevents reloading the model multiple times during runtime.
        """
        if cls._instance is None:
            cls._instance = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        return cls._instance

def get_embedding_model() -> HuggingFaceEmbeddings:
    return EmbeddingService.get_embeddings()
