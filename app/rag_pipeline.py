import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vector_store import load_vector_store
from .llm import get_llm
from .config import settings

logger = logging.getLogger(__name__)

def format_docs(docs):
    """
    Combines the page content of multiple documents with clear separators.
    """
    formatted = []
    for i, doc in enumerate(docs):
        page_num = doc.metadata.get('page', 'N/A')
        formatted.append(f"--- Source Chunk {i+1} (Page {page_num}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)

def answer_question(question: str) -> dict:
    """
    Core RAG pipeline:
    1. Loads the vector store.
    2. Retrieves top relevant chunks.
    3. Formats prompt and invokes LLM.
    Returns the generated answer along with the source chunks.
    """
    vector_store = load_vector_store()
    if not vector_store:
        logger.warning("Attempted to ask question without initialized vector DB.")
        return {
            "answer": "Error: Vector database not initialized. Please upload a PDF first.", 
            "sources": []
        }
    
    # Fetch relevant chunks using similarity search directly for more control
    logger.info(f"Retrieving top {settings.RETRIEVER_K} chunks for question: {question}")
    retrieved_docs = vector_store.similarity_search(question, k=settings.RETRIEVER_K)
    
    if not retrieved_docs:
        return {
            "answer": "Information not found in document",
            "sources": []
        }
        
    context_text = format_docs(retrieved_docs)
    
    # Improved prompt template that forces comprehensive answers
    template = """You are a highly detailed AI legal assistant. You must answer questions using ONLY the provided context from legal documents.

INSTRUCTIONS:
1. Read ALL the context chunks carefully and thoroughly.
2. Provide a COMPLETE and DETAILED answer that includes ALL relevant information found in the context.
3. Do NOT summarize or shorten the answer. Include every relevant detail, list item, law name, regulation, date, and fact mentioned in the context.
4. If the context contains a list of items (like laws, rules, regulations, or points), you MUST include ALL items from the list, not just the first few.
5. Format your answer clearly using numbered lists or bullet points when the context contains multiple items.
6. If the context does not contain the answer at all, respond EXACTLY with: "Information not found in document"
7. Do NOT use any external knowledge. Only use what is explicitly stated in the context below.

Context from the document:
{context}

Question: {question}

Provide a complete and detailed answer:"""

    prompt = PromptTemplate.from_template(template)
    llm = get_llm()
    
    # Using LCEL to create the generation chain
    rag_chain = prompt | llm | StrOutputParser()
    
    try:
        logger.info("Invoking LLM for answer generation...")
        response = rag_chain.invoke({
            "context": context_text, 
            "question": question
        })
        logger.info("Successfully generated answer.")
    except Exception as e:
        logger.error(f"Error invoking LLM: {str(e)}")
        return {
            "answer": f"Error generating answer from LLM: {str(e)}",
            "sources": retrieved_docs
        }
    
    return {
        "answer": response.strip(), 
        "sources": retrieved_docs
    }
