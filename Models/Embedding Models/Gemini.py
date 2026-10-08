# Reference: https://reference.langchain.com/python/langchain-google-genai/embeddings/GoogleGenerativeAIEmbeddings
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    output_dimensionality=768, # Suggested: 768, 1536, or 3072 (default)
    task_type='RETRIEVAL_QUERY', # RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY ... 
    google_api_key=os.getenv(key="GOOGLE_API_KEY")
)

query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
embeddings: list[list[float]] = model.embed_query(text=query) # embed_documents for batch_embedding
print(embeddings)

# NOTE: While gemini-embedding-2-preview natively supports multimodal inputs (text, images, video, audio, and PDFs) via the Google GenAI SDK, the LangChain Embeddings interface (embed_query / embed_documents) currently only accepts text.