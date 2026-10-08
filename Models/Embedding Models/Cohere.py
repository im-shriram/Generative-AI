# Reference: https://reference.langchain.com/python/langchain-cohere/embeddings/CohereEmbeddings
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from langchain_cohere import CohereEmbeddings

model = CohereEmbeddings(
    model="embed-english-light-v3.0",
    cohere_api_key=os.getenv(key="COHERE_API_KEY"),
    embedding_types=['float']
)

query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
embeddings: list[list[float]] = model.embed_query(text=query) # embed_documents for batch_embedding
print(embeddings)