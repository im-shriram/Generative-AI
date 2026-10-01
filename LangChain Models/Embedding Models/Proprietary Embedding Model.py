import os
import pathlib
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def embedding_model(
    model_name: str="gemini-embedding-2-preview",
    output_dimensionality: int=768
) -> GoogleGenerativeAIEmbeddings:
    # Initialize the embedding model endpoint
    model = GoogleGenerativeAIEmbeddings(
        model=model_name,
        output_dimensionality=output_dimensionality, # Suggested: 768, 1536, or 3072 (default)
        google_api_key=os.getenv(key="GOOGLE_API_KEY")
    )

    return model

def generate_embeddings(
    embedding_model: GoogleGenerativeAIEmbeddings,
    query: str="Hello, How are you today?"
) -> list[list[float]]:
    embeddings: list[list[float]] = embedding_model.embed_query(text=query) # embed_documents for batch_embedding
    return embeddings


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    load_dotenv(dotenv_path=parent_path / ".env")

    # Initializing model
    model: GoogleGenerativeAIEmbeddings = embedding_model(model_name="gemini-embedding-2-preview")

    # Generating response
    query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
    embeddings: list[list[float]] = generate_embeddings(embedding_model=model, query=query)
    print(embeddings)


if __name__ == "__main__": 
    main()