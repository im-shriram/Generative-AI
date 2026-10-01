# TODO: Unable to fetch Thinking Content
import os
import pathlib
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings


def embedding_model(
    model_name: str="Qwen/Qwen3-Embedding-8B",
    task: str="feature-extraction",
    provider: str="scaleway",
):
    # Initialize the embedding model endpoint
    model = HuggingFaceEndpointEmbeddings(
        repo_id=model_name, # Ensure this is a reasoning-capable model
        task=task,
        provider=provider,

        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={
            # Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 32 to 4096
            "dimensions": 4096
        }
    )

    return model

def generate_embeddings(
    embedding_model: HuggingFaceEndpointEmbeddings,
    query: list[str] |str = ["Hello, How are you today?"]
) -> list[list[float]]:
    embeddings: list[list[float]] = embedding_model.embed_documents(texts=[query])
    return embeddings


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    load_dotenv(dotenv_path=parent_path / ".env")

    # Initializing model
    model = embedding_model(model_name="Qwen/Qwen3-Embedding-8B")

    # Generating response
    query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
    embeddings: list[list[float]] = generate_embeddings(embedding_model=model, query=query)
    print(embeddings)


if __name__ == "__main__": 
    main()