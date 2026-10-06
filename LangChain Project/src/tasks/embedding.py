import json
import pathlib
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from ..components import Models
from langchain_huggingface import HuggingFaceEndpointEmbeddings

class Embedding:
    def __init__(self: Embedding, model: HuggingFaceEndpointEmbeddings) -> None:
        self.embedding_model: HuggingFaceEndpointEmbeddings = model
        self.query_embeddings: list[float] | None = None
        self.docs_embeddings: list[list[float]] | None = None

    def embed_query(self: Embedding, query: str):
        self.query_embeddings: list[float] = self.embedding_model.embed_query(text=query)
        return self.query_embeddings

    def embed_docs(self: Embedding, docs: list[str]):
        self.docs_embeddings: list[list[float]] = self.embedding_model.embed_documents(texts=docs)
        return self.docs_embeddings
    
    def store_embeddings(self, embeddings: list[list[float]], path: pathlib.Path) -> None:
        """ We are only storing embeddings of knowledge base in json format """
        with open(file=path, mode='w') as fp:
            json.dump(obj=embeddings, fp=fp)

def main():
    dotenv_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    models = Models()
    embedding_model = models.initialize_embedding_model()
    embedding_obj = Embedding(model=embedding_model)

    query: str = "Hi, How are you feeling today?"
    docs: list[str] = [
        "Hello, there",
        "How are you?",
        "Its nice to meet you."
    ]
    query_embeddings: list[float] = embedding_obj.embed_query(query=query)
    docs_embeddings: list[list[float]] = embedding_obj.embed_docs(docs=docs)

    print(f"Query Embedding: {query_embeddings} \n")
    print(f"Document Embedding: {docs_embeddings} \n")

    store_path: pathlib.Path = pathlib.Path(__file__).parent.parent.parent / "data" / "embeddings" / "knowledge_base_embeddings.json"
    embedding_obj.store_embeddings(embeddings=docs_embeddings, path=store_path)

if __name__ == "__main__":
    main()