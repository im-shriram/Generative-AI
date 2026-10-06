from typing import Any
import pathlib

from .tasks import DocumentLoader, Embedding
from .components import Models

from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables import RunnableLambda

class Runnables:
    def __init__(
        self: Runnables,
        doc_loader_obj: DocumentLoader,
        embed_obj: Embedding
    ):

        self.doc_loader_obj = doc_loader_obj
        self.document_loader_runnable: RunnableLambda = RunnableLambda(
            func=self.loading_documents
        )
        self.text_splitter_runnable: RunnableLambda = RunnableLambda(
            func=self.splitting_texts
        )

        self.embed_obj = embed_obj
        self.embedding_document_runnable: RunnableLambda = RunnableLambda(
            func=self.embedding
        )
        self.store_embeddings_runnable: RunnableLambda = RunnableLambda(
            func=self.store_embeddings
        )

    def loading_documents(self, text: str="") -> str:
        return self.doc_loader_obj.load_documents(path=pathlib.Path(__file__).parent.parent / "data" / "knowledge base" / "knowledge_base.txt")

    def splitting_texts(self, doc: str) -> list[str]:
        return self.doc_loader_obj.split_documents(doc=doc)

    def embedding(self, docs: list[str]) -> list[list[float]]:
        return self.embed_obj.embed_docs(docs=docs)

    def store_embeddings(self, embeddings: list[list[float]]) -> None:
        self.embed_obj.store_embeddings(embeddings=embeddings, path=pathlib.Path(__file__).parent.parent / "data" / "embeddings" / "knowledge_base_embeddings.json")

class Pipeline:
    def __init__(
        self: Pipeline,
        obj: Runnables
    ):
        self.obj: Runnables = obj

    def build_embedding_knowledge_base_chain(self) -> RunnableSerializable:
        chain: RunnableSerializable[Any, Any] = self.obj.document_loader_runnable | self.obj.text_splitter_runnable | self.obj.embedding_document_runnable | self.obj.store_embeddings_runnable
        return chain

    def build_response_chain(self):
        pass

def main():
    runnable = Runnables(
        doc_loader_obj=DocumentLoader(),
        embed_obj=Embedding(model=Models().initialize_embedding_model())
    )
    pipeline: Pipeline = Pipeline(obj=runnable)
    pipeline.build_embedding_knowledge_base_chain().invoke(input="")

if __name__ == "__main__":
    main()