# Importing necessary libraries
from langchain_core.messages.ai import AIMessage
from typing import Any
import os
import pathlib
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings


class Models:
    def __init__(self):
        pass

    def embedding_model(
        self: Models,
        model_name: str="Qwen/Qwen3-Embedding-8B",
        embedding_dimensionality: int=512,
        task: str="feature-extraction",
        provider: str="scaleway",
    ) -> HuggingFaceEndpointEmbeddings:

        # Initialize the embedding model endpoint
        model = HuggingFaceEndpointEmbeddings(
            repo_id=model_name, # Ensure this is a reasoning-capable model
            task=task,
            provider=provider,

            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={
                # Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 32 to 4096
                "dimensions": embedding_dimensionality
            }
        )
        return model

    def chat_model(
        self: Models,
        model_name: str="gemini-3-flash-preview",
        temperature: float=1
    ) -> ChatGoogleGenerativeAI:

        chat_model = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_tokens=2048,
            thinking_level="low", # For Gemini 3: 'minimal', 'low', 'medium', 'high'
            include_thoughts=True, # REQUIRED to see the reasoning in the output
        )
        return chat_model


def embed_documents(model: HuggingFaceEndpointEmbeddings, docs: list[str] | str) -> list[float] | list[list[float]]:
    if type(docs) == str:
        # Embedding Query
        return model.embed_query(text=docs)
    else:
        # Embedding Knowledge Base
        return model.embed_documents(texts=docs)

def select_top_k(embedded_query: list[float], embedded_knowledge_base: list[list[float]], top_k: int=5) -> list[tuple[int, float]]:
    similarity_scores: list[tuple[int, float]] = []

    for idx, vector in enumerate(embedded_knowledge_base):
        similarity_scores.append((idx, sum([x * y for x, y in zip(embedded_query, vector)])))

    similarity_scores.sort(reverse=True, key=lambda a: a[1]) # sorting decending wrt similarity scores
    return similarity_scores[:top_k]

def generate_response(model: ChatGoogleGenerativeAI, prompt: str) -> tuple[dict[Any, Any], str, str]:
    response_dict: AIMessage = model.invoke(
        input=prompt,
        automatic_function_calling={"disable": True}
    )

    content: str = response_dict.content[1]["text"]
    reasoning: str = response_dict.content[0]["thinking"]

    return response_dict, content, reasoning


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent
    dotenv_path: pathlib.Path = parent_path / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    # Fetching knowledge-base
    with open(file=parent_path / "data" / "knowledge_base.txt") as fp:
        data: str = fp.read()
    knowledge_base: list[str] = data.split(sep=". ")

    # Taking user query as input
    query: str = input("Enter your query: ")

    # Initializing embedding model and chat-model
    obj: Models = Models()
    embedding_model: HuggingFaceEndpointEmbeddings = obj.embedding_model(
        model_name="Qwen/Qwen3-Embedding-8B",
        embedding_dimensionality=512
    )
    chat_model: ChatGoogleGenerativeAI = obj.chat_model(
        model_name="gemini-3-flash-preview",
        temperature=1.0
    )

    # Embedding user query and knowledge-base
    embedded_query: list[float] | list[list[float]] = embed_documents(model=embedding_model, docs=query)
    embedded_knowledge_base: list[float] | list[list[float]] = embed_documents(model=embedding_model, docs=knowledge_base)

    """
        print(f"Embedded Query: {embedded_query[:5]} \n\n")
        print(f"Embedded KB: {embedded_knowledge_base}")
    """

    # Retrieving top 25 documents from knowledge base
    similarity_scores: list[tuple[int, float]] = select_top_k(embedded_query=embedded_query, embedded_knowledge_base=embedded_knowledge_base, top_k=25)
    relevant_docs: list[str] = []
    for idx, score in similarity_scores:
        relevant_docs.append(knowledge_base[idx])
    print(f"Relevant Documents: {relevant_docs}", end="\n\n")

    # Creating detailed prompt
    prompt: str = f"Provide a detailed response to the user's query: {query} using only the information from the provided context {relevant_docs}. If the context does not contain sufficient information to answer the question, reply exactly with: 'Sorry, I don't have enough information to answer your question."

    # Generating response
    response_dict, content, reasoning = generate_response(model=chat_model, prompt=prompt)
    print(f"Response: {content}", end="\n\n")
    print(f"Reasioning Content: {reasoning}")


if __name__ == "__main__":
    main()