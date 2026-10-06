from langchain_core.messages.ai import AIMessage
import os
import pathlib
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq

class Models:
    def __init__(self):
        self.embedding_model: HuggingFaceEndpointEmbeddings | None = None
        self.chat_model: ChatGroq | None = None

    def initialize_embedding_model(
        self: Models,
        model_name: str="Qwen/Qwen3-Embedding-8B",
        embedding_dimensionality: int=32,
        task: str="feature-extraction",
        provider: str="scaleway",
    ) -> HuggingFaceEndpointEmbeddings:

        self.embedding_model = HuggingFaceEndpointEmbeddings(
            repo_id=model_name,
            task=task,
            provider=provider,

            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={
                # Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 32 to 4096
                "dimensions": embedding_dimensionality
            }
        )
        return self.embedding_model

    def initialize_chat_model(
        self: Models,
        model_name: str="openai/gpt-oss-120b",
        temperature: float=1
    ) -> ChatGroq:

        self.chat_model: ChatGroq = ChatGroq(
            model=model_name,
            temperature=temperature, 
            max_tokens=2048,
            reasoning_effort="high",
            reasoning_format="parsed", 
            api_key=os.getenv("GROQ_API_TOKEN"),
            timeout=None, 
            max_retries=2
        )
        return self.chat_model

def main():
    dotenv_path: pathlib.Path = pathlib.Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    models: Models = Models()
    embedding_model: HuggingFaceEndpointEmbeddings = models.initialize_embedding_model()
    chat_model: ChatGroq = models.initialize_chat_model()

    query: str = "Hi, How are you feeling today?"
    embeddings: list[float] = embedding_model.embed_query(text=query) # TODO: Why embedding models does not have invoke() method
    response_dict: AIMessage = chat_model.invoke(input=query)
    response: str = response_dict.content
    reasoning: str | None = response_dict.additional_kwargs.get("reasoning_content")

    print(f"Embeddings: {embeddings} \n")
    print(f"Response: {response} \n")
    if reasoning:
        print(f"Reasoning: {reasoning}")

if __name__ == "__main__":
    main()