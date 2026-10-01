# TODO: Unable to fetch Thinking Content
import os
import pathlib
from dotenv import load_dotenv
from typing import Any

import langchain_huggingface
from langchain_core.messages.ai import AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


def chat_model(
    model_name: str="Qwen/Qwen3-8B",
    task: str="text-generation",
    provider: str="nscale",
    temperature: float=1.0
):
    # Initialize the LLM endpoint
    llm = HuggingFaceEndpoint(
        repo_id=model_name, # Ensure this is a reasoning-capable model
        task=task,
        provider=provider,

        temperature=temperature,
        max_new_tokens=2048, # resoning + response

        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Wrap it in ChatHuggingFace and enable thinking
    model = ChatHuggingFace(
        llm=llm,
        chat_template_kwargs={
            # Name of this parameter might be different for other models (refer - HuggingFace Model Card)
            "enable_thinking": True
        }
    )

    return model

def generate_response(
    chat_model: langchain_huggingface.chat_models.ChatHuggingFace,
    query: str = "Hello, How are you today?"
) -> tuple[Any, str]:
    response_dict: AIMessage = chat_model.invoke(input=query)

    raw_content: str = response_dict.content
    return response_dict, raw_content

def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    load_dotenv(dotenv_path=parent_path / ".env")

    # Initializing model
    model = chat_model(model_name="Qwen/Qwen3-8B")

    # Generating response
    query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
    response_dict, raw_content = generate_response(chat_model=model, query=query)
    print(raw_content)


if __name__ == "__main__": 
    main()