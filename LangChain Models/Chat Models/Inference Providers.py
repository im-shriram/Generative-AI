# Importing necessary libraries
from typing import Any
import langchain_groq.chat_models
from langchain_core.messages.ai import AIMessage
from langchain_groq import ChatGroq
import pathlib
import os
from dotenv import load_dotenv


def large_language_model(
    model_name: str="openai/gpt-oss-120b",
    temperature: float=1
) -> langchain_groq.chat_models.ChatGroq:
    llm = ChatGroq(
        model=model_name,
        temperature=temperature, # recommended for reasoning phase

        max_tokens=2048, # Context Window (Reasoning + Response): 2048, None - no limit
        reasoning_effort="high",
        reasoning_format="parsed", # Crucial for accessing thinking content 

        api_key=os.getenv("GROQ_API_TOKEN"),
        timeout=None, # The timeout for API requests in seconds. None uses the default
        max_retries=2, # Maximum number of retries for a failed request. Default is 2
    )

    return llm

def generate_response(
    llm: langchain_groq.chat_models.ChatGroq,
    query: str = "Hello, How are you today?"
) -> tuple[Any, str, str]:
    response_dict: AIMessage = llm.invoke(input=query)

    content: str = response_dict.content
    reasoning: Any | None = response_dict.additional_kwargs.get("reasoning_content")

    if reasoning:
        return response_dict, content, reasoning
    return response_dict, content, ""


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    load_dotenv(dotenv_path=parent_path / ".env")

    # Building model instance
    llm: langchain_groq.chat_models.ChatGroq = large_language_model(model_name="openai/gpt-oss-120b", temperature=1)

    # Generating response
    query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
    response_dict, content, reasoning = generate_response(llm=llm, query=query)

    print(f"Response: {content}", end="\n\n")
    print(f"Reasioning Content: {reasoning}")


if __name__ == "__main__": 
    main()
