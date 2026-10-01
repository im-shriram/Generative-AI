# Importing necessary libraries
from typing import Any
import pathlib
import os
from dotenv import load_dotenv

from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def large_language_model(
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

def generate_response(
    chat_model: ChatGoogleGenerativeAI,
    query: str = "Hello, How are you today?"
) -> tuple[Any, str, str]:
    response_dict: AIMessage = chat_model.invoke(
        input=query,
        automatic_function_calling={"disable": True}
    )

    content: str = response_dict.content[1]["text"]
    reasoning: str = response_dict.content[0]["thinking"]

    return response_dict, content, reasoning


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    load_dotenv(dotenv_path=parent_path / ".env")

    # Building model instance
    chat_model: ChatGoogleGenerativeAI = large_language_model(model_name="gemini-3-flash-preview", temperature=1)

    # Generating response
    query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
    response_dict, content, reasoning = generate_response(chat_model=chat_model, query=query)

    print(f"Response: {content}", end="\n\n")
    print(f"Reasioning Content: {reasoning}")


if __name__ == "__main__": 
    main()