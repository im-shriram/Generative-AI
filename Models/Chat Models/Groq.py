# Reference: https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq
import os
import pathlib
from typing import Any
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from langchain_core.messages.ai import AIMessage
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.0, # It nither shrinks nor expands the probabilities

    max_tokens=2048, # Context Window (Reasoning + Response): 2048, None - no limit
    reasoning_effort="high",
    reasoning_format="parsed", # Crucial for accessing thinking content 

    groq_api_key=os.getenv("GROQ_API_TOKEN"),
    timeout=None, # The timeout for API requests in seconds. None uses the default
    max_retries=2, # Maximum number of retries for a failed request. Default is 2
)

query: str = "What is the difference between `Generative AI` and `Agentic AI`?"
response_dict: AIMessage = llm.invoke(input=query)
print(f"Complete Response: {response_dict} \n") # Returns an AIMessage with content and metadata, no reasoning content

content: str = response_dict.content
reasoning: Any | None = response_dict.additional_kwargs.get("reasoning_content")
print(f"Response: {content}", end="\n\n")
print(f"Reasioning Content: {reasoning}")