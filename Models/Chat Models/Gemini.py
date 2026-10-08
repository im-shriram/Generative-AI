# Reference: https://reference.langchain.com/python/langchain-google-genai/chat_models/ChatGoogleGenerativeAI
from typing import Any
import pathlib
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from langchain_core.messages.ai import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

chat_model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1, # Reference: https://www.youtube.com/watch?v=hqiSmC3-Ktw
    max_tokens=2048,
    reasoning_effort="low", # 'minimal', 'low', 'medium', 'high'
    # thinking_level="low" - Alias for reasoning_effort
    include_thoughts=True, # Required to see the reasoning in the output
)

response_dict: AIMessage = chat_model.invoke(
    input="Write a short poem on 'Cricket'",
    automatic_function_calling={"disable": True}
); print(f"Complete Response: {response_dict} \n") # Returns a List of dictonaries, one for reasoning and another for output

content: str = response_dict.content[1]["text"]
reasoning: str = response_dict.content[0]["thinking"]

print(f"Response: {content}", end="\n")
print(f"Reasioning Content: {reasoning} \n")