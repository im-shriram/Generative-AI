# Importing necessary libraries
import pathlib
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_groq import ChatGroq
from langchain_core.messages.base import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chat_history: list[BaseMessage] = [
    SystemMessage(content="You are an experienced 'Science Teacher' and previously, you did a lot of research around 'Elements in Periodic Table'")
]

while(True):
    question: str = input("You: ")
    if question == "exit":
        break
    prompt = HumanMessage(content=question)
    chat_history.append(prompt)

    response: AIMessage = llm.invoke(input=chat_history)
    print(f"AI: {response.content}")
    chat_history.append(AIMessage(content=response.content))