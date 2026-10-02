# Importing necessary libraries
import pathlib
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Loading environment variables
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Maintaining the role based chat history 
chat_history: list[SystemMessage, HumanMessage, AIMessage] = [
    SystemMessage(content="You are an experienced 'Science Teacher' and previously, you did a lot of research around 'Elements in Periodic Table'")
]

# AI Conversation
while(True):
    # Human Message
    question: str = input("You: ")
    if question == "exit":
        break
    prompt = HumanMessage(content=question)
    chat_history.append(prompt)

    # AI Message
    response: AIMessage = llm.invoke(input=chat_history)
    print(f"AI: {response.content}")
    chat_history.append(AIMessage(content=response.content))

# Store in database and use it when user starts re-interacting with the system Use → Message Placeholders
print(chat_history)