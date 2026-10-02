import pathlib
import os
import json
from dotenv import load_dotenv

from langchain_core.messages.base import BaseMessage
import langchain_core.messages.ai
import langchain_core.messages.human
from langchain_core.prompt_values import PromptValue
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import messages_to_dict, messages_from_dict
from langchain_core.load import dumps

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

# Creating prompt tenplate
chat_prompt_template = ChatPromptTemplate(
    messages = [
        ("system", "You are an experienced {role} and previously, you did a lot of research around {technology}"),
        MessagesPlaceholder(variable_name="chat_history"),
    ],
    input_variables=["role", "technology", "chat_history"],
    validate_template=True
)

# Saving demo chat history
history: list[langchain_core.messages.ai.AIMessage | langchain_core.messages.human.HumanMessage] = [
    HumanMessage(content="Include 'Machine Learning' and 'Deep Learning' reference..."),
    AIMessage(content="Sure, just tell me the 'Libraries and Frameworks'..."),
]; print(messages_to_dict(history))

with open("LangChain Prompts/Saved Prompt Templates/chat history.json", "r+") as fp:
    json.dump(messages_to_dict(history), fp, indent=4) # Convert a sequence of Messages to a list of dictionaries

# Loading the demo chat
with open("LangChain Prompts/Saved Prompt Templates/chat history.json", "r") as fp:
    data = json.load(fp) # history

previous_chat_history: list[BaseMessage] = messages_from_dict(data) # Convert a list of dicts (holds type and content) to list of messages [HumanMessage(), AiMessage()...]
print(previous_chat_history)

# Creating prompt
prompt: PromptValue = chat_prompt_template.invoke(input={
    "role": "Generative AI Engineer",
    "technology": "RAG",
    "chat_history": previous_chat_history
})

# Maintaining chat history
chat_history: list[SystemMessage, HumanMessage, AIMessage] = prompt.to_messages() # NOTE: Current chat history has both previous chat history and current conversactions, so you dont need to explicitely update the chat history
print(chat_history, end="\n\n")

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

# Storing new chat history
with open(file="LangChain Prompts/Saved Prompt Templates/chat history.json", mode="w") as fp:
    json_chat_history: str = dumps(obj=chat_history, indent=4)
    fp.write(json_chat_history)
    print(chat_history)