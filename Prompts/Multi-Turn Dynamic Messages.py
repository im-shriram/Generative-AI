import pathlib
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.load import dumps

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chat_prompt_template = ChatPromptTemplate(
    messages = [
        ("system", """You are an experienced "{role}" and previously, you did a lot of research around "{technology}". Your task is to understand the user query and answer accordingly.""")
    ], # options: system, human, ai
    input_variables=["role", "technology"],
    validate_template=True
) # Use-case → Messages with Dynamic Prompts

prompt: PromptValue = chat_prompt_template.invoke(input={
    "role": "Generative AI Engineer",
    "technology": "Retrieval Augmented Generation (RAG)"
})

chat_history: list[BaseMessage] = prompt.to_messages() # Return prompt as a list of messages.
# NOTE: Since we are storing the entire chat history and passing to LLM might throw context-length overflow error

while(True):
    query: str = input("You: ")

    if query == "exit":
        break
    else:
        template = PromptTemplate(
            template="""{query}""",
            input_variables=["query"],
            validate_template=True
        ); user_query: PromptValue = template.invoke(input={"query": query})
        chat_history.append(user_query.to_messages()[0])

    response: AIMessage = llm.invoke(input=chat_history)
    print(f"AI: {response.content}")
    chat_history.append(AIMessage(content=response.content))

with open(file=pathlib.Path(__file__).parent / "Templates" / "chat history.json", mode='w') as fp:
    chat_history_json: str = dumps(obj=chat_history)
    fp.write(chat_history_json)