# Importing necessary libraries
from langchain_core.prompt_values import PromptValue
import pathlib
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

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
        ("system", """You are an experienced "{role}" and previously, you did a lot of research around "{technology}". Your task is to understand the user query and answer accordingly only if that query related to the things inside role and technology. If not then just reply with - I'm unable to resolve your query.""")
    ],
    input_variables=["role", "technology"],
    validate_template=True
) # Use-case → Messages with Dynamic Prompts

# Creating prompt
prompt: PromptValue = chat_prompt_template.invoke(input={
    "role": "Generative AI Engineer",
    "technology": "Retrieval Augmented Generation (RAG)"
})

# Maintaining chat history
chat_history: list[SystemMessage, HumanMessage, AIMessage] = prompt.to_messages() # Return prompt as a list of messages.

# AI Conversation
while(True):
    # Human Message → You can either use Static Prompt or Dynamic Prompt (recommended)
    question: str = input("You: ")
    if question == "exit":
        break
    else:
        template = PromptTemplate(
            template="""{question}""",
            input_variables=["question"],
            validate_template=True
        ); user_query: PromptValue = template.invoke(input={"question": question})
        chat_history.append(user_query.to_messages()[0]) # PromptTemplate().to_messages by-default returns HumanMessage

    # AI Message
    response: AIMessage = llm.invoke(input=chat_history)
    print(f"AI: {response.content}")
    chat_history.append(AIMessage(content=response.content))

print(chat_history)