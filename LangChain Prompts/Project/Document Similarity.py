# Importing necessary libraries
from langchain_core.prompt_values import PromptValue
from io import open_code
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.ai import AIMessage
from typing import Any
import os
import pathlib
from dotenv import load_dotenv
import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.load import dumps
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, messages_to_dict, messages_from_dict


class Models:
    def __init__(self):
        pass

    def embedding_model(
        self: Models,
        model_name: str="Qwen/Qwen3-Embedding-8B",
        embedding_dimensionality: int=512,
        task: str="feature-extraction",
        provider: str="scaleway",
    ) -> HuggingFaceEndpointEmbeddings:

        # Initialize the embedding model endpoint
        model = HuggingFaceEndpointEmbeddings(
            repo_id=model_name, # Ensure this is a reasoning-capable model
            task=task,
            provider=provider,

            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            model_kwargs={
                # Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 32 to 4096
                "dimensions": embedding_dimensionality
            }
        )
        return model

    def chat_model(
        self: Models,
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


def embed_documents(model: HuggingFaceEndpointEmbeddings, docs: list[str] | str) -> list[float] | list[list[float]]:
    if type(docs) == str:
        # Embedding Query
        return model.embed_query(text=docs)
    else:
        # Embedding Knowledge Base
        return model.embed_documents(texts=docs)

def select_top_k(embedded_query: list[float], embedded_knowledge_base: list[list[float]], top_k: int=5) -> list[tuple[int, float]]:
    similarity_scores: list[tuple[int, float]] = []

    for idx, vector in enumerate(embedded_knowledge_base):
        similarity_scores.append((idx, sum([x * y for x, y in zip(embedded_query, vector)])))

    similarity_scores.sort(reverse=True, key=lambda a: a[1]) # sorting decending wrt similarity scores
    return similarity_scores[:top_k]

def build_prompt() -> tuple[PromptValue, ChatPromptTemplate, list[BaseMessage]]:
    # Fetching previous chat history
    with open(file="LangChain Prompts/Project/data/chat_history.json", mode="r") as fp:
        try:
            previous_chat_history: list[BaseMessage] = messages_from_dict(json.load(fp))
        except Exception as e:
            print("Starting Fresh")
            previous_chat_history = []

    # Create and save a Prompt Template
    prompt_template = ChatPromptTemplate(
        messages=[
            ("system", """Provide a detailed response to the user query's throughout conversation using only the information from the provided context along with each query. Also consider the previous conversations (the questions user have asked and response you have given) and use it to provide the answer if relevent to the current query. If the context does not contain sufficient information to answer the question, reply exactly with: 'Sorry, I don't have enough information to answer your question."""),
            MessagesPlaceholder(variable_name="previous_chat_history")
        ],
        input_variables=["previous_chat_history"],
        validate_template=True
    )
    prompt_template_json: str = dumps(obj=prompt_template, pretty=True, indent=4)
    with open(file="LangChain Prompts/Project/data/prompt_template.json", mode="w") as fp:
        fp.write(prompt_template_json)

    # Invoking the prompt template
    prompt: PromptValue = prompt_template.invoke(
        input={
            "previous_chat_history": previous_chat_history
        }
    )

    # Building current chat history
    current_chat_history: list[BaseMessage] = []
    current_chat_history.extend(prompt.to_messages())
    """
        This step is crucial. We load the previous conversations from storage and convert them into a list of message objects. After that, MessagePlaceholder injects them into the prompt in this format:
            messages = [
                ("system", "some text"),
                ("human", "previous user message"),
                ("ai", "previous assistant reply")
            ]
        It recursively reconstructs the conversation history and inserts it into the prompt, preserving the context for the model.
    """
    return prompt, prompt_template, current_chat_history

def generate_response(model: ChatGoogleGenerativeAI, prompt: list[BaseMessage]) -> tuple[dict[Any, Any], str, str]:
    response_dict: AIMessage = model.invoke(
        input=prompt,
        automatic_function_calling={"disable": True}
    )

    # NOTE: Somethims model do not think and does not return the reasoning content so below check handle this
    if len(response_dict.content) > 1:
        # Thinking content available
        content: str = response_dict.content[1]["text"]
        reasoning: str = response_dict.content[0]["thinking"]
    else:
        content: str = response_dict.content[0]["text"]
        reasoning: str = ""

    return response_dict, content, reasoning


def main() -> None:
    # Loading environment variables
    parent_path: pathlib.Path = pathlib.Path(__file__).parent.parent
    dotenv_path: pathlib.Path = parent_path / ".env"
    load_dotenv(dotenv_path=dotenv_path)

    # Fetching knowledge-base
    with open(file=parent_path / "Project" / "data" / "knowledge_base.txt") as fp:
        data: str = fp.read()
    knowledge_base: list[str] = data.split(sep=". ")

    # Initializing embedding model and chat-model
    obj: Models = Models()
    embedding_model: HuggingFaceEndpointEmbeddings = obj.embedding_model(
        model_name="Qwen/Qwen3-Embedding-8B",
        embedding_dimensionality=512
    )
    chat_model: ChatGoogleGenerativeAI = obj.chat_model(
        model_name="gemini-3-flash-preview",
        temperature=1.0
    )

    # Embedding user query and knowledge-base
    embedded_knowledge_base: list[float] | list[list[float]] = embed_documents(model=embedding_model, docs=knowledge_base)

    """
        print(f"Embedded Query: {embedded_query[:5]} \n\n")
        print(f"Embedded KB: {embedded_knowledge_base}")
    """

    # Creating detailed prompt
    prompt, _, current_chat_history = build_prompt()

    # Prompt Template for each user's query
    prompt_template = PromptTemplate(
        template="""
            User Query: {query},
            Similer Documents: {similer_docs}
        """,
        input_variables=["query", "similer_docs"]
    )

    # Generating response
    while True:
        # Taking user query as input and performing embedding
        query: str = input("Enter your query: ")
        if query.lower() == "exit":
            print("System: Thanks for interacting with us. See you next time")
            break

        else:
            embedded_query: list[float] | list[list[float]] = embed_documents(model=embedding_model, docs=query)

            # Retrieving top 25 documents from knowledge base
            similarity_scores: list[tuple[int, float]] = select_top_k(embedded_query=embedded_query, embedded_knowledge_base=embedded_knowledge_base, top_k=25)
            relevant_docs: list[str] = []
            for idx, score in similarity_scores:
                relevant_docs.append(knowledge_base[idx])
        
            current_prompt: PromptValue = prompt_template.invoke(
                input={
                    "query": query,
                    "similer_docs": relevant_docs
                }
            )
            current_chat_history.extend(current_prompt.to_messages())
            response_dict, content, reasoning = generate_response(model=chat_model, prompt=current_chat_history)
            
            print(f"Response: {content}", end="\n\n")
            print(f"Reasioning Content: {reasoning}")
            current_chat_history.append(AIMessage(content=content))

    # Updating chat history
    with open(file="LangChain Prompts/Project/data/chat_history.json", mode="w") as fp:
        json.dump(messages_to_dict(current_chat_history), fp, indent=4)


if __name__ == "__main__":
    main()