from typing import Any
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

from langchain_core.runnables.base import RunnableSerializable
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import PromptValue


# Creating model endpoint for communication
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    provider="featherless-ai",
    temperature=0.5,
    max_new_tokens=5000,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
); model = ChatHuggingFace(llm=llm)

# Initializing output parser
output_parser = StrOutputParser()

# # Template
template = PromptTemplate(
    template="""
        Write a detailed report on "{topic}".
        Output format instructions: {format_instructions}
    """,
    input_variables=['topic'],
    # partial_variables={"format_instructions": output_parser.get_format_instructions()} → This is manindatory argument for JSON and Pydantic output parseers you have to provide. It simply returns "Return JSON object". But for string parser it is not implemented
)

prompt: PromptValue = template.invoke(input={"topic": "Transformers"})
print(f"Prompt: {prompt}", end="\n\n")

response: AIMessage = model.invoke(input=prompt)
print(f" Response: {response}", end="\n\n")

result: str = output_parser.invoke(input=response)
print(result)

# Chain for above activities
def build_chain():
    chain: RunnableSerializable[dict[str, Any], str] = template | model | output_parser
    response: str = chain.invoke(
        input={
            "topic": "Transformers"
        }
    )
    return response