import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.base import RunnableSerializable


# Creating model endpoint for communication
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=5000,
    temperature=0.5,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
); model = ChatHuggingFace(llm=llm)

class Schema(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "net"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    sentiment_probability: float = Field(gt=0.0, lt=1.0, default=0.5, description="Return a probability of positive sentement")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Initializing output parser
output_parser: PydanticOutputParser[Any] = PydanticOutputParser(pydantic_object=Schema)

# Template
template = PromptTemplate(
    template="Write a detailed report on {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction": output_parser.get_format_instructions()} # This variable "format_instruction" is filled at compile time → Returns: Return Pydantic object
); print(f"Prompt: {template.invoke(input={"topic": "Black Holes"})}")

# Initializing a chain
chain: RunnableSerializable[dict[str, Any], str] = template | model | output_parser
result: str = chain.invoke(input={"topic": "Black Holes"})
print(result, type(result))


"""
    How Output Parsers work?
    →
        1. The critical mechanism is prompt injection. A parser like PydanticOutputParser or JsonOutputParser exposes a method called get_format_instructions() that returns a string describing the desired schema. You inject this string into your prompt via partial_variables, so the LLM sees instructions like:
            "Return a JSON object with keys name (string), age (integer), and email (string)."
        2. The LLM then attempts to comply based on its training and the prompt instructions. When the response (string of json or pudantic object) arrives, the parser runs its parse() method on the raw text — splitting, regex-matching, JSON-decoding, and validating against the schema.
        3. If the LLM deviates (adds conversational text, forgets a field, produces malformed JSON), the parser raises an OutputParserException
"""