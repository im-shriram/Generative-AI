from langchain_core.runnables.base import RunnableSerializable
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env")

from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# Creating model endpoint for communication
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=5000, # The above model is a thinking model so the default value of "max_new_tokens" may get exhausted for thinking tokens
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
output_parser = JsonOutputParser(pydantic_object=Schema) # StructuredOutputParser is depricated, you can pass the schema directly inside JSON

# Template
template = PromptTemplate(
    template="Write a detailed report on {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={"format_instruction": output_parser.get_format_instructions()} # This variable "format_instruction" is filled at compile time → Returns: Return JSON object
); print(f"Prompt: {template.invoke(input={"topic": "Mixture of Experts"})}")

# Initializing a chain
chain: RunnableSerializable[dict[str, Any], str] = template | model | output_parser
result: str = chain.invoke(input={"topic": "Mixture of Experts"})
print(result)