import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import PromptValue
from langchain_core.messages.ai import AIMessage

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1,
    max_tokens=2048
)

# String output parser returns only the content part of the response instead of AIMessage that contains content and metadata
output_parser = StrOutputParser()

template = PromptTemplate(
    template="""Summarize the topic - {topic} in 5 lines""",
    input_variables=['topic'],
    # partial_variables={"format_instructions": output_parser.get_format_instructions()} → This is manindatory argument for JSON and Pydantic output parsers you have to provide. It simply returns "Return JSON object. Schema ...". But for string parser it is not implemented
)

prompt: PromptValue = template.invoke(input={"topic": "Transformers"})
response: AIMessage = model.invoke(input=prompt)
result: str = output_parser.invoke(input=response)
print(result)

# NOTE: Output Parsers work with both LLMs that provide built-in structured output and the LLMs that don't