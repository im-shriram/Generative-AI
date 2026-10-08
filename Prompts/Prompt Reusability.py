import pathlib
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.load import (
    dumps, # converts to JSON
    dumpd, # conterts to python dictonary
    load, # loads JSON object - The default allow list includes core LangChain types (messages, prompts, documents, etc.)
    loads # loads JSON object
)
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

system_prompt: str = """
Task: Please summarize the research paper titled as "{paper_input}" with the following
    specifications:
        Explanation Style: {style_input}
        Explanation Length: {length_input} [Optional if not None]

    Response Content:
        1. Mathematical Details:
            Include relevant mathematical equations if present in the paper.
            Explain the mathematical concepts using simple, intuitive code snippets where applicable.
        2. Analogies:
            Use relatable analogies to simplify complex ideas.
            If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

        Instructions: Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
template = PromptTemplate(
    template=system_prompt,
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)
 
prompt_template_json: str = dumps(obj=template, pretty=True, indent=4) # Return a "JSON string" representation of an object.
prompt_template_dictonary = dumpd(obj=template) # Return a dict representation of an object.

with open(file=pathlib.Path(__file__).parent / "Templates" / "prompt template.json", mode='w') as fp:
    fp.write(prompt_template_json)

with open(file=pathlib.Path(__file__).parent / "Templates" / "prompt template.json", mode='r') as fp:
    template = loads(fp.read(), allowed_objects="core") # "allowed_objects": "core" → deserialized only (messages, prompts, documents, etc.)

paper_input: str = input("Enter the name of the research paper: ")
style_input: str = input("Enter the response style: ")
length_input: str = input("Enter the response length [optional]: ")

prompt: PromptValue = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_format="parsed",
    max_tokens=None
)

response: AIMessage = llm.invoke(input=prompt)
print(response.content)