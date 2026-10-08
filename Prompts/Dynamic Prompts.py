import pathlib
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_core.load import dumps
from langchain_core.messages.ai import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=os.getenv("GROQ_API_TOKEN"),
    reasoning_effort="low",
    reasoning_format="parsed"
)

system_prompt: str = """
Task: Summarize the research paper titled as "{paper_input}" with the following
    specifications:
        Explanation Style: {style_input}
        Explanation Length [Optional]: {length_input}

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
    input_variables=["paper_input", "style_input", "length_input"], # Names mentioned here, in the template and in invoke function must be same
    input_types={
        "paper_input": str,
        "style_input": str,
        "length_input": str
    },
    validate_template=True # Checks all the required variables to pass in the prompt, if not then throws an error
)

template_json: str = dumps(obj=template)
with open(file=pathlib.Path(__file__).parent / "Templates" / "prompt template.json", mode='w') as fp:
    fp.write(template_json)

paper_input: str = input("Enter the name of the research papar: ")
style_input: str = input("Enter the response style: ")
length_input: str = input("Enter the response length [optional]: ")

prompt: PromptValue = template.format(
    # PromptTemplate is a runnable so you can use invoke method instead of format
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input,
)
print(prompt, "\n") # This automatically returns a HumanMessage

response: AIMessage = llm.invoke(input=prompt)
print(response.content)