import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / ".env")

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

joke_template = PromptTemplate(
    template="Generate a joke on given topic - {topic}",
    input_variables=["topic"],
    validate_template=True
)
explain_template = PromptTemplate(
    template="Explain the provided joke in simple terms - {joke}",
    input_variables=["joke"],
    validate_template=True
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.5, # for creativity
    api_key=os.getenv("GROQ_API_TOKEN")
)

output_parser = StrOutputParser()

chain: RunnableSequence = RunnableSequence(
    joke_template, model, output_parser, explain_template, model, output_parser
)
"""
    explain_template expect a input "joke". Since output_parser [1st call] provides plain text which is as it is passed to explain_template, since explain_template requires only one input it does not matter whether you pass a dictonary or a string it will be as it is passed into it, but if explain_template requires more than one input, it will only work when your previous component pass a dictonary having required items.
        Execution → Chains/Conditional Chain.py 
"""

print(chain.invoke(
    input={
        "topic": "Mathematics"
    }
))