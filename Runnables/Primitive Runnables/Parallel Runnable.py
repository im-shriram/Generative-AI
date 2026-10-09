from typing import Any
import os
import pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent.parent / ".env")

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

twitter_template = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)
linkedin_template = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=1.5, # for creativity
    api_key=os.getenv("GROQ_API_TOKEN")
)

output_parser = StrOutputParser()

chain: RunnableParallel = RunnableParallel({
    'tweet': RunnableSequence(twitter_template, model, output_parser),
    'linkedin': RunnableSequence(linkedin_template, model, output_parser)
})

result: dict[str, Any] = chain.invoke(input={
    'topic':'AI'
}); print(result)